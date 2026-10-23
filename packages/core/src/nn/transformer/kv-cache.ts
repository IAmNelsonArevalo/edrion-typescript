import { TransformerError } from '@core/nn/transformer/transformer.errors';
import { Tensor } from '@core/tensor/tensor';

export interface TransformerLayerKeyValueCache {
    readonly key: Tensor;
    readonly value: Tensor;
}

/** Mutable inference cache containing projected K/V tensors for every Transformer layer. */
export class TransformerKeyValueCache {
    private readonly layers: Array<TransformerLayerKeyValueCache | undefined>;
    private currentSequenceLength: number = 0;

    public constructor(
        public readonly numberOfLayers: number,
        public readonly contextLength: number,
        public readonly backend: string,
        public readonly batchSize: number,
    ) {
        if (!Number.isInteger(numberOfLayers) || numberOfLayers <= 0) {
            throw new TransformerError(`KV cache numberOfLayers must be positive. Received ${numberOfLayers}.`);
        }
        if (!Number.isInteger(contextLength) || contextLength <= 0) {
            throw new TransformerError(`KV cache contextLength must be positive. Received ${contextLength}.`);
        }
        if (!Number.isInteger(batchSize) || batchSize <= 0) {
            throw new TransformerError(`KV cache batchSize must be positive. Received ${batchSize}.`);
        }
        this.layers = new Array<TransformerLayerKeyValueCache | undefined>(numberOfLayers);
    }

    public get sequenceLength(): number {
        return this.currentSequenceLength;
    }

    public getLayer(index: number): TransformerLayerKeyValueCache | undefined {
        this.assertLayerIndex(index);
        return this.layers[index];
    }

    public setLayer(index: number, entry: TransformerLayerKeyValueCache): void {
        this.assertLayerIndex(index);
        this.layers[index] = entry;
    }

    public commitStep(sequenceLength: number): void {
        if (!Number.isInteger(sequenceLength) || sequenceLength < 0 || sequenceLength > this.contextLength) {
            throw new TransformerError(
                `KV cache sequence length must be between 0 and ${this.contextLength}. Received ${sequenceLength}.`,
            );
        }
        this.currentSequenceLength = sequenceLength;
    }

    public clear(): void {
        for (let index: number = 0; index < this.layers.length; index += 1) {
            this.layers[index] = undefined;
        }
        this.currentSequenceLength = 0;
    }

    private assertLayerIndex(index: number): void {
        if (!Number.isInteger(index) || index < 0 || index >= this.numberOfLayers) {
            throw new TransformerError(`KV cache layer index ${index} is outside [0, ${this.numberOfLayers - 1}].`);
        }
    }
}

/** Concatenates inference tensors along the sequence axis and intentionally detaches the result. */
export function concatenateCachedSequence(previous: Tensor | undefined, current: Tensor): Tensor {
    if (previous === undefined) {
        return current.detach();
    }
    if (previous.rank !== current.rank || (current.rank !== 2 && current.rank !== 3)) {
        throw new TransformerError('Cached sequence tensors must have matching rank 2 or rank 3.');
    }
    if (previous.backend !== current.backend || previous.dtype !== current.dtype) {
        throw new TransformerError('Cached sequence tensors must use the same backend and dtype.');
    }

    const previousShape: readonly number[] = previous.shape;
    const currentShape: readonly number[] = current.shape;
    const featureIndex: number = current.rank - 1;
    const sequenceIndex: number = current.rank - 2;
    const previousFeatures: number | undefined = previousShape[featureIndex];
    const currentFeatures: number | undefined = currentShape[featureIndex];
    const previousSequence: number | undefined = previousShape[sequenceIndex];
    const currentSequence: number | undefined = currentShape[sequenceIndex];
    if (
        previousFeatures === undefined ||
        currentFeatures === undefined ||
        previousSequence === undefined ||
        currentSequence === undefined ||
        previousFeatures !== currentFeatures
    ) {
        throw new TransformerError('Cached sequence tensor dimensions are incompatible.');
    }

    const batchSize: number = current.rank === 3 ? (currentShape[0] ?? 0) : 1;
    if (current.rank === 3 && previousShape[0] !== currentShape[0]) {
        throw new TransformerError('Cached sequence tensor batch sizes must match.');
    }

    const outputSequence: number = previousSequence + currentSequence;
    const outputShape: number[] = current.rank === 3
        ? [batchSize, outputSequence, currentFeatures]
        : [outputSequence, currentFeatures];
    const output: Float32Array = new Float32Array(batchSize * outputSequence * currentFeatures);
    const previousData: Float32Array = previous.toFloat32Array();
    const currentData: Float32Array = current.toFloat32Array();

    for (let batch: number = 0; batch < batchSize; batch += 1) {
        const outputBatchOffset: number = batch * outputSequence * currentFeatures;
        const previousBatchOffset: number = batch * previousSequence * currentFeatures;
        const currentBatchOffset: number = batch * currentSequence * currentFeatures;
        output.set(
            previousData.subarray(previousBatchOffset, previousBatchOffset + previousSequence * currentFeatures),
            outputBatchOffset,
        );
        output.set(
            currentData.subarray(currentBatchOffset, currentBatchOffset + currentSequence * currentFeatures),
            outputBatchOffset + previousSequence * currentFeatures,
        );
    }

    return new Tensor(output, outputShape, { backend: current.backend, dtype: current.dtype });
}
