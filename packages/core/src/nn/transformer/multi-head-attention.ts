import { Linear } from '@core/nn/layers/linear';
import { Module } from '@core/nn/module';
import { createCausalMask } from '@core/nn/transformer/causal-mask';
import { concatenateCachedSequence, type TransformerLayerKeyValueCache } from '@core/nn/transformer/kv-cache';
import { ScaledDotProductAttention } from '@core/nn/transformer/scaled-dot-product-attention';
import { AttentionShapeError, TransformerConfigurationError } from '@core/nn/transformer/transformer.errors';
import { Tensor } from '@core/tensor/tensor';

export interface IncrementalAttentionResult {
    readonly output: Tensor;
    readonly cache: TransformerLayerKeyValueCache;
}

export interface MultiHeadAttentionOptions {
    readonly causal?: boolean;
    readonly bias?: boolean;
    readonly backend?: string;
    readonly rotaryEmbedding?: boolean;
    readonly rotaryBase?: number;
    readonly rotaryDimension?: number;
}

/** Multi-head self-attention with optional rotary positional embedding. */
export class MultiHeadAttention extends Module {
    public readonly embeddingDimension: number;
    public readonly numberOfHeads: number;
    public readonly headDimension: number;
    public readonly causal: boolean;
    public readonly rotaryEmbedding: boolean;
    public readonly rotaryBase: number;
    public readonly rotaryDimension: number;
    public readonly queryProjection: Linear;
    public readonly keyProjection: Linear;
    public readonly valueProjection: Linear;
    public readonly outputProjection: Linear;

    private readonly attention: ScaledDotProductAttention = new ScaledDotProductAttention();

    public constructor(embeddingDimension: number, numberOfHeads: number, options: MultiHeadAttentionOptions = {}) {
        super();
        assertPositiveInteger(embeddingDimension, 'embeddingDimension');
        assertPositiveInteger(numberOfHeads, 'numberOfHeads');

        if (embeddingDimension % numberOfHeads !== 0) {
            throw new TransformerConfigurationError(
                `embeddingDimension ${embeddingDimension} must be divisible by numberOfHeads ${numberOfHeads}.`,
            );
        }

        this.embeddingDimension = embeddingDimension;
        this.numberOfHeads = numberOfHeads;
        this.headDimension = embeddingDimension / numberOfHeads;
        this.causal = options.causal ?? true;
        this.rotaryEmbedding = options.rotaryEmbedding ?? false;
        this.rotaryBase = options.rotaryBase ?? 10_000;
        this.rotaryDimension = options.rotaryDimension ?? this.headDimension;

        if (this.rotaryEmbedding && (this.rotaryDimension > this.headDimension || this.rotaryDimension % 2 !== 0)) {
            throw new TransformerConfigurationError(
                `rotaryDimension ${this.rotaryDimension} must be even and <= headDimension ${this.headDimension}.`,
            );
        }

        const linearOptions = {
            bias: options.bias ?? true,
            ...(options.backend === undefined ? {} : { backend: options.backend }),
        };

        this.queryProjection = this.registerModule(
            'queryProjection',
            new Linear(embeddingDimension, embeddingDimension, linearOptions),
        );
        this.keyProjection = this.registerModule(
            'keyProjection',
            new Linear(embeddingDimension, embeddingDimension, linearOptions),
        );
        this.valueProjection = this.registerModule(
            'valueProjection',
            new Linear(embeddingDimension, embeddingDimension, linearOptions),
        );
        this.outputProjection = this.registerModule(
            'outputProjection',
            new Linear(embeddingDimension, embeddingDimension, linearOptions),
        );
    }

    /** Inference-only one-step attention using a per-layer KV cache. */
    public forwardIncremental(
        input: Tensor,
        previousCache: TransformerLayerKeyValueCache | undefined,
        positionOffset: number,
    ): IncrementalAttentionResult {
        if (input.rank !== 2 && input.rank !== 3) {
            throw new AttentionShapeError(
                `Incremental attention expects [sequence, embedding] or [batch, sequence, embedding]. Received rank ${input.rank}.`,
            );
        }

        const shape: readonly number[] = input.shape;
        const sequenceLength: number | undefined = shape[input.rank - 2];
        const embeddingDimension: number | undefined = shape[input.rank - 1];
        if (sequenceLength === undefined || embeddingDimension === undefined || sequenceLength <= 0) {
            throw new AttentionShapeError('Unable to resolve incremental attention dimensions.');
        }
        if (embeddingDimension !== this.embeddingDimension) {
            throw new AttentionShapeError(
                `Incremental attention expected embedding dimension ${this.embeddingDimension} but received ${embeddingDimension}.`,
            );
        }

        const projectedQuery: Tensor = this.queryProjection.forward(input);
        const projectedKey: Tensor = this.keyProjection.forward(input);
        const projectedValue: Tensor = this.valueProjection.forward(input);
        const queryHeads: Tensor[] = new Array<Tensor>(this.numberOfHeads);
        const currentKeyHeads: Tensor[] = new Array<Tensor>(this.numberOfHeads);

        for (let head: number = 0; head < this.numberOfHeads; head += 1) {
            const startColumn: number = head * this.headDimension;
            const endColumn: number = startColumn + this.headDimension;
            let queryHead: Tensor = projectedQuery.sliceColumns(startColumn, endColumn);
            let keyHead: Tensor = projectedKey.sliceColumns(startColumn, endColumn);

            if (this.rotaryEmbedding) {
                queryHead = queryHead.applyRotaryPositionEmbedding(
                    this.rotaryDimension,
                    this.rotaryBase,
                    positionOffset,
                );
                keyHead = keyHead.applyRotaryPositionEmbedding(this.rotaryDimension, this.rotaryBase, positionOffset);
            }
            queryHeads[head] = queryHead;
            currentKeyHeads[head] = keyHead;
        }

        const currentKey: Tensor = Tensor.concatColumns(currentKeyHeads);
        const key: Tensor = concatenateCachedSequence(previousCache?.key, currentKey);
        const value: Tensor = concatenateCachedSequence(previousCache?.value, projectedValue);
        const heads: Tensor[] = new Array<Tensor>(this.numberOfHeads);

        for (let head: number = 0; head < this.numberOfHeads; head += 1) {
            const startColumn: number = head * this.headDimension;
            const endColumn: number = startColumn + this.headDimension;
            const queryHead: Tensor | undefined = queryHeads[head];
            if (queryHead === undefined) {
                throw new AttentionShapeError(`Missing incremental query head ${head}.`);
            }
            const keyHead: Tensor = key.sliceColumns(startColumn, endColumn);
            const valueHead: Tensor = value.sliceColumns(startColumn, endColumn);
            heads[head] = this.attention.forward(queryHead, keyHead, valueHead);
        }

        return {
            output: this.outputProjection.forward(Tensor.concatColumns(heads)),
            cache: { key, value },
        };
    }

    public override forward(input: Tensor): Tensor {
        if (input.rank !== 2 && input.rank !== 3) {
            throw new AttentionShapeError(
                `MultiHeadAttention expects [sequence, embedding] or [batch, sequence, embedding] input. Received rank ${input.rank}.`,
            );
        }

        const shape: readonly number[] = input.shape;
        const sequenceLength: number | undefined = shape[input.rank - 2];
        const embeddingDimension: number | undefined = shape[input.rank - 1];
        if (sequenceLength === undefined || embeddingDimension === undefined) {
            throw new AttentionShapeError('Unable to resolve self-attention input dimensions.');
        }
        if (embeddingDimension !== this.embeddingDimension) {
            throw new AttentionShapeError(
                `MultiHeadAttention expected embedding dimension ${this.embeddingDimension} but received ${embeddingDimension}.`,
            );
        }

        const query: Tensor = this.queryProjection.forward(input);
        const key: Tensor = this.keyProjection.forward(input);
        const value: Tensor = this.valueProjection.forward(input);
        const mask: Tensor | undefined = this.causal ? createCausalMask(sequenceLength, input.backend) : undefined;
        const heads: Tensor[] = new Array<Tensor>(this.numberOfHeads);

        for (let head: number = 0; head < this.numberOfHeads; head += 1) {
            const startColumn: number = head * this.headDimension;
            const endColumn: number = startColumn + this.headDimension;
            let queryHead: Tensor = query.sliceColumns(startColumn, endColumn);
            let keyHead: Tensor = key.sliceColumns(startColumn, endColumn);
            const valueHead: Tensor = value.sliceColumns(startColumn, endColumn);

            if (this.rotaryEmbedding) {
                queryHead = queryHead.applyRotaryPositionEmbedding(this.rotaryDimension, this.rotaryBase);
                keyHead = keyHead.applyRotaryPositionEmbedding(this.rotaryDimension, this.rotaryBase);
            }

            heads[head] = this.attention.forward(queryHead, keyHead, valueHead, mask);
        }

        return this.outputProjection.forward(Tensor.concatColumns(heads));
    }
}

function assertPositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new TransformerConfigurationError(`${name} must be a positive integer. Received ${value}.`);
    }
}
