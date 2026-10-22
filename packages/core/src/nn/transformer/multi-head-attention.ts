import { Linear } from '@core/nn/layers/linear';
import { Module } from '@core/nn/module';
import { createCausalMask } from '@core/nn/transformer/causal-mask';
import { ScaledDotProductAttention } from '@core/nn/transformer/scaled-dot-product-attention';
import { AttentionShapeError, TransformerConfigurationError } from '@core/nn/transformer/transformer.errors';
import { Tensor } from '@core/tensor/tensor';

export interface MultiHeadAttentionOptions {
    readonly causal?: boolean;
    readonly bias?: boolean;
    readonly backend?: string;
}

/**
 * Multi-head self-attention for [sequence, embedding] and
 * [batch, sequence, embedding] tensors.
 */
export class MultiHeadAttention extends Module {
    public readonly embeddingDimension: number;
    public readonly numberOfHeads: number;
    public readonly headDimension: number;
    public readonly causal: boolean;
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

    public override forward(input: Tensor): Tensor {
        if (input.rank !== 2 && input.rank !== 3) {
            throw new AttentionShapeError(
                `MultiHeadAttention expects [sequence, embedding] or [batch, sequence, embedding] input. Received rank ${input.rank}.`,
            );
        }

        const shape: readonly number[] = input.shape;
        const sequenceAxis: number = input.rank - 2;
        const embeddingAxis: number = input.rank - 1;
        const sequenceLength: number | undefined = shape[sequenceAxis];
        const embeddingDimension: number | undefined = shape[embeddingAxis];

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
            const queryHead: Tensor = query.sliceColumns(startColumn, endColumn);
            const keyHead: Tensor = key.sliceColumns(startColumn, endColumn);
            const valueHead: Tensor = value.sliceColumns(startColumn, endColumn);

            heads[head] = this.attention.forward(queryHead, keyHead, valueHead, mask);
        }

        const combined: Tensor = Tensor.concatColumns(heads);
        return this.outputProjection.forward(combined);
    }
}

function assertPositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new TransformerConfigurationError(`${name} must be a positive integer. Received ${value}.`);
    }
}
