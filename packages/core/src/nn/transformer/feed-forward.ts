import { GELU } from '@core/nn/activations/gelu';
import { Linear } from '@core/nn/layers/linear';
import { Module } from '@core/nn/module';
import { TransformerConfigurationError } from '@core/nn/transformer/transformer.errors';
import { Tensor } from '@core/tensor/tensor';

export interface TransformerFeedForwardOptions {
    readonly bias?: boolean;
    readonly backend?: string;
}

/**
 * Position-wise Transformer feed-forward network.
 */
export class TransformerFeedForward extends Module {
    public readonly inputProjection: Linear;
    public readonly outputProjection: Linear;
    public readonly activation: GELU;

    public constructor(
        embeddingDimension: number,
        feedForwardDimension: number,
        options: TransformerFeedForwardOptions = {},
    ) {
        super();
        assertPositiveInteger(embeddingDimension, 'embeddingDimension');
        assertPositiveInteger(feedForwardDimension, 'feedForwardDimension');

        const linearOptions = {
            bias: options.bias ?? true,
            ...(options.backend === undefined ? {} : { backend: options.backend }),
        };

        this.inputProjection = this.registerModule(
            'inputProjection',
            new Linear(embeddingDimension, feedForwardDimension, linearOptions),
        );
        this.activation = this.registerModule('activation', new GELU());
        this.outputProjection = this.registerModule(
            'outputProjection',
            new Linear(feedForwardDimension, embeddingDimension, linearOptions),
        );
    }

    public override forward(input: Tensor): Tensor {
        const hidden: Tensor = this.inputProjection.forward(input);
        const activated: Tensor = this.activation.forward(hidden);
        return this.outputProjection.forward(activated);
    }
}

function assertPositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new TransformerConfigurationError(`${name} must be a positive integer. Received ${value}.`);
    }
}
