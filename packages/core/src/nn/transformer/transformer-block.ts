import { LayerNormalization } from '@core/nn/layers/layer-normalization';
import { Module } from '@core/nn/module';
import { TransformerFeedForward } from '@core/nn/transformer/feed-forward';
import { MultiHeadAttention } from '@core/nn/transformer/multi-head-attention';
import { Tensor } from '@core/tensor/tensor';

export interface TransformerBlockOptions {
    readonly layerNormEpsilon?: number;
    readonly backend?: string;
}

/**
 * Pre-normalization causal Transformer block.
 */
export class TransformerBlock extends Module {
    public readonly attentionNormalization: LayerNormalization;
    public readonly attention: MultiHeadAttention;
    public readonly feedForwardNormalization: LayerNormalization;
    public readonly feedForward: TransformerFeedForward;

    public constructor(
        embeddingDimension: number,
        numberOfHeads: number,
        feedForwardDimension: number,
        options: TransformerBlockOptions = {},
    ) {
        super();

        const normalizationOptions = {
            ...(options.layerNormEpsilon === undefined ? {} : { epsilon: options.layerNormEpsilon }),
            ...(options.backend === undefined ? {} : { backend: options.backend }),
        };

        this.attentionNormalization = this.registerModule(
            'attentionNormalization',
            new LayerNormalization(embeddingDimension, normalizationOptions),
        );
        this.attention = this.registerModule(
            'attention',
            new MultiHeadAttention(embeddingDimension, numberOfHeads, {
                causal: true,
                ...(options.backend === undefined ? {} : { backend: options.backend }),
            }),
        );
        this.feedForwardNormalization = this.registerModule(
            'feedForwardNormalization',
            new LayerNormalization(embeddingDimension, normalizationOptions),
        );
        this.feedForward = this.registerModule(
            'feedForward',
            new TransformerFeedForward(
                embeddingDimension,
                feedForwardDimension,
                options.backend === undefined ? {} : { backend: options.backend },
            ),
        );
    }

    public override forward(input: Tensor): Tensor {
        const normalizedAttentionInput: Tensor = this.attentionNormalization.forward(input);
        const attentionOutput: Tensor = this.attention.forward(normalizedAttentionInput);
        const afterAttention: Tensor = input.add(attentionOutput);
        const normalizedFeedForwardInput: Tensor = this.feedForwardNormalization.forward(afterAttention);
        const feedForwardOutput: Tensor = this.feedForward.forward(normalizedFeedForwardInput);

        return afterAttention.add(feedForwardOutput);
    }
}
