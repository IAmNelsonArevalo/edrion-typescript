import { LayerNormalization } from '@core/nn/layers/layer-normalization';
import { RMSNormalization } from '@core/nn/layers/rms-normalization';
import { Module } from '@core/nn/module';
import { TransformerFeedForward } from '@core/nn/transformer/feed-forward';
import { MultiHeadAttention, type IncrementalAttentionResult } from '@core/nn/transformer/multi-head-attention';
import type { TransformerLayerKeyValueCache } from '@core/nn/transformer/kv-cache';
import { SwiGLUFeedForward } from '@core/nn/transformer/swiglu-feed-forward';
import type { TransformerArchitecture } from '@core/nn/transformer/transformer-config';
import { Tensor } from '@core/tensor/tensor';

export interface TransformerBlockOptions {
    readonly layerNormEpsilon?: number;
    readonly backend?: string;
    readonly architecture?: TransformerArchitecture;
    readonly rotaryBase?: number;
    readonly rotaryDimension?: number;
}

type NormalizationModule = LayerNormalization | RMSNormalization;
type FeedForwardModule = TransformerFeedForward | SwiGLUFeedForward;

/** Pre-normalization causal Transformer block with legacy and modern modes. */
export class TransformerBlock extends Module {
    public readonly attentionNormalization: NormalizationModule;
    public readonly attention: MultiHeadAttention;
    public readonly feedForwardNormalization: NormalizationModule;
    public readonly feedForward: FeedForwardModule;
    public readonly architecture: TransformerArchitecture;

    public constructor(
        embeddingDimension: number,
        numberOfHeads: number,
        feedForwardDimension: number,
        options: TransformerBlockOptions = {},
    ) {
        super();
        this.architecture = options.architecture ?? 'legacy';
        const normalizationOptions = {
            ...(options.layerNormEpsilon === undefined ? {} : { epsilon: options.layerNormEpsilon }),
            ...(options.backend === undefined ? {} : { backend: options.backend }),
        };

        this.attentionNormalization = this.registerModule(
            'attentionNormalization',
            this.architecture === 'modern'
                ? new RMSNormalization(embeddingDimension, normalizationOptions)
                : new LayerNormalization(embeddingDimension, normalizationOptions),
        );
        this.attention = this.registerModule(
            'attention',
            new MultiHeadAttention(embeddingDimension, numberOfHeads, {
                causal: true,
                bias: this.architecture !== 'modern',
                rotaryEmbedding: this.architecture === 'modern',
                ...(options.rotaryBase === undefined ? {} : { rotaryBase: options.rotaryBase }),
                ...(options.rotaryDimension === undefined ? {} : { rotaryDimension: options.rotaryDimension }),
                ...(options.backend === undefined ? {} : { backend: options.backend }),
            }),
        );
        this.feedForwardNormalization = this.registerModule(
            'feedForwardNormalization',
            this.architecture === 'modern'
                ? new RMSNormalization(embeddingDimension, normalizationOptions)
                : new LayerNormalization(embeddingDimension, normalizationOptions),
        );
        this.feedForward = this.registerModule(
            'feedForward',
            this.architecture === 'modern'
                ? new SwiGLUFeedForward(
                      embeddingDimension,
                      feedForwardDimension,
                      options.backend === undefined ? {} : { backend: options.backend },
                  )
                : new TransformerFeedForward(
                      embeddingDimension,
                      feedForwardDimension,
                      options.backend === undefined ? {} : { backend: options.backend },
                  ),
        );
    }

    /** Inference-only Transformer block step backed by a KV cache. */
    public forwardIncremental(
        input: Tensor,
        previousCache: TransformerLayerKeyValueCache | undefined,
        positionOffset: number,
    ): { readonly output: Tensor; readonly cache: TransformerLayerKeyValueCache } {
        const normalizedAttentionInput: Tensor = this.attentionNormalization.forward(input);
        const attentionResult: IncrementalAttentionResult = this.attention.forwardIncremental(
            normalizedAttentionInput,
            previousCache,
            positionOffset,
        );
        const afterAttention: Tensor = input.add(attentionResult.output);
        const normalizedFeedForwardInput: Tensor = this.feedForwardNormalization.forward(afterAttention);
        const feedForwardOutput: Tensor = this.feedForward.forward(normalizedFeedForwardInput);
        return {
            output: afterAttention.add(feedForwardOutput),
            cache: attentionResult.cache,
        };
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
