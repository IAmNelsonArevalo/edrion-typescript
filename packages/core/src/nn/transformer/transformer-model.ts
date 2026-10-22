import { Embedding } from '@core/nn/layers/embedding';
import { LayerNormalization } from '@core/nn/layers/layer-normalization';
import { Linear } from '@core/nn/layers/linear';
import { Module } from '@core/nn/module';
import {
    normalizeTransformerConfig,
    type NormalizedTransformerConfig,
    type TransformerConfig,
} from '@core/nn/transformer/transformer-config';
import { TransformerBlock } from '@core/nn/transformer/transformer-block';
import { TransformerError } from '@core/nn/transformer/transformer.errors';
import { Tensor } from '@core/tensor/tensor';

/**
 * Decoder-only Transformer language-model foundation.
 *
 * FEATURE-006 accepts one token sequence at a time with input shape [sequence].
 */
export class TransformerModel extends Module {
    public readonly config: NormalizedTransformerConfig;
    public readonly tokenEmbedding: Embedding;
    public readonly positionEmbedding: Embedding;
    public readonly blocks: readonly TransformerBlock[];
    public readonly finalNormalization: LayerNormalization;
    public readonly languageModelHead: Linear;

    public constructor(config: TransformerConfig) {
        super();
        this.config = normalizeTransformerConfig(config);

        this.tokenEmbedding = this.registerModule(
            'tokenEmbedding',
            new Embedding(this.config.vocabularySize, this.config.embeddingDimension, { backend: this.config.backend }),
        );
        this.positionEmbedding = this.registerModule(
            'positionEmbedding',
            new Embedding(this.config.contextLength, this.config.embeddingDimension, { backend: this.config.backend }),
        );

        const blocks: TransformerBlock[] = new Array<TransformerBlock>(this.config.numberOfLayers);

        for (let index: number = 0; index < this.config.numberOfLayers; index += 1) {
            const block: TransformerBlock = new TransformerBlock(
                this.config.embeddingDimension,
                this.config.numberOfHeads,
                this.config.feedForwardDimension,
                {
                    layerNormEpsilon: this.config.layerNormEpsilon,
                    backend: this.config.backend,
                },
            );

            blocks[index] = this.registerModule(`blocks${index}`, block);
        }

        this.blocks = blocks;
        this.finalNormalization = this.registerModule(
            'finalNormalization',
            new LayerNormalization(this.config.embeddingDimension, {
                epsilon: this.config.layerNormEpsilon,
                backend: this.config.backend,
            }),
        );
        this.languageModelHead = this.registerModule(
            'languageModelHead',
            new Linear(this.config.embeddingDimension, this.config.vocabularySize, {
                bias: false,
                backend: this.config.backend,
            }),
        );
    }

    /**
     * Produces vocabulary logits with shape [sequence, vocabularySize].
     */
    public override forward(tokens: Tensor): Tensor {
        if (tokens.rank !== 1) {
            throw new TransformerError(
                `TransformerModel expects token input with rank 1. Received rank ${tokens.rank}.`,
            );
        }

        const tokenShape: readonly number[] = tokens.shape;
        const sequenceLength: number | undefined = tokenShape[0];

        if (sequenceLength === undefined) {
            throw new TransformerError('Unable to resolve token sequence length.');
        }

        if (sequenceLength > this.config.contextLength) {
            throw new TransformerError(
                `Sequence length ${sequenceLength} exceeds context length ${this.config.contextLength}.`,
            );
        }

        if (tokens.backend !== this.config.backend) {
            throw new TransformerError(
                `Token backend "${tokens.backend}" does not match model backend "${this.config.backend}".`,
            );
        }

        const positionValues: Float32Array = new Float32Array(sequenceLength);

        for (let index: number = 0; index < sequenceLength; index += 1) {
            positionValues[index] = index;
        }

        const positions: Tensor = new Tensor(positionValues, [sequenceLength], { backend: tokens.backend });
        const tokenEmbeddings: Tensor = this.tokenEmbedding.forward(tokens);
        const positionEmbeddings: Tensor = this.positionEmbedding.forward(positions);
        let hiddenState: Tensor = tokenEmbeddings.add(positionEmbeddings);

        for (let index: number = 0; index < this.blocks.length; index += 1) {
            const block: TransformerBlock | undefined = this.blocks[index];

            if (block === undefined) {
                throw new TransformerError(`Transformer block at index ${index} does not exist.`);
            }

            hiddenState = block.forward(hiddenState);
        }

        const normalized: Tensor = this.finalNormalization.forward(hiddenState);
        return this.languageModelHead.forward(normalized);
    }
}
