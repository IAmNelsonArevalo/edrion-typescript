import { Embedding } from '@core/nn/layers/embedding';
import { LayerNormalization } from '@core/nn/layers/layer-normalization';
import { RMSNormalization } from '@core/nn/layers/rms-normalization';
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

type FinalNormalization = LayerNormalization | RMSNormalization;

/** Decoder-only Transformer with FEATURE-014 modern architecture support. */
export class TransformerModel extends Module {
    public readonly config: NormalizedTransformerConfig;
    public readonly tokenEmbedding: Embedding;
    public readonly positionEmbedding: Embedding | undefined;
    public readonly blocks: readonly TransformerBlock[];
    public readonly finalNormalization: FinalNormalization;
    public readonly languageModelHead: Linear;

    public constructor(config: TransformerConfig) {
        super();
        this.config = normalizeTransformerConfig(config);
        this.tokenEmbedding = this.registerModule(
            'tokenEmbedding',
            new Embedding(this.config.vocabularySize, this.config.embeddingDimension, { backend: this.config.backend }),
        );

        this.positionEmbedding =
            this.config.architecture === 'legacy'
                ? this.registerModule(
                      'positionEmbedding',
                      new Embedding(this.config.contextLength, this.config.embeddingDimension, {
                          backend: this.config.backend,
                      }),
                  )
                : undefined;

        const blocks: TransformerBlock[] = new Array<TransformerBlock>(this.config.numberOfLayers);
        for (let index: number = 0; index < this.config.numberOfLayers; index += 1) {
            const block: TransformerBlock = new TransformerBlock(
                this.config.embeddingDimension,
                this.config.numberOfHeads,
                this.config.feedForwardDimension,
                {
                    layerNormEpsilon: this.config.layerNormEpsilon,
                    backend: this.config.backend,
                    architecture: this.config.architecture,
                    rotaryBase: this.config.rotaryBase,
                    rotaryDimension: this.config.rotaryDimension,
                },
            );
            blocks[index] = this.registerModule(`blocks${index}`, block);
        }
        this.blocks = blocks;
        this.finalNormalization = this.registerModule(
            'finalNormalization',
            this.config.architecture === 'modern'
                ? new RMSNormalization(this.config.embeddingDimension, {
                      epsilon: this.config.layerNormEpsilon,
                      backend: this.config.backend,
                  })
                : new LayerNormalization(this.config.embeddingDimension, {
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

    public override forward(tokens: Tensor): Tensor {
        if (tokens.rank !== 1 && tokens.rank !== 2) {
            throw new TransformerError(
                `TransformerModel expects [sequence] or [batch, sequence] token input. Received rank ${tokens.rank}.`,
            );
        }
        const sequenceLength: number | undefined = tokens.shape[tokens.rank - 1];
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

        const tokenEmbeddings: Tensor = this.tokenEmbedding.forward(tokens);
        let hiddenState: Tensor = tokenEmbeddings;

        if (this.positionEmbedding !== undefined) {
            const positionValues: Float32Array = new Float32Array(sequenceLength);
            for (let index: number = 0; index < sequenceLength; index += 1) {
                positionValues[index] = index;
            }
            const positions: Tensor = new Tensor(positionValues, [sequenceLength], { backend: tokens.backend });
            hiddenState = hiddenState.add(this.positionEmbedding.forward(positions));
        }

        for (let index: number = 0; index < this.blocks.length; index += 1) {
            const block: TransformerBlock | undefined = this.blocks[index];
            if (block === undefined) {
                throw new TransformerError(`Transformer block at index ${index} does not exist.`);
            }
            hiddenState = block.forward(hiddenState);
        }

        return this.languageModelHead.forward(this.finalNormalization.forward(hiddenState));
    }
}
