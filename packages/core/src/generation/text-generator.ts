import { GenerationError } from '@core/generation/generation.errors';
import {
    normalizeGenerationConfig,
    type GenerationConfig,
    type NormalizedGenerationConfig,
} from '@core/generation/generation-config';
import { GreedySampler } from '@core/generation/greedy-sampler';
import { TemperatureSampler } from '@core/generation/temperature-sampler';
import { TopKSampler } from '@core/generation/top-k-sampler';
import { defaultRandomSource, type RandomSource, type Sampler } from '@core/generation/sampler';
import { TransformerModel } from '@core/nn/transformer/transformer-model';
import type { Tokenizer } from '@core/tokenizer/tokenizer.interface';
import { Tensor } from '@core/tensor/tensor';

export interface GenerationResult {
    readonly text: string;
    readonly tokenIds: readonly number[];
    readonly generatedTokenIds: readonly number[];
}

export class TextGenerator {
    private readonly model: TransformerModel;
    private readonly tokenizer: Tokenizer;
    private readonly randomSource: RandomSource;

    public constructor(
        model: TransformerModel,
        tokenizer: Tokenizer,
        randomSource: RandomSource = defaultRandomSource,
    ) {
        if (model.config.vocabularySize !== tokenizer.vocabularySize) {
            throw new GenerationError(
                `Model vocabulary size ${model.config.vocabularySize} does not match tokenizer vocabulary size ${tokenizer.vocabularySize}.`,
            );
        }
        this.model = model;
        this.tokenizer = tokenizer;
        this.randomSource = randomSource;
    }

    public generate(prompt: string, config: GenerationConfig = {}): GenerationResult {
        const normalized: NormalizedGenerationConfig = normalizeGenerationConfig(config);
        const promptIds: readonly number[] = this.tokenizer.encode(prompt, false);
        const allIds: number[] = promptIds.length === 0 ? [this.tokenizer.beginningOfSequenceTokenId] : [...promptIds];
        const generated: number[] = [];
        const sampler: Sampler = this.createSampler(normalized);

        for (let step: number = 0; step < normalized.maxNewTokens; step += 1) {
            const contextIds: readonly number[] = allIds.slice(-this.model.config.contextLength);
            const values: Float32Array = new Float32Array(contextIds.length);
            for (let index: number = 0; index < contextIds.length; index += 1) {
                const tokenId: number | undefined = contextIds[index];
                if (tokenId === undefined) {
                    throw new GenerationError(`Context token id at index ${index} does not exist.`);
                }
                values[index] = tokenId;
            }
            const tokens: Tensor = new Tensor(values, [contextIds.length], { backend: this.model.config.backend });
            const logits: Tensor = this.model.forward(tokens);
            const shape: readonly number[] = logits.shape;
            const sequenceLength: number | undefined = shape[0];
            const vocabularySize: number | undefined = shape[1];
            if (sequenceLength === undefined || vocabularySize === undefined) {
                throw new GenerationError('Transformer logits must have shape [sequence, vocabulary].');
            }
            const lastRow: number[] = new Array<number>(vocabularySize);
            for (let vocabularyIndex: number = 0; vocabularyIndex < vocabularySize; vocabularyIndex += 1) {
                lastRow[vocabularyIndex] = logits.get(sequenceLength - 1, vocabularyIndex);
            }
            const nextTokenId: number = sampler.sample(lastRow);
            allIds.push(nextTokenId);
            generated.push(nextTokenId);
            if (normalized.stopOnEndOfSequence && nextTokenId === this.tokenizer.endOfSequenceTokenId) {
                break;
            }
        }

        return {
            text: this.tokenizer.decode(allIds, true),
            tokenIds: [...allIds],
            generatedTokenIds: [...generated],
        };
    }

    private createSampler(config: NormalizedGenerationConfig): Sampler {
        if (config.temperature === 0) {
            return new GreedySampler();
        }
        if (config.topK !== undefined) {
            return new TopKSampler(config.topK, config.temperature, this.randomSource);
        }
        return new TemperatureSampler(config.temperature, this.randomSource);
    }
}
