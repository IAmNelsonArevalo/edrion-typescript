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
import type { TransformerKeyValueCache } from '@core/nn/transformer/kv-cache';
import type { Tokenizer } from '@core/tokenizer/tokenizer.interface';
import { Tensor } from '@core/tensor/tensor';

export interface GenerationMetrics {
    readonly promptTokenCount: number;
    readonly generatedTokenCount: number;
    readonly durationMs: number;
    readonly tokensPerSecond: number;
    readonly modelForwardCalls: number;
    readonly cacheEnabled: boolean;
    readonly cacheResetCount: number;
}

export interface GenerationTokenEvent {
    readonly index: number;
    readonly tokenId: number;
    readonly text: string;
    readonly elapsedMs: number;
}

export interface GenerationResult {
    readonly text: string;
    readonly tokenIds: readonly number[];
    readonly generatedTokenIds: readonly number[];
    readonly metrics: GenerationMetrics;
}

interface CachedInferenceState {
    readonly cache: TransformerKeyValueCache;
    readonly logits: Tensor;
    readonly forwardCalls: number;
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

    /** Generates a complete result. Modern models use the KV cache by default. */
    public generate(prompt: string, config: GenerationConfig = {}): GenerationResult {
        const iterator: Generator<GenerationTokenEvent, GenerationResult, void> = this.stream(prompt, config);
        let item: IteratorResult<GenerationTokenEvent, GenerationResult> = iterator.next();
        while (!item.done) {
            item = iterator.next();
        }
        return item.value;
    }

    /**
     * Streams generated token events synchronously. The Generator return value is the final GenerationResult.
     * KV-cache inference is used automatically for modern Transformers unless useKvCache is false.
     */
    public *stream(
        prompt: string,
        config: GenerationConfig = {},
    ): Generator<GenerationTokenEvent, GenerationResult, void> {
        const normalized: NormalizedGenerationConfig = normalizeGenerationConfig(config);
        const promptIds: readonly number[] = this.tokenizer.encode(prompt, false);
        const initialIds: number[] =
            promptIds.length === 0 ? [this.tokenizer.beginningOfSequenceTokenId] : [...promptIds];
        const allIds: number[] = [...initialIds];
        const generated: number[] = [];
        const sampler: Sampler = this.createSampler(normalized);
        const startedAt: number = Date.now();
        const cacheEnabled: boolean = normalized.useKvCache && this.model.config.architecture === 'modern';
        let forwardCalls: number = 0;
        let cacheResetCount: number = 0;

        if (normalized.maxNewTokens === 0) {
            return this.createResult(
                allIds,
                generated,
                startedAt,
                forwardCalls,
                cacheEnabled,
                cacheResetCount,
                initialIds.length,
            );
        }

        if (cacheEnabled) {
            let state: CachedInferenceState = this.prefillCache(allIds.slice(-this.model.config.contextLength));
            forwardCalls += state.forwardCalls;

            for (let step: number = 0; step < normalized.maxNewTokens; step += 1) {
                const nextTokenId: number = sampler.sample(this.readLastLogits(state.logits));
                allIds.push(nextTokenId);
                generated.push(nextTokenId);
                yield this.createEvent(step, nextTokenId, startedAt);

                if (normalized.stopOnEndOfSequence && nextTokenId === this.tokenizer.endOfSequenceTokenId) {
                    break;
                }
                if (step + 1 >= normalized.maxNewTokens) {
                    break;
                }

                if (state.cache.sequenceLength >= this.model.config.contextLength) {
                    state = this.prefillCache(allIds.slice(-this.model.config.contextLength));
                    forwardCalls += state.forwardCalls;
                    cacheResetCount += 1;
                } else {
                    const token: Tensor = this.createTokenTensor(nextTokenId);
                    const result = this.model.forwardIncremental(token, state.cache);
                    state = { cache: result.cache, logits: result.logits, forwardCalls: 1 };
                    forwardCalls += 1;
                }
            }
        } else {
            for (let step: number = 0; step < normalized.maxNewTokens; step += 1) {
                const contextIds: readonly number[] = allIds.slice(-this.model.config.contextLength);
                const tokens: Tensor = this.createSequenceTensor(contextIds);
                const logits: Tensor = this.model.forward(tokens);
                forwardCalls += 1;
                const nextTokenId: number = sampler.sample(this.readLastLogits(logits));
                allIds.push(nextTokenId);
                generated.push(nextTokenId);
                yield this.createEvent(step, nextTokenId, startedAt);
                if (normalized.stopOnEndOfSequence && nextTokenId === this.tokenizer.endOfSequenceTokenId) {
                    break;
                }
            }
        }

        return this.createResult(
            allIds,
            generated,
            startedAt,
            forwardCalls,
            cacheEnabled,
            cacheResetCount,
            initialIds.length,
        );
    }

    private prefillCache(contextIds: readonly number[]): CachedInferenceState {
        if (contextIds.length === 0) {
            throw new GenerationError('KV-cache prefill requires at least one token.');
        }
        const cache: TransformerKeyValueCache = this.model.createKeyValueCache(1);
        let logits: Tensor | undefined = undefined;
        let forwardCalls: number = 0;
        for (let index: number = 0; index < contextIds.length; index += 1) {
            const tokenId: number | undefined = contextIds[index];
            if (tokenId === undefined) {
                throw new GenerationError(`Prefill token id at index ${index} does not exist.`);
            }
            const result = this.model.forwardIncremental(this.createTokenTensor(tokenId), cache);
            logits = result.logits;
            forwardCalls += 1;
        }
        if (logits === undefined) {
            throw new GenerationError('KV-cache prefill did not produce logits.');
        }
        return { cache, logits, forwardCalls };
    }

    private createTokenTensor(tokenId: number): Tensor {
        return new Tensor(new Float32Array([tokenId]), [1], { backend: this.model.config.backend });
    }

    private createSequenceTensor(tokenIds: readonly number[]): Tensor {
        const values: Float32Array = new Float32Array(tokenIds.length);
        for (let index: number = 0; index < tokenIds.length; index += 1) {
            const tokenId: number | undefined = tokenIds[index];
            if (tokenId === undefined) {
                throw new GenerationError(`Context token id at index ${index} does not exist.`);
            }
            values[index] = tokenId;
        }
        return new Tensor(values, [tokenIds.length], { backend: this.model.config.backend });
    }

    private readLastLogits(logits: Tensor): readonly number[] {
        if (logits.rank !== 2) {
            throw new GenerationError(
                `Text generation expects rank-2 logits [sequence, vocabulary]. Received rank ${logits.rank}.`,
            );
        }
        const shape: readonly number[] = logits.shape;
        const sequenceLength: number | undefined = shape[0];
        const vocabularySize: number | undefined = shape[1];
        if (sequenceLength === undefined || vocabularySize === undefined || sequenceLength <= 0) {
            throw new GenerationError('Transformer logits must have shape [sequence, vocabulary].');
        }
        const lastRow: number[] = new Array<number>(vocabularySize);
        for (let vocabularyIndex: number = 0; vocabularyIndex < vocabularySize; vocabularyIndex += 1) {
            lastRow[vocabularyIndex] = logits.get(sequenceLength - 1, vocabularyIndex);
        }
        return lastRow;
    }

    private createEvent(index: number, tokenId: number, startedAt: number): GenerationTokenEvent {
        return {
            index,
            tokenId,
            text: this.tokenizer.decode([tokenId], true),
            elapsedMs: Math.max(0, Date.now() - startedAt),
        };
    }

    private createResult(
        allIds: readonly number[],
        generated: readonly number[],
        startedAt: number,
        forwardCalls: number,
        cacheEnabled: boolean,
        cacheResetCount: number,
        promptTokenCount: number,
    ): GenerationResult {
        const durationMs: number = Math.max(0, Date.now() - startedAt);
        const tokensPerSecond: number =
            generated.length === 0 ? 0 : (generated.length * 1_000) / Math.max(1, durationMs);
        return {
            text: this.tokenizer.decode(allIds, true),
            tokenIds: [...allIds],
            generatedTokenIds: [...generated],
            metrics: {
                promptTokenCount,
                generatedTokenCount: generated.length,
                durationMs,
                tokensPerSecond,
                modelForwardCalls: forwardCalls,
                cacheEnabled,
                cacheResetCount,
            },
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
