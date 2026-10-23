import type { Dataset } from '@core/data';
import { LanguageModelError } from '@core/language-model/language-model.errors';
import { Tensor } from '@core/tensor';
import type { Tokenizer } from '@core/tokenizer';
import type { TrainingSample } from '@core/training';

export interface CausalLanguageModelDatasetOptions {
    readonly sequenceLength: number;
    readonly stride?: number;
    readonly backend?: string;
}

/** Sliding-window next-token dataset for decoder-only language-model pretraining. */
export class CausalLanguageModelDataset implements Dataset<TrainingSample> {
    private readonly tokenIds: readonly number[];
    private readonly sequenceLength: number;
    private readonly stride: number;
    private readonly backend: string;
    private readonly sampleCount: number;

    public constructor(tokenIds: readonly number[], options: CausalLanguageModelDatasetOptions) {
        if (!Number.isInteger(options.sequenceLength) || options.sequenceLength <= 0) {
            throw new LanguageModelError(
                `sequenceLength must be a positive integer. Received ${options.sequenceLength}.`,
            );
        }
        this.sequenceLength = options.sequenceLength;
        this.stride = options.stride ?? options.sequenceLength;
        if (!Number.isInteger(this.stride) || this.stride <= 0) {
            throw new LanguageModelError(`stride must be a positive integer. Received ${this.stride}.`);
        }
        if (tokenIds.length < this.sequenceLength + 1) {
            throw new LanguageModelError(
                `Causal LM dataset needs at least ${this.sequenceLength + 1} tokens. Received ${tokenIds.length}.`,
            );
        }
        this.tokenIds = [...tokenIds];
        this.backend = options.backend ?? 'cpu';
        this.sampleCount = Math.floor((tokenIds.length - (this.sequenceLength + 1)) / this.stride) + 1;
    }

    public static fromTexts(
        texts: readonly string[],
        tokenizer: Tokenizer,
        options: CausalLanguageModelDatasetOptions,
    ): CausalLanguageModelDataset {
        const tokens: number[] = [];
        for (let index: number = 0; index < texts.length; index += 1) {
            const text: string | undefined = texts[index];
            if (text === undefined) continue;
            tokens.push(
                tokenizer.beginningOfSequenceTokenId,
                ...tokenizer.encode(text, false),
                tokenizer.endOfSequenceTokenId,
            );
        }
        return new CausalLanguageModelDataset(tokens, options);
    }

    public get length(): number {
        return this.sampleCount;
    }

    public get(index: number): TrainingSample {
        if (!Number.isInteger(index) || index < 0 || index >= this.sampleCount) {
            throw new LanguageModelError(
                `Language-model sample index ${index} is outside [0, ${this.sampleCount - 1}].`,
            );
        }
        const start: number = index * this.stride;
        const input: Float32Array = new Float32Array(this.sequenceLength);
        const target: Float32Array = new Float32Array(this.sequenceLength);
        for (let offset: number = 0; offset < this.sequenceLength; offset += 1) {
            const inputId: number | undefined = this.tokenIds[start + offset];
            const targetId: number | undefined = this.tokenIds[start + offset + 1];
            if (inputId === undefined || targetId === undefined) {
                throw new LanguageModelError(`Missing causal token at sample ${index}, offset ${offset}.`);
            }
            input[offset] = inputId;
            target[offset] = targetId;
        }
        return {
            input: new Tensor(input, [this.sequenceLength], { backend: this.backend }),
            target: new Tensor(target, [this.sequenceLength], { backend: this.backend }),
        };
    }
}
