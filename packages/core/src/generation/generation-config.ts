import { GenerationError } from '@core/generation/generation.errors';

export interface GenerationConfig {
    readonly maxNewTokens?: number;
    readonly temperature?: number;
    readonly topK?: number;
    readonly stopOnEndOfSequence?: boolean;
    /** Uses incremental KV-cache inference for modern Transformer models. Defaults to true. */
    readonly useKvCache?: boolean;
}

export interface NormalizedGenerationConfig {
    readonly maxNewTokens: number;
    readonly temperature: number;
    readonly topK: number | undefined;
    readonly stopOnEndOfSequence: boolean;
    readonly useKvCache: boolean;
}

export function normalizeGenerationConfig(config: GenerationConfig = {}): NormalizedGenerationConfig {
    const maxNewTokens: number = config.maxNewTokens ?? 32;
    const temperature: number = config.temperature ?? 1;
    const topK: number | undefined = config.topK;
    const stopOnEndOfSequence: boolean = config.stopOnEndOfSequence ?? true;
    const useKvCache: boolean = config.useKvCache ?? true;

    if (!Number.isInteger(maxNewTokens) || maxNewTokens < 0) {
        throw new GenerationError(`maxNewTokens must be a non-negative integer. Received ${maxNewTokens}.`);
    }
    if (!Number.isFinite(temperature) || temperature < 0) {
        throw new GenerationError(`temperature must be finite and non-negative. Received ${temperature}.`);
    }
    if (topK !== undefined && (!Number.isInteger(topK) || topK <= 0)) {
        throw new GenerationError(`topK must be a positive integer when provided. Received ${topK}.`);
    }
    return { maxNewTokens, temperature, topK, stopOnEndOfSequence, useKvCache };
}
