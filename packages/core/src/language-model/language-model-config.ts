import type { TransformerConfig } from '@core/nn/transformer/transformer-config';
import { LanguageModelError } from '@core/language-model/language-model.errors';

export type LanguageModelPreset = 'tiny' | 'small';

export interface LanguageModelConfig extends Omit<TransformerConfig, 'architecture'> {
    readonly architecture?: 'modern';
}

/** Builds a modern decoder-only Transformer configuration from a named Edrion LM preset. */
export function createLanguageModelConfig(
    preset: LanguageModelPreset,
    vocabularySize: number,
    overrides: Partial<LanguageModelConfig> = {},
): LanguageModelConfig {
    if (!Number.isInteger(vocabularySize) || vocabularySize <= 4) {
        throw new LanguageModelError(
            `Language-model vocabularySize must be an integer greater than 4. Received ${vocabularySize}.`,
        );
    }
    const base: LanguageModelConfig =
        preset === 'tiny'
            ? {
                  vocabularySize,
                  contextLength: 128,
                  embeddingDimension: 128,
                  numberOfLayers: 4,
                  numberOfHeads: 4,
                  feedForwardDimension: 384,
                  architecture: 'modern',
              }
            : {
                  vocabularySize,
                  contextLength: 512,
                  embeddingDimension: 384,
                  numberOfLayers: 8,
                  numberOfHeads: 6,
                  feedForwardDimension: 1152,
                  architecture: 'modern',
              };
    return { ...base, ...overrides, vocabularySize, architecture: 'modern' };
}
