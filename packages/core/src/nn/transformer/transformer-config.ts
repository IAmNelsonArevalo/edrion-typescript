import { TransformerConfigurationError } from '@core/nn/transformer/transformer.errors';

export interface TransformerConfig {
    readonly vocabularySize: number;
    readonly contextLength: number;
    readonly embeddingDimension: number;
    readonly numberOfLayers: number;
    readonly numberOfHeads: number;
    readonly feedForwardDimension: number;
    readonly layerNormEpsilon?: number;
    readonly backend?: string;
}

export interface NormalizedTransformerConfig {
    readonly vocabularySize: number;
    readonly contextLength: number;
    readonly embeddingDimension: number;
    readonly numberOfLayers: number;
    readonly numberOfHeads: number;
    readonly feedForwardDimension: number;
    readonly layerNormEpsilon: number;
    readonly backend: string;
}

export function normalizeTransformerConfig(config: TransformerConfig): NormalizedTransformerConfig {
    assertPositiveInteger(config.vocabularySize, 'vocabularySize');
    assertPositiveInteger(config.contextLength, 'contextLength');
    assertPositiveInteger(config.embeddingDimension, 'embeddingDimension');
    assertPositiveInteger(config.numberOfLayers, 'numberOfLayers');
    assertPositiveInteger(config.numberOfHeads, 'numberOfHeads');
    assertPositiveInteger(config.feedForwardDimension, 'feedForwardDimension');

    if (config.embeddingDimension % config.numberOfHeads !== 0) {
        throw new TransformerConfigurationError(
            `embeddingDimension ${config.embeddingDimension} must be divisible by numberOfHeads ${config.numberOfHeads}.`,
        );
    }

    const layerNormEpsilon: number = config.layerNormEpsilon ?? 1e-5;

    if (!Number.isFinite(layerNormEpsilon) || layerNormEpsilon <= 0) {
        throw new TransformerConfigurationError(
            `layerNormEpsilon must be finite and greater than zero. Received ${layerNormEpsilon}.`,
        );
    }

    const backend: string = config.backend?.trim().toLowerCase() ?? 'cpu';

    if (backend.length === 0) {
        throw new TransformerConfigurationError('Transformer backend cannot be empty.');
    }

    return {
        vocabularySize: config.vocabularySize,
        contextLength: config.contextLength,
        embeddingDimension: config.embeddingDimension,
        numberOfLayers: config.numberOfLayers,
        numberOfHeads: config.numberOfHeads,
        feedForwardDimension: config.feedForwardDimension,
        layerNormEpsilon,
        backend,
    };
}

function assertPositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new TransformerConfigurationError(`${name} must be a positive integer. Received ${value}.`);
    }
}
