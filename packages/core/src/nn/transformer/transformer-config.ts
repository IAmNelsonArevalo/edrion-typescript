import { TransformerConfigurationError } from '@core/nn/transformer/transformer.errors';

export type TransformerArchitecture = 'legacy' | 'modern';

export interface TransformerConfig {
    readonly vocabularySize: number;
    readonly contextLength: number;
    readonly embeddingDimension: number;
    readonly numberOfLayers: number;
    readonly numberOfHeads: number;
    readonly feedForwardDimension: number;
    readonly layerNormEpsilon?: number;
    readonly backend?: string;
    /** FEATURE-014 architecture mode. Defaults to legacy for v1.x compatibility. */
    readonly architecture?: TransformerArchitecture;
    /** RoPE base used by modern attention. */
    readonly rotaryBase?: number;
    /** Number of dimensions per head participating in RoPE. Defaults to full head dimension. */
    readonly rotaryDimension?: number;
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
    readonly architecture: TransformerArchitecture;
    readonly rotaryBase: number;
    readonly rotaryDimension: number;
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

    const architecture: TransformerArchitecture = config.architecture ?? 'legacy';
    const layerNormEpsilon: number = config.layerNormEpsilon ?? (architecture === 'modern' ? 1e-6 : 1e-5);

    if (!Number.isFinite(layerNormEpsilon) || layerNormEpsilon <= 0) {
        throw new TransformerConfigurationError(
            `layerNormEpsilon must be finite and greater than zero. Received ${layerNormEpsilon}.`,
        );
    }

    const backend: string = config.backend?.trim().toLowerCase() ?? 'cpu';
    if (backend.length === 0) {
        throw new TransformerConfigurationError('Transformer backend cannot be empty.');
    }

    const headDimension: number = config.embeddingDimension / config.numberOfHeads;
    const rotaryBase: number = config.rotaryBase ?? 10_000;
    const rotaryDimension: number = config.rotaryDimension ?? headDimension;

    if (!Number.isFinite(rotaryBase) || rotaryBase <= 1) {
        throw new TransformerConfigurationError(`rotaryBase must be finite and greater than 1. Received ${rotaryBase}.`);
    }

    if (
        !Number.isInteger(rotaryDimension) ||
        rotaryDimension <= 0 ||
        rotaryDimension > headDimension ||
        rotaryDimension % 2 !== 0
    ) {
        throw new TransformerConfigurationError(
            `rotaryDimension must be a positive even integer <= headDimension ${headDimension}. Received ${rotaryDimension}.`,
        );
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
        architecture,
        rotaryBase,
        rotaryDimension,
    };
}

function assertPositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new TransformerConfigurationError(`${name} must be a positive integer. Received ${value}.`);
    }
}
