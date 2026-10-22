import type { EdrionBackendName } from '@core/backend';
import type { EdrionRuntimeMetadata } from '@core/runtime/execution-context';

/**
 * Edrion observability configuration.
 */
export interface EdrionObservabilityOptions {
    /**
     * Enables runtime observability.
     */
    enabled?: boolean;
}

/**
 * Runtime-specific configuration.
 */
export interface EdrionRuntimeOptions {
    /**
     * Metadata associated with this runtime.
     */
    metadata?: EdrionRuntimeMetadata;
}

/**
 * Main public Edrion configuration.
 */
export interface EdrionOptions {
    /**
     * Computational backend used by Edrion.
     */
    backend?: EdrionBackendName;

    /**
     * Runtime observability configuration.
     */
    observability?: EdrionObservabilityOptions;

    /**
     * Runtime-specific configuration.
     */
    runtime?: EdrionRuntimeOptions;
}

/**
 * Internal normalized configuration.
 */
export interface NormalizedEdrionOptions {
    backend: EdrionBackendName;

    observability: {
        enabled: boolean;
    };

    runtime: {
        metadata: EdrionRuntimeMetadata;
    };
}

/**
 * Default Edrion configuration.
 */
export const DEFAULT_EDRION_OPTIONS: Readonly<NormalizedEdrionOptions> = {
    backend: 'cpu',
    observability: {
        enabled: true,
    },
    runtime: {
        metadata: {},
    },
};

/**
 * Normalizes partial Edrion configuration.
 */
export function normalizeEdrionOptions(options: EdrionOptions = {}): NormalizedEdrionOptions {
    return {
        backend: options.backend ?? DEFAULT_EDRION_OPTIONS.backend,
        observability: {
            enabled: options.observability?.enabled ?? DEFAULT_EDRION_OPTIONS.observability.enabled,
        },
        runtime: {
            metadata: {
                ...DEFAULT_EDRION_OPTIONS.runtime.metadata,

                ...options.runtime?.metadata,
            },
        },
    };
}
