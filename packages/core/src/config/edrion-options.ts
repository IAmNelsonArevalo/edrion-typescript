import type { EdrionBackendName } from '@core/backend';

/** Observability configuration. */
export interface EdrionObservabilityOptions {
    readonly enabled?: boolean;
}

/** Metadata associated with one Edrion runtime. */
export interface EdrionRuntimeMetadata {
    readonly application?: string;
    readonly environment?: string;
    readonly [key: string]: string | number | boolean | undefined;
}

/** Runtime configuration. */
export interface EdrionRuntimeOptions {
    readonly metadata?: EdrionRuntimeMetadata;
}

/** Public Edrion configuration. */
export interface EdrionOptions {
    readonly backend?: EdrionBackendName;
    readonly observability?: EdrionObservabilityOptions;
    readonly runtime?: EdrionRuntimeOptions;
}

/** Normalized observability configuration. */
export interface NormalizedEdrionObservabilityOptions {
    readonly enabled: boolean;
}

/** Normalized runtime configuration. */
export interface NormalizedEdrionRuntimeOptions {
    readonly metadata: Readonly<Record<string, string | number | boolean>>;
}

/** Internal normalized Edrion configuration. */
export interface NormalizedEdrionOptions {
    readonly backend: EdrionBackendName;
    readonly observability: NormalizedEdrionObservabilityOptions;
    readonly runtime: NormalizedEdrionRuntimeOptions;
}

/** Normalizes public Edrion options. */
export function normalizeEdrionOptions(options: EdrionOptions = {}): NormalizedEdrionOptions {
    const metadata: Record<string, string | number | boolean> = {};
    const sourceMetadata: EdrionRuntimeMetadata | undefined = options.runtime?.metadata;

    if (sourceMetadata !== undefined) {
        const entries: [string, string | number | boolean | undefined][] = Object.entries(sourceMetadata);

        for (let index: number = 0; index < entries.length; index += 1) {
            const entry: [string, string | number | boolean | undefined] | undefined = entries[index];

            if (entry === undefined) {
                continue;
            }

            const [key, value]: [string, string | number | boolean | undefined] = entry;

            if (value !== undefined) {
                metadata[key] = value;
            }
        }
    }

    return {
        backend: normalizeBackendName(options.backend),
        observability: {
            enabled: options.observability?.enabled ?? false,
        },
        runtime: {
            metadata,
        },
    };
}

function normalizeBackendName(backend: EdrionBackendName | undefined): EdrionBackendName {
    if (backend === undefined) {
        return 'cpu';
    }

    const normalizedBackend: string = backend.trim().toLowerCase();

    if (normalizedBackend.length === 0) {
        throw new Error('Edrion backend cannot be empty.');
    }

    return normalizedBackend;
}
