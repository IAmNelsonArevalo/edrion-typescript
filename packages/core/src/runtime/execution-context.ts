import type { EdrionRuntimeMetadata } from '@core/config/edrion-options';

/** Runtime execution context created once per Edrion instance. */
export interface EdrionExecutionContext {
    readonly runtimeId: string;
    readonly createdAt: Date;
    readonly metadata: Readonly<Record<string, string | number | boolean>>;
}

/** Creates an immutable execution context for one runtime instance. */
export function createExecutionContext(metadata: EdrionRuntimeMetadata): EdrionExecutionContext {
    const runtimeId: string = createRuntimeId();
    const createdAt: Date = new Date();
    const normalizedMetadata: Record<string, string | number | boolean> = {};
    const entries: [string, string | number | boolean | undefined][] = Object.entries(metadata);

    for (let index: number = 0; index < entries.length; index += 1) {
        const entry: [string, string | number | boolean | undefined] | undefined = entries[index];

        if (entry === undefined) {
            continue;
        }

        const [key, value]: [string, string | number | boolean | undefined] = entry;

        if (value !== undefined) {
            normalizedMetadata[key] = value;
        }
    }

    return {
        runtimeId,
        createdAt,
        metadata: normalizedMetadata,
    };
}

function createRuntimeId(): string {
    const cryptoApi: Crypto | undefined = globalThis.crypto;

    if (cryptoApi === undefined || typeof cryptoApi.randomUUID !== 'function') {
        throw new Error('Edrion requires crypto.randomUUID() support to create runtime identifiers.');
    }

    return cryptoApi.randomUUID();
}
