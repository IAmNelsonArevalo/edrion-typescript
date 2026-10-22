/**
 * Arbitrary metadata associated with an Edrion runtime.
 */
export type EdrionRuntimeMetadata = Record<string, unknown>;

/**
 * Execution context associated with an Edrion runtime.
 */
export interface EdrionExecutionContext {
    /**
     * Unique runtime identifier.
     */
    runtimeId: string;

    /**
     * Runtime creation timestamp.
     */
    createdAt: Date;

    /**
     * Runtime metadata.
     */
    metadata: Readonly<EdrionRuntimeMetadata>;
}

/**
 * Creates a new Edrion execution context.
 */
export function createExecutionContext(metadata: EdrionRuntimeMetadata = {}): EdrionExecutionContext {
    return {
        runtimeId: globalThis.crypto.randomUUID(),
        createdAt: new Date(),
        metadata: { ...metadata },
    };
}
