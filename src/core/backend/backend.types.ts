/**
 * Backend name used by the Edrion runtime.
 *
 * The first built-in backend is "cpu".
 * Future implementations may include WASM and WebGPU.
 */
export type EdrionBackendName = string;

/**
 * Public information exposed by a computational backend.
 */
export interface EdrionBackendInfo {
    /**
     * Unique backend identifier.
     */
    name: string;

    /**
     * Device used for computation.
     */
    device: string;

    /**
     * Indicates whether the backend is initialized.
     */
    initialized: boolean;
}

/**
 * Context provided to a backend during initialization.
 */
export interface BackendInitializationContext {
    /**
     * Unique Edrion runtime identifier.
     */
    runtimeId: string;

    /**
     * Indicates whether observability is enabled.
     */
    observabilityEnabled: boolean;

    /**
     * Runtime metadata.
     */
    metadata: Readonly<Record<string, unknown>>;
}
