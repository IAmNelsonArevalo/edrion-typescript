import type { BackendInitializationContext, EdrionBackendInfo, EdrionBackendName } from '@core/backend/backend.types';

/**
 * Contract implemented by every Edrion execution backend.
 */
export interface EdrionBackend {
    /** Unique backend identifier. */
    readonly name: EdrionBackendName;

    /** Initializes backend resources for one Edrion runtime. */
    initialize(context: BackendInitializationContext): Promise<void>;

    /** Releases every resource owned by the backend. */
    dispose(): Promise<void>;

    /** Indicates whether the backend is currently initialized. */
    isInitialized(): boolean;

    /** Returns public backend information. */
    getInfo(): EdrionBackendInfo;
}
