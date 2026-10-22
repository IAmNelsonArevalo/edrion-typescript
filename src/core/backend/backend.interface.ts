import type { BackendInitializationContext, EdrionBackendInfo } from '@core/backend/backend.types';

/**
 * Contract that every Edrion computational backend
 * must implement.
 */
export interface EdrionBackend {
    /**
     * Unique backend name.
     */
    readonly name: string;

    /**
     * Initializes backend resources.
     */
    initialize(context: BackendInitializationContext): Promise<void>;

    /**
     * Releases backend resources.
     */
    dispose(): Promise<void>;

    /**
     * Indicates whether the backend is initialized.
     */
    isInitialized(): boolean;

    /**
     * Returns backend information.
     */
    getInfo(): EdrionBackendInfo;
}
