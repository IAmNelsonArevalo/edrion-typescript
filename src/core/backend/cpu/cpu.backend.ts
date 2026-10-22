import type { EdrionBackend } from '@core/backend/backend.interface';
import type { BackendInitializationContext, EdrionBackendInfo } from '@core/backend/backend.types';

/**
 * Default CPU backend.
 *
 * Feature 001 establishes only its lifecycle.
 *
 * Numerical tensor operations will be introduced
 * in future features.
 */
export class CPUBackend implements EdrionBackend {
    readonly name = 'cpu';

    private initialized = false;

    /**
     * Initializes the CPU backend.
     */
    async initialize(context: BackendInitializationContext): Promise<void> {
        if (this.initialized) return;
        void context;
        this.initialized = true;
        await Promise.resolve();
    }

    /**
     * Releases CPU backend resources.
     */
    async dispose(): Promise<void> {
        this.initialized = false;
        await Promise.resolve();
    }

    /**
     * Indicates whether the CPU backend
     * is currently initialized.
     */
    isInitialized(): boolean {
        return this.initialized;
    }

    /**
     * Returns CPU backend information.
     */
    getInfo(): EdrionBackendInfo {
        return {
            name: this.name,
            device: 'cpu',
            initialized: this.initialized,
        };
    }
}
