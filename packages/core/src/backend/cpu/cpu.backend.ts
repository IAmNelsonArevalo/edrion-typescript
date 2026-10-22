import type { EdrionBackend } from '@core/backend/backend.interface';
import type { BackendInitializationContext, EdrionBackendInfo, EdrionBackendName } from '@core/backend/backend.types';

/**
 * Default CPU backend.
 *
 * FEATURE-001 defines lifecycle and backend abstraction only. Numeric
 * tensor kernels are introduced in later features.
 */
export class CPUBackend implements EdrionBackend {
    public readonly name: EdrionBackendName = 'cpu';

    private initialized: boolean = false;

    private initializationContext: BackendInitializationContext | undefined = undefined;

    /** Initializes the CPU backend. */
    public async initialize(context: BackendInitializationContext): Promise<void> {
        if (this.initialized) {
            return;
        }

        this.initializationContext = context;
        this.initialized = true;

        await Promise.resolve();
    }

    /** Releases CPU backend resources. */
    public async dispose(): Promise<void> {
        if (!this.initialized) {
            return;
        }

        this.initialized = false;
        this.initializationContext = undefined;

        await Promise.resolve();
    }

    /** Indicates whether the backend is initialized. */
    public isInitialized(): boolean {
        return this.initialized;
    }

    /** Returns backend information. */
    public getInfo(): EdrionBackendInfo {
        return {
            name: this.name,
            initialized: this.initialized,
        };
    }

    /** Returns the initialization context when available. */
    public getInitializationContext(): BackendInitializationContext | undefined {
        return this.initializationContext;
    }
}
