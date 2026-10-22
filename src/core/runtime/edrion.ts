import { BackendRegistry, createDefaultBackendRegistry, type EdrionBackend, type EdrionBackendInfo } from '@core/backend';
import { type EdrionOptions, type NormalizedEdrionOptions, normalizeEdrionOptions } from '@core/config/edrion-options';
import { EdrionRuntimeMetrics, type EdrionRuntimeMetricsSnapshot } from '@core/metrics';
import { createExecutionContext, type EdrionExecutionContext } from '@core/runtime/execution-context';
import { EdrionRuntimeState, type EdrionRuntimeStatus } from '@core/runtime/runtime-state';

/**
 * Dependencies that can be provided to an Edrion runtime.
 */
export interface EdrionRuntimeDependencies {
    /**
     * Optional custom backend registry.
     */
    backendRegistry?: BackendRegistry;
}

/**
 * Public runtime information.
 */
export interface EdrionRuntimeInfo {
    /**
     * Runtime name.
     */
    name: string;

    /**
     * Current Edrion package version.
     */
    version: string;

    /**
     * Unique runtime identifier.
     */
    runtimeId: string;

    /**
     * Current lifecycle status.
     */
    status: EdrionRuntimeStatus;

    /**
     * Active backend information.
     */
    backend: EdrionBackendInfo | null;

    /**
     * Indicates whether observability is enabled.
     */
    observabilityEnabled: boolean;

    /**
     * Runtime creation date.
     */
    createdAt: Date;

    /**
     * Runtime metrics.
     */
    metrics: EdrionRuntimeMetricsSnapshot;
}

/**
 * Framework-independent Edrion runtime.
 *
 * Framework integrations such as NestJS must consume
 * this runtime rather than implementing AI core behavior
 * directly.
 */
export class Edrion {
    private readonly options: NormalizedEdrionOptions;
    private readonly state = new EdrionRuntimeState();
    private readonly metrics = new EdrionRuntimeMetrics();
    private readonly context: EdrionExecutionContext;
    private readonly backendRegistry: BackendRegistry;
    private backend: EdrionBackend | undefined;

    /**
     * Creates an Edrion runtime.
     */
    constructor(options: EdrionOptions = {}, dependencies: EdrionRuntimeDependencies = {}) {
        this.options = normalizeEdrionOptions(options);
        this.context = createExecutionContext(this.options.runtime.metadata);
        this.backendRegistry = dependencies.backendRegistry ?? createDefaultBackendRegistry();
    }

    /**
     * Initializes Edrion and its configured backend.
     */
    async initialize(): Promise<void> {
        if (this.state.isReady()) return;
        this.state.markInitializing();

        try {
            const backend = this.backendRegistry.create(this.options.backend);

            await backend.initialize({
                runtimeId: this.context.runtimeId,
                observabilityEnabled: this.options.observability.enabled,
                metadata: this.context.metadata,
            });

            this.backend = backend;
            this.state.markReady();
            this.metrics.recordInitialization();
        } catch (error) {
            const runtimeError = this.normalizeError(error);
            this.metrics.recordFailure();
            this.state.markFailed(runtimeError);
            throw runtimeError;
        }
    }

    /**
     * Gracefully shuts down Edrion.
     */
    async shutdown(): Promise<void> {
        const status = this.state.getStatus();

        if (status === 'stopped') return;
        if (status === 'created') return this.state.markStopped();

        this.state.markStopping();

        try {
            if (this.backend?.isInitialized()) await this.backend.dispose();

            this.state.markStopped();
            this.metrics.recordShutdown();
        } catch (error) {
            const runtimeError = this.normalizeError(error);
            this.metrics.recordFailure();
            this.state.markFailed(runtimeError);
            throw runtimeError;
        }
    }

    /**
     * Returns normalized Edrion options.
     */
    getOptions(): Readonly<NormalizedEdrionOptions> {
        return this.options;
    }

    /**
     * Returns runtime execution context.
     */
    getExecutionContext(): Readonly<EdrionExecutionContext> {
        return this.context;
    }

    /**
     * Returns current computational backend.
     */
    getBackend(): EdrionBackend | undefined {
        return this.backend;
    }

    /**
     * Returns the backend registry.
     */
    getBackendRegistry(): BackendRegistry {
        return this.backendRegistry;
    }

    /**
     * Returns current runtime status.
     */
    getStatus(): EdrionRuntimeStatus {
        return this.state.getStatus();
    }

    /**
     * Indicates whether Edrion is ready.
     */
    isReady(): boolean {
        return this.state.isReady();
    }

    /**
     * Returns runtime metrics.
     */
    getMetrics(): EdrionRuntimeMetricsSnapshot {
        return this.metrics.snapshot();
    }

    /**
     * Returns runtime information.
     */
    getRuntimeInfo(): EdrionRuntimeInfo {
        return {
            name: 'edrion-typescript',
            version: '0.2.0',
            runtimeId: this.context.runtimeId,
            status: this.state.getStatus(),
            backend: this.backend?.getInfo() ?? null,
            observabilityEnabled: this.options.observability.enabled,
            createdAt: this.context.createdAt,
            metrics: this.metrics.snapshot(),
        };
    }

    /**
     * Converts unknown thrown values into Error instances.
     */
    private normalizeError(error: unknown): Error {
        if (error instanceof Error) return error;
        return new Error(String(error));
    }
}
