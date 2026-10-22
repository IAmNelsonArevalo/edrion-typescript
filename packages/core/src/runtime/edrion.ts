import {
    createDefaultBackendRegistry,
    type BackendRegistry,
    type EdrionBackend,
    type EdrionBackendInfo,
} from '@core/backend';
import { normalizeEdrionOptions, type EdrionOptions, type NormalizedEdrionOptions } from '@core/config/edrion-options';
import { EdrionRuntimeMetrics, type EdrionRuntimeMetricsSnapshot } from '@core/metrics';
import { createExecutionContext, type EdrionExecutionContext } from '@core/runtime/execution-context';
import {
    assertRuntimeStateTransition,
    type EdrionRuntimeState,
    type EdrionRuntimeStatus,
} from '@core/runtime/runtime-state';

/** Optional dependencies used by the Edrion runtime. */
export interface EdrionRuntimeDependencies {
    readonly backendRegistry?: BackendRegistry;
}

/** Complete public diagnostic information for one runtime. */
export interface EdrionRuntimeInfo {
    readonly version: string;
    readonly executionContext: EdrionExecutionContext;
    readonly status: EdrionRuntimeStatus;
    readonly backend: EdrionBackendInfo | undefined;
    readonly metrics: EdrionRuntimeMetricsSnapshot;
}

/**
 * Main framework-independent Edrion runtime.
 *
 * FEATURE-001 introduces backend abstraction, lifecycle states,
 * execution context and runtime metrics.
 */
export class Edrion {
    private static readonly VERSION: string = '0.7.0';

    private readonly options: NormalizedEdrionOptions;
    private readonly backendRegistry: BackendRegistry;
    private readonly executionContext: EdrionExecutionContext;
    private readonly metrics: EdrionRuntimeMetrics = new EdrionRuntimeMetrics();

    private state: EdrionRuntimeState = 'created';
    private backend: EdrionBackend | undefined = undefined;

    public constructor(options: EdrionOptions = {}, dependencies: EdrionRuntimeDependencies = {}) {
        this.options = normalizeEdrionOptions(options);
        this.backendRegistry = dependencies.backendRegistry ?? createDefaultBackendRegistry();
        this.executionContext = createExecutionContext(this.options.runtime.metadata);
    }

    /** Initializes Edrion and its selected backend. */
    public async initialize(): Promise<void> {
        if (this.state === 'ready') {
            return;
        }

        this.transitionTo('initializing');

        try {
            const backend: EdrionBackend = this.backendRegistry.create(this.options.backend);
            const initializedAt: Date = new Date();

            await backend.initialize({
                runtimeId: this.executionContext.runtimeId,
                initializedAt,
            });

            this.backend = backend;
            this.transitionTo('ready');
            this.metrics.recordInitialization();
        } catch (error: unknown) {
            this.metrics.recordFailure();
            this.state = 'failed';
            throw error;
        }
    }

    /** Gracefully shuts the runtime and active backend down. */
    public async shutdown(): Promise<void> {
        if (this.state === 'created' || this.state === 'stopped') {
            return;
        }

        this.transitionTo('stopping');

        try {
            const activeBackend: EdrionBackend | undefined = this.backend;

            if (activeBackend !== undefined) {
                await activeBackend.dispose();
            }

            this.backend = undefined;
            this.transitionTo('stopped');
            this.metrics.recordShutdown();
        } catch (error: unknown) {
            this.metrics.recordFailure();
            this.state = 'failed';
            throw error;
        }
    }

    /** Returns normalized runtime configuration. */
    public getOptions(): NormalizedEdrionOptions {
        return this.options;
    }

    /** Returns the runtime execution context. */
    public getExecutionContext(): EdrionExecutionContext {
        return this.executionContext;
    }

    /** Returns the active backend when initialized. */
    public getBackend(): EdrionBackend | undefined {
        return this.backend;
    }

    /** Returns the backend registry used by this runtime. */
    public getBackendRegistry(): BackendRegistry {
        return this.backendRegistry;
    }

    /** Returns current runtime lifecycle status. */
    public getStatus(): EdrionRuntimeStatus {
        const ready: boolean = this.state === 'ready';

        return {
            state: this.state,
            ready,
        };
    }

    /**
     * Indicates whether initialization completed successfully.
     * Kept for backward compatibility with FEATURE-000.
     */
    public isInitialized(): boolean {
        return this.isReady();
    }

    /** Indicates whether the runtime is ready to execute work. */
    public isReady(): boolean {
        return this.state === 'ready';
    }

    /** Returns current runtime metrics. */
    public getMetrics(): EdrionRuntimeMetricsSnapshot {
        return this.metrics.snapshot();
    }

    /** Returns complete runtime diagnostics. */
    public getRuntimeInfo(): EdrionRuntimeInfo {
        const activeBackend: EdrionBackend | undefined = this.backend;
        const backendInfo: EdrionBackendInfo | undefined =
            activeBackend === undefined ? undefined : activeBackend.getInfo();
        const status: EdrionRuntimeStatus = this.getStatus();
        const metrics: EdrionRuntimeMetricsSnapshot = this.getMetrics();

        return {
            version: Edrion.VERSION,
            executionContext: this.executionContext,
            status,
            backend: backendInfo,
            metrics,
        };
    }

    /** Returns the framework version. */
    public getVersion(): string {
        return Edrion.VERSION;
    }

    private transitionTo(nextState: EdrionRuntimeState): void {
        assertRuntimeStateTransition(this.state, nextState);
        this.state = nextState;
    }
}
