import type { EdrionBackendInfo } from '@core/backend';
import type { EdrionRuntimeMetricsSnapshot } from '@core/metrics';
import type { EdrionRuntimeStatus } from '@core/runtime/runtime-state';

/** High-level production readiness state. */
export type EdrionHealthStatus = 'ready' | 'not-ready' | 'failed';

/** Immutable health report suitable for application health endpoints. */
export interface EdrionHealthReport {
    readonly health: EdrionHealthStatus;
    readonly ready: boolean;
    readonly runtime: EdrionRuntimeStatus;
    readonly backend: EdrionBackendInfo | undefined;
    readonly metrics: EdrionRuntimeMetricsSnapshot;
}

/** Creates a production-oriented runtime health report. */
export function createEdrionHealthReport(
    runtime: EdrionRuntimeStatus,
    backend: EdrionBackendInfo | undefined,
    metrics: EdrionRuntimeMetricsSnapshot,
): EdrionHealthReport {
    const health: EdrionHealthStatus = resolveHealth(runtime, backend);

    return {
        health,
        ready: health === 'ready',
        runtime,
        backend,
        metrics,
    };
}

function resolveHealth(runtime: EdrionRuntimeStatus, backend: EdrionBackendInfo | undefined): EdrionHealthStatus {
    if (runtime.state === 'failed') {
        return 'failed';
    }

    if (runtime.ready && backend !== undefined && backend.initialized) {
        return 'ready';
    }

    return 'not-ready';
}
