import { describe, expect, it } from 'vitest';
import { EdrionRuntimeMetrics, type EdrionRuntimeMetricsSnapshot } from '@core';

describe('EdrionRuntimeMetrics', (): void => {
    it('Should record initialization', (): void => {
        const metrics: EdrionRuntimeMetrics = new EdrionRuntimeMetrics();

        metrics.recordInitialization();

        const snapshot: EdrionRuntimeMetricsSnapshot = metrics.snapshot();

        expect(snapshot.initializationCount).toBe(1);
        expect(snapshot.lastInitializedAt).toBeInstanceOf(Date);
        expect(snapshot.failureCount).toBe(0);
    });

    it('Should record shutdown', (): void => {
        const metrics: EdrionRuntimeMetrics = new EdrionRuntimeMetrics();

        metrics.recordInitialization();
        metrics.recordShutdown();

        const snapshot: EdrionRuntimeMetricsSnapshot = metrics.snapshot();

        expect(snapshot.shutdownCount).toBe(1);
        expect(snapshot.lastShutdownAt).toBeInstanceOf(Date);
        expect(snapshot.uptimeMs).toBe(0);
    });

    it('Should record failures', (): void => {
        const metrics: EdrionRuntimeMetrics = new EdrionRuntimeMetrics();

        metrics.recordFailure();

        expect(metrics.snapshot().failureCount).toBe(1);
    });
});
