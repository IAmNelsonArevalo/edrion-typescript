/**
 * Immutable snapshot containing runtime metrics.
 */
export interface EdrionRuntimeMetricsSnapshot {
    /**
     * Number of successful runtime initializations.
     */
    initializationCount: number;

    /**
     * Number of successful shutdowns.
     */
    shutdownCount: number;

    /**
     * Number of runtime failures.
     */
    failureCount: number;

    /**
     * Last successful initialization.
     */
    lastInitializedAt: Date | undefined;

    /**
     * Last successful shutdown.
     */
    lastShutdownAt: Date | undefined;

    /**
     * Current runtime uptime in milliseconds.
     */
    uptimeMs: number;
}

/**
 * Tracks Edrion runtime lifecycle metrics.
 *
 * Tensor, training and inference metrics will be
 * introduced by future features.
 */
export class EdrionRuntimeMetrics {
    private initializationCount = 0;
    private shutdownCount = 0;
    private failureCount = 0;
    private lastInitializedAt: Date | undefined;
    private lastShutdownAt: Date | undefined;
    private runningSince: number | undefined;

    /**
     * Records a successful runtime initialization.
     */
    recordInitialization(): void {
        this.initializationCount += 1;
        this.lastInitializedAt = new Date();
        this.runningSince = Date.now();
    }

    /**
     * Records a successful runtime shutdown.
     */
    recordShutdown(): void {
        this.shutdownCount += 1;
        this.lastShutdownAt = new Date();
        this.runningSince = undefined;
    }

    /**
     * Records a runtime failure.
     */
    recordFailure(): void {
        this.failureCount += 1;
    }

    /**
     * Returns a runtime metrics snapshot.
     */
    snapshot(): EdrionRuntimeMetricsSnapshot {
        const uptimeMs: number = this.runningSince === undefined ? 0 : Math.max(0, Date.now() - this.runningSince);

        return {
            initializationCount: this.initializationCount,
            shutdownCount: this.shutdownCount,
            failureCount: this.failureCount,
            lastInitializedAt: this.lastInitializedAt,
            lastShutdownAt: this.lastShutdownAt,
            uptimeMs,
        };
    }
}
