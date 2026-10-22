/** Immutable lifecycle metrics exposed by the Edrion runtime. */
export interface EdrionRuntimeMetricsSnapshot {
    readonly initializationCount: number;
    readonly shutdownCount: number;
    readonly failureCount: number;
    readonly lastInitializedAt: Date | undefined;
    readonly lastShutdownAt: Date | undefined;
    readonly uptimeMs: number;
}

/**
 * Collects lifecycle metrics for one Edrion runtime instance.
 */
export class EdrionRuntimeMetrics {
    private initializationCount: number = 0;
    private shutdownCount: number = 0;
    private failureCount: number = 0;
    private lastInitializedAt: Date | undefined = undefined;
    private lastShutdownAt: Date | undefined = undefined;
    private runningSince: number | undefined = undefined;

    /** Records a successful initialization. */
    public recordInitialization(): void {
        const now: Date = new Date();

        this.initializationCount += 1;
        this.lastInitializedAt = now;
        this.runningSince = now.getTime();
    }

    /** Records a successful shutdown. */
    public recordShutdown(): void {
        const now: Date = new Date();

        this.shutdownCount += 1;
        this.lastShutdownAt = now;
        this.runningSince = undefined;
    }

    /** Records a runtime lifecycle failure. */
    public recordFailure(): void {
        this.failureCount += 1;
    }

    /** Returns an immutable metrics snapshot. */
    public snapshot(): EdrionRuntimeMetricsSnapshot {
        return {
            initializationCount: this.initializationCount,
            shutdownCount: this.shutdownCount,
            failureCount: this.failureCount,
            lastInitializedAt: this.lastInitializedAt,
            lastShutdownAt: this.lastShutdownAt,
            uptimeMs: this.calculateUptime(),
        };
    }

    private calculateUptime(): number {
        if (this.runningSince === undefined) {
            return 0;
        }

        const currentTime: number = Date.now();

        return Math.max(0, currentTime - this.runningSince);
    }
}
