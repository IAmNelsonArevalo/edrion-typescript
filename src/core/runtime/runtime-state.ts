/**
 * Possible lifecycle states of an Edrion runtime.
 */
export type EdrionRuntimeStatus = 'created' | 'initializing' | 'ready' | 'stopping' | 'stopped' | 'failed';

/**
 * Controls the Edrion runtime lifecycle.
 */
export class EdrionRuntimeState {
    private status: EdrionRuntimeStatus = 'created';
    private lastError: Error | undefined;

    /**
     * Returns current runtime status.
     */
    getStatus(): EdrionRuntimeStatus {
        return this.status;
    }

    /**
     * Returns the latest runtime failure.
     */
    getLastError(): Error | undefined {
        return this.lastError;
    }

    /**
     * Indicates whether Edrion is ready.
     */
    isReady(): boolean {
        return this.status === 'ready';
    }

    /**
     * Marks runtime initialization.
     */
    markInitializing(): void {
        this.assertTransition(['created', 'stopped', 'failed'], 'initializing');
        this.status = 'initializing';
        this.lastError = undefined;
    }

    /**
     * Marks runtime as ready.
     */
    markReady(): void {
        this.assertTransition(['initializing'], 'ready');
        this.status = 'ready';
    }

    /**
     * Marks runtime shutdown.
     */
    markStopping(): void {
        this.assertTransition(['ready', 'failed'], 'stopping');
        this.status = 'stopping';
    }

    /**
     * Marks runtime as stopped.
     */
    markStopped(): void {
        this.assertTransition(['created', 'stopping'], 'stopped');
        this.status = 'stopped';
    }

    /**
     * Marks runtime as failed.
     */
    markFailed(error: Error): void {
        this.lastError = error;
        this.status = 'failed';
    }

    /**
     * Validates a lifecycle transition.
     */
    private assertTransition(allowed: EdrionRuntimeStatus[], target: EdrionRuntimeStatus): void {
        if (!allowed.includes(this.status)) {
            throw new Error(`Invalid Edrion runtime state transition: ${this.status} -> ${target}.`);
        }
    }
}
