/** Runtime lifecycle state. */
export type EdrionRuntimeState = 'created' | 'initializing' | 'ready' | 'stopping' | 'stopped' | 'failed';

/** Public runtime lifecycle status. */
export interface EdrionRuntimeStatus {
    readonly state: EdrionRuntimeState;
    readonly ready: boolean;
}

const RUNTIME_TRANSITIONS: Readonly<Record<EdrionRuntimeState, readonly EdrionRuntimeState[]>> = {
    created: ['initializing'],
    initializing: ['ready', 'failed'],
    ready: ['stopping', 'failed'],
    stopping: ['stopped', 'failed'],
    stopped: ['initializing'],
    failed: ['initializing', 'stopping'],
};

/** Ensures a runtime lifecycle transition is valid. */
export function assertRuntimeStateTransition(current: EdrionRuntimeState, next: EdrionRuntimeState): void {
    const allowedTransitions: readonly EdrionRuntimeState[] = RUNTIME_TRANSITIONS[current];

    if (!allowedTransitions.includes(next)) {
        throw new Error(`Invalid Edrion runtime transition from "${current}" to "${next}".`);
    }
}
