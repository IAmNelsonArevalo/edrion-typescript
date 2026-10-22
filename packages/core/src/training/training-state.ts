/** Serializable trainer progress state. */
export interface TrainingState {
    readonly epoch: number;
    readonly globalStep: number;
    readonly completed: boolean;
}

/** Creates the initial training state. */
export function createInitialTrainingState(): TrainingState {
    return {
        epoch: 0,
        globalStep: 0,
        completed: false,
    };
}
