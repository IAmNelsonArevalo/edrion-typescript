/** Metrics produced for one training batch. */
export interface BatchMetrics {
    readonly epoch: number;
    readonly batch: number;
    readonly samples: number;
    readonly loss: number;
    readonly durationMs: number;
}

/** Metrics produced for one training epoch. */
export interface EpochMetrics {
    readonly epoch: number;
    readonly samples: number;
    readonly batches: number;
    readonly averageLoss: number;
    readonly durationMs: number;
}

/** Complete training result. */
export interface TrainingResult {
    readonly epochs: readonly EpochMetrics[];
    readonly totalDurationMs: number;
    readonly finalLoss: number;
}
