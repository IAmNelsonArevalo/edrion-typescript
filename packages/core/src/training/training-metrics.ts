/** Metrics produced for one training batch. */
export interface BatchMetrics {
    readonly epoch: number;
    readonly batch: number;
    readonly samples: number;
    readonly loss: number;
    readonly durationMs: number;
    readonly optimizerStep: boolean;
    readonly learningRate: number;
    readonly gradientNorm?: number;
}

/** Evaluation metrics for one validation pass. */
export interface EvaluationMetrics {
    readonly samples: number;
    readonly batches: number;
    readonly averageLoss: number;
    readonly durationMs: number;
}

/** Metrics produced for one training epoch. */
export interface EpochMetrics {
    readonly epoch: number;
    readonly samples: number;
    readonly batches: number;
    readonly averageLoss: number;
    readonly durationMs: number;
    readonly optimizerSteps: number;
    readonly learningRate: number;
    readonly validation?: EvaluationMetrics;
}

/** Complete training result. */
export interface TrainingResult {
    readonly epochs: readonly EpochMetrics[];
    readonly totalDurationMs: number;
    readonly finalLoss: number;
    readonly finalValidationLoss?: number;
}
