import { TrainingConfigError } from '@core/training/training.errors';

/** Public trainer configuration. */
export interface TrainingConfig {
    readonly epochs: number;
    readonly batchSize?: number;
}

/** Normalized trainer configuration. */
export interface NormalizedTrainingConfig {
    readonly epochs: number;
    readonly batchSize: number;
}

/** Validates and normalizes trainer configuration. */
export function normalizeTrainingConfig(config: TrainingConfig): NormalizedTrainingConfig {
    assertPositiveInteger(config.epochs, 'Training epochs');

    const batchSize: number = config.batchSize ?? 1;
    assertPositiveInteger(batchSize, 'Training batchSize');

    return {
        epochs: config.epochs,
        batchSize,
    };
}

function assertPositiveInteger(value: number, label: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new TrainingConfigError(`${label} must be a positive integer. Received ${value}.`);
    }
}
