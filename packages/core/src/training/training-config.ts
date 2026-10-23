import { TrainingConfigError } from '@core/training/training.errors';

/** Public trainer configuration. */
export interface TrainingConfig {
    readonly epochs: number;
    readonly batchSize?: number;
    readonly shuffle?: boolean;
    readonly dropLast?: boolean;
    readonly seed?: number;
    readonly gradientAccumulationSteps?: number;
    readonly maxGradientNorm?: number;
    readonly evaluateEveryEpoch?: boolean;
}

/** Normalized trainer configuration. */
export interface NormalizedTrainingConfig {
    readonly epochs: number;
    readonly batchSize: number;
    readonly shuffle: boolean;
    readonly dropLast: boolean;
    readonly seed: number;
    readonly gradientAccumulationSteps: number;
    readonly maxGradientNorm?: number;
    readonly evaluateEveryEpoch: boolean;
}

/** Validates and normalizes trainer configuration. */
export function normalizeTrainingConfig(config: TrainingConfig): NormalizedTrainingConfig {
    assertPositiveInteger(config.epochs, 'Training epochs');
    const batchSize: number = config.batchSize ?? 1;
    assertPositiveInteger(batchSize, 'Training batchSize');
    const seed: number = config.seed ?? 1337;
    if (!Number.isInteger(seed) || !Number.isFinite(seed)) {
        throw new TrainingConfigError(`Training seed must be a finite integer. Received ${seed}.`);
    }
    const gradientAccumulationSteps: number = config.gradientAccumulationSteps ?? 1;
    assertPositiveInteger(gradientAccumulationSteps, 'Training gradientAccumulationSteps');
    const maxGradientNorm: number | undefined = config.maxGradientNorm;
    if (maxGradientNorm !== undefined && (!Number.isFinite(maxGradientNorm) || maxGradientNorm <= 0)) {
        throw new TrainingConfigError(
            `Training maxGradientNorm must be a finite number greater than zero. Received ${maxGradientNorm}.`,
        );
    }
    return {
        epochs: config.epochs,
        batchSize,
        shuffle: config.shuffle ?? false,
        dropLast: config.dropLast ?? false,
        seed,
        gradientAccumulationSteps,
        ...(maxGradientNorm === undefined ? {} : { maxGradientNorm }),
        evaluateEveryEpoch: config.evaluateEveryEpoch ?? true,
    };
}

function assertPositiveInteger(value: number, label: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new TrainingConfigError(`${label} must be a positive integer. Received ${value}.`);
    }
}
