import type { Optimizer } from '@core/optim/optimizer';
import { OptimizerConfigurationError } from '@core/optim/optimizer.errors';
import { LearningRateScheduler } from './learning-rate-scheduler';

export interface CosineDecayOptions {
    readonly totalSteps: number;
    readonly minimumLearningRate?: number;
}

/** Cosine learning-rate decay. */
export class CosineDecayScheduler extends LearningRateScheduler {
    private readonly totalSteps: number;
    private readonly minimumLearningRate: number;

    public constructor(optimizer: Optimizer, options: CosineDecayOptions) {
        super(optimizer);
        if (!Number.isInteger(options.totalSteps) || options.totalSteps <= 0) {
            throw new OptimizerConfigurationError(
                `totalSteps must be a positive integer. Received ${options.totalSteps}.`,
            );
        }
        const minimumLearningRate: number = options.minimumLearningRate ?? this.baseLearningRate * 0.01;
        if (
            !Number.isFinite(minimumLearningRate) ||
            minimumLearningRate <= 0 ||
            minimumLearningRate > this.baseLearningRate
        ) {
            throw new OptimizerConfigurationError(
                `minimumLearningRate must be in (0, baseLearningRate]. Received ${minimumLearningRate}.`,
            );
        }
        this.totalSteps = options.totalSteps;
        this.minimumLearningRate = minimumLearningRate;
    }

    protected override getLearningRate(step: number): number {
        const clampedStep: number = Math.min(step, this.totalSteps);
        const progress: number = clampedStep / this.totalSteps;
        const cosine: number = 0.5 * (1 + Math.cos(Math.PI * progress));
        return this.minimumLearningRate + (this.baseLearningRate - this.minimumLearningRate) * cosine;
    }
}
