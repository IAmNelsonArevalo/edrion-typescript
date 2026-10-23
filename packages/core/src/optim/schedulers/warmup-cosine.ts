import type { Optimizer } from '@core/optim/optimizer';
import { OptimizerConfigurationError } from '@core/optim/optimizer.errors';
import { LearningRateScheduler } from './learning-rate-scheduler';

export interface WarmupCosineOptions {
    readonly warmupSteps: number;
    readonly totalSteps: number;
    readonly startFactor?: number;
    readonly minimumLearningRate?: number;
}

/** Linear warmup followed by cosine decay. */
export class WarmupCosineScheduler extends LearningRateScheduler {
    private readonly warmupSteps: number;
    private readonly totalSteps: number;
    private readonly startFactor: number;
    private readonly minimumLearningRate: number;

    public constructor(optimizer: Optimizer, options: WarmupCosineOptions) {
        super(optimizer);
        if (!Number.isInteger(options.warmupSteps) || options.warmupSteps <= 0) {
            throw new OptimizerConfigurationError(
                `warmupSteps must be a positive integer. Received ${options.warmupSteps}.`,
            );
        }
        if (!Number.isInteger(options.totalSteps) || options.totalSteps <= options.warmupSteps) {
            throw new OptimizerConfigurationError(
                `totalSteps must be an integer greater than warmupSteps. Received ${options.totalSteps}.`,
            );
        }
        const startFactor: number = options.startFactor ?? 0.1;
        const minimumLearningRate: number = options.minimumLearningRate ?? this.baseLearningRate * 0.01;
        if (!Number.isFinite(startFactor) || startFactor <= 0 || startFactor > 1) {
            throw new OptimizerConfigurationError(`startFactor must be in (0, 1]. Received ${startFactor}.`);
        }
        if (
            !Number.isFinite(minimumLearningRate) ||
            minimumLearningRate <= 0 ||
            minimumLearningRate > this.baseLearningRate
        ) {
            throw new OptimizerConfigurationError(
                `minimumLearningRate must be in (0, baseLearningRate]. Received ${minimumLearningRate}.`,
            );
        }
        this.warmupSteps = options.warmupSteps;
        this.totalSteps = options.totalSteps;
        this.startFactor = startFactor;
        this.minimumLearningRate = minimumLearningRate;
    }

    protected override getLearningRate(step: number): number {
        if (step <= this.warmupSteps) {
            const progress: number = step / this.warmupSteps;
            const factor: number = this.startFactor + (1 - this.startFactor) * progress;
            return this.baseLearningRate * factor;
        }
        const decaySteps: number = this.totalSteps - this.warmupSteps;
        const decayStep: number = Math.min(step - this.warmupSteps, decaySteps);
        const progress: number = decayStep / decaySteps;
        const cosine: number = 0.5 * (1 + Math.cos(Math.PI * progress));
        return this.minimumLearningRate + (this.baseLearningRate - this.minimumLearningRate) * cosine;
    }
}
