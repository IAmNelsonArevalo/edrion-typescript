import type { Optimizer } from '@core/optim/optimizer';
import { OptimizerConfigurationError } from '@core/optim/optimizer.errors';
import { LearningRateScheduler } from './learning-rate-scheduler';

export interface LinearWarmupOptions {
    readonly warmupSteps: number;
    readonly startFactor?: number;
}

/** Linear warmup from startFactor * base LR to base LR. */
export class LinearWarmupScheduler extends LearningRateScheduler {
    private readonly warmupSteps: number;
    private readonly startFactor: number;

    public constructor(optimizer: Optimizer, options: LinearWarmupOptions) {
        super(optimizer);
        if (!Number.isInteger(options.warmupSteps) || options.warmupSteps <= 0) {
            throw new OptimizerConfigurationError(`warmupSteps must be a positive integer. Received ${options.warmupSteps}.`);
        }
        const startFactor: number = options.startFactor ?? 0.1;
        if (!Number.isFinite(startFactor) || startFactor <= 0 || startFactor > 1) {
            throw new OptimizerConfigurationError(`startFactor must be in (0, 1]. Received ${startFactor}.`);
        }
        this.warmupSteps = options.warmupSteps;
        this.startFactor = startFactor;
    }

    protected override getLearningRate(step: number): number {
        if (step >= this.warmupSteps) {
            return this.baseLearningRate;
        }
        const progress: number = step / this.warmupSteps;
        const factor: number = this.startFactor + (1 - this.startFactor) * progress;
        return this.baseLearningRate * factor;
    }
}
