import type { Optimizer } from '@core/optim/optimizer';
import { OptimizerConfigurationError } from '@core/optim/optimizer.errors';

/** Base learning-rate scheduler. */
export abstract class LearningRateScheduler {
    protected readonly baseLearningRate: number;
    private lastLearningRateInternal: number;

    protected constructor(protected readonly optimizer: Optimizer) {
        this.baseLearningRate = optimizer.learningRate;
        this.lastLearningRateInternal = optimizer.learningRate;
    }

    public get lastLearningRate(): number {
        return this.lastLearningRateInternal;
    }

    public step(step: number): number {
        if (!Number.isInteger(step) || step <= 0) {
            throw new OptimizerConfigurationError(`Scheduler step must be a positive integer. Received ${step}.`);
        }
        const learningRate: number = this.getLearningRate(step);
        if (!Number.isFinite(learningRate) || learningRate <= 0) {
            throw new OptimizerConfigurationError(`Scheduler produced invalid learning rate ${learningRate} at step ${step}.`);
        }
        this.optimizer.setLearningRate(learningRate);
        this.lastLearningRateInternal = learningRate;
        return learningRate;
    }

    protected abstract getLearningRate(step: number): number;
}
