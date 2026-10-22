import type { Loss } from '@core/losses/loss';
import type { Module } from '@core/nn/module';
import type { Optimizer } from '@core/optim/optimizer';
import type { Tensor } from '@core/tensor/tensor';
import {
    normalizeTrainingConfig,
    type NormalizedTrainingConfig,
    type TrainingConfig,
} from '@core/training/training-config';
import { TrainingDataError } from '@core/training/training.errors';
import type { EpochMetrics, TrainingResult } from '@core/training/training-metrics';
import type { TrainingState } from '@core/training/training-state';

/** One supervised training example. */
export interface TrainingSample {
    readonly input: Tensor;
    readonly target: Tensor;
}

/** Optional callbacks emitted during training. */
export interface TrainerCallbacks {
    readonly onEpochEnd?: (metrics: EpochMetrics) => void;
}

/**
 * Generic supervised trainer built on Edrion Module, Loss and Optimizer.
 *
 * Gradient accumulation is used to support logical batch sizes while the
 * current Transformer foundation still processes one sequence at a time.
 */
export class Trainer {
    private readonly config: NormalizedTrainingConfig;
    private stateInternal: TrainingState = {
        epoch: 0,
        globalStep: 0,
        completed: false,
    };

    public constructor(
        private readonly model: Module,
        private readonly optimizer: Optimizer,
        private readonly loss: Loss,
        config: TrainingConfig,
    ) {
        this.config = normalizeTrainingConfig(config);
    }

    /** Returns a copy of current trainer progress. */
    public get state(): TrainingState {
        return {
            ...this.stateInternal,
        };
    }

    /** Restores trainer progress metadata. */
    public restoreState(state: TrainingState): void {
        if (!Number.isInteger(state.epoch) || state.epoch < 0) {
            throw new TrainingDataError(
                `Training state epoch must be a non-negative integer. Received ${state.epoch}.`,
            );
        }

        if (!Number.isInteger(state.globalStep) || state.globalStep < 0) {
            throw new TrainingDataError(
                `Training state globalStep must be a non-negative integer. Received ${state.globalStep}.`,
            );
        }

        this.stateInternal = {
            ...state,
        };
    }

    /** Executes configured training epochs. */
    public train(dataset: readonly TrainingSample[], callbacks: TrainerCallbacks = {}): TrainingResult {
        if (dataset.length === 0) {
            throw new TrainingDataError('Training dataset cannot be empty.');
        }

        this.model.train();

        const startTime: number = Date.now();
        const epochMetrics: EpochMetrics[] = [];

        for (let epochOffset: number = 0; epochOffset < this.config.epochs; epochOffset += 1) {
            const epochNumber: number = this.stateInternal.epoch + 1;
            const metrics: EpochMetrics = this.trainEpoch(dataset, epochNumber);

            this.stateInternal = {
                epoch: epochNumber,
                globalStep: this.stateInternal.globalStep,
                completed: epochOffset === this.config.epochs - 1,
            };

            epochMetrics.push(metrics);
            callbacks.onEpochEnd?.(metrics);
        }

        const finalMetrics: EpochMetrics | undefined = epochMetrics[epochMetrics.length - 1];

        if (finalMetrics === undefined) {
            throw new TrainingDataError('Training did not produce epoch metrics.');
        }

        return {
            epochs: epochMetrics,
            totalDurationMs: Math.max(0, Date.now() - startTime),
            finalLoss: finalMetrics.averageLoss,
        };
    }

    private trainEpoch(dataset: readonly TrainingSample[], epochNumber: number): EpochMetrics {
        const startedAt: number = Date.now();
        let totalLoss: number = 0;
        let batchCount: number = 0;
        let batchStart: number = 0;

        while (batchStart < dataset.length) {
            const batchEnd: number = Math.min(batchStart + this.config.batchSize, dataset.length);
            const batchLength: number = batchEnd - batchStart;

            this.optimizer.zeroGrad();

            for (let sampleIndex: number = batchStart; sampleIndex < batchEnd; sampleIndex += 1) {
                const sample: TrainingSample | undefined = dataset[sampleIndex];

                if (sample === undefined) {
                    throw new TrainingDataError(`Missing training sample at index ${sampleIndex}.`);
                }

                const prediction: Tensor = this.model.forward(sample.input);
                const sampleLoss: Tensor = this.loss.forward(prediction, sample.target);
                const scalarLoss: number = sampleLoss.get();

                if (!Number.isFinite(scalarLoss)) {
                    throw new TrainingDataError(
                        `Training loss is not finite at sample ${sampleIndex}. Received ${scalarLoss}.`,
                    );
                }

                totalLoss += scalarLoss;

                const normalizedLoss: Tensor = sampleLoss.div(batchLength);
                normalizedLoss.backward();
            }

            this.optimizer.step();
            batchCount += 1;
            this.stateInternal = {
                epoch: this.stateInternal.epoch,
                globalStep: this.stateInternal.globalStep + 1,
                completed: false,
            };

            batchStart = batchEnd;
        }

        return {
            epoch: epochNumber,
            samples: dataset.length,
            batches: batchCount,
            averageLoss: totalLoss / dataset.length,
            durationMs: Math.max(0, Date.now() - startedAt),
        };
    }
}
