import { ArrayDataset, DataLoader, type DataBatch, type Dataset } from '@core/data';
import type { Loss } from '@core/losses/loss';
import type { Module } from '@core/nn/module';
import type { Optimizer } from '@core/optim/optimizer';
import { Tensor } from '@core/tensor/tensor';
import {
    normalizeTrainingConfig,
    type NormalizedTrainingConfig,
    type TrainingConfig,
} from '@core/training/training-config';
import { TrainingDataError } from '@core/training/training.errors';
import type { BatchMetrics, EpochMetrics, TrainingResult } from '@core/training/training-metrics';
import type { TrainingState } from '@core/training/training-state';

/** One supervised training example. */
export interface TrainingSample {
    readonly input: Tensor;
    readonly target: Tensor;
}

/** Dataset forms accepted by Trainer. */
export type TrainingDataSource = readonly TrainingSample[] | Dataset<TrainingSample> | DataLoader<TrainingSample>;

/** Optional callbacks emitted during training. */
export interface TrainerCallbacks {
    readonly onBatchEnd?: (metrics: BatchMetrics) => void;
    readonly onEpochEnd?: (metrics: EpochMetrics) => void;
}

/**
 * Generic supervised trainer built on Edrion Module, Loss and Optimizer.
 *
 * FEATURE-012 executes real tensor batches through Tensor.stack() and DataLoader
 * instead of simulating batches exclusively through gradient accumulation.
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
    public train(source: TrainingDataSource, callbacks: TrainerCallbacks = {}): TrainingResult {
        const loader: DataLoader<TrainingSample> = this.resolveDataLoader(source);

        if (loader.dataset.length === 0) {
            throw new TrainingDataError('Training dataset cannot be empty.');
        }

        if (loader.batchCount === 0) {
            throw new TrainingDataError(
                'Training DataLoader does not emit any batches. Disable dropLast or reduce batchSize.',
            );
        }

        this.model.train();

        const startTime: number = Date.now();
        const epochMetrics: EpochMetrics[] = [];

        for (let epochOffset: number = 0; epochOffset < this.config.epochs; epochOffset += 1) {
            const epochNumber: number = this.stateInternal.epoch + 1;
            const metrics: EpochMetrics = this.trainEpoch(loader, epochNumber, callbacks);

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

    private trainEpoch(
        loader: DataLoader<TrainingSample>,
        epochNumber: number,
        callbacks: TrainerCallbacks,
    ): EpochMetrics {
        const startedAt: number = Date.now();
        let weightedLoss: number = 0;
        let processedSamples: number = 0;
        let batchCount: number = 0;

        for (const batch of loader.batches(epochNumber - 1)) {
            const batchMetrics: BatchMetrics = this.trainBatch(batch, epochNumber);

            weightedLoss += batchMetrics.loss * batchMetrics.samples;
            processedSamples += batchMetrics.samples;
            batchCount += 1;
            callbacks.onBatchEnd?.(batchMetrics);
        }

        if (processedSamples === 0) {
            throw new TrainingDataError(`Epoch ${epochNumber} did not process any samples.`);
        }

        return {
            epoch: epochNumber,
            samples: processedSamples,
            batches: batchCount,
            averageLoss: weightedLoss / processedSamples,
            durationMs: Math.max(0, Date.now() - startedAt),
        };
    }

    private trainBatch(batch: DataBatch<TrainingSample>, epochNumber: number): BatchMetrics {
        if (batch.samples.length === 0) {
            throw new TrainingDataError(`Training batch ${batch.batchIndex} cannot be empty.`);
        }

        const startedAt: number = Date.now();
        const inputs: Tensor[] = new Array<Tensor>(batch.samples.length);
        const targets: Tensor[] = new Array<Tensor>(batch.samples.length);

        for (let index: number = 0; index < batch.samples.length; index += 1) {
            const sample: TrainingSample | undefined = batch.samples[index];

            if (sample === undefined) {
                throw new TrainingDataError(`Missing training sample at batch offset ${index}.`);
            }

            inputs[index] = sample.input;
            targets[index] = sample.target;
        }

        const batchInput: Tensor = Tensor.stack(inputs);
        const batchTarget: Tensor = Tensor.stack(targets);

        this.optimizer.zeroGrad();

        const prediction: Tensor = this.model.forward(batchInput);
        const batchLoss: Tensor = this.loss.forward(prediction, batchTarget);
        const scalarLoss: number = batchLoss.get();

        if (!Number.isFinite(scalarLoss)) {
            throw new TrainingDataError(
                `Training loss is not finite at epoch ${epochNumber}, batch ${batch.batchIndex}. Received ${scalarLoss}.`,
            );
        }

        batchLoss.backward();
        this.optimizer.step();

        this.stateInternal = {
            epoch: this.stateInternal.epoch,
            globalStep: this.stateInternal.globalStep + 1,
            completed: false,
        };

        return {
            epoch: epochNumber,
            batch: batch.batchIndex,
            samples: batch.samples.length,
            loss: scalarLoss,
            durationMs: Math.max(0, Date.now() - startedAt),
        };
    }

    private resolveDataLoader(source: TrainingDataSource): DataLoader<TrainingSample> {
        if (source instanceof DataLoader) {
            return source;
        }

        const dataset: Dataset<TrainingSample> = Array.isArray(source)
            ? new ArrayDataset(source)
            : (source as Dataset<TrainingSample>);

        return new DataLoader(dataset, {
            batchSize: this.config.batchSize,
            shuffle: this.config.shuffle,
            dropLast: this.config.dropLast,
            seed: this.config.seed,
        });
    }
}
