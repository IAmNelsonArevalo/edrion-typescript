import { ArrayDataset, DataLoader, type DataBatch, type Dataset } from '@core/data';
import type { Loss } from '@core/losses/loss';
import type { Module } from '@core/nn/module';
import type { LearningRateScheduler } from '@core/optim/schedulers';
import type { Optimizer } from '@core/optim/optimizer';
import type { LossScaler, PrecisionPolicy } from '@core/precision';
import { clipGradientsByGlobalNorm, computeGlobalGradientNorm } from '@core/stability';
import { Tensor } from '@core/tensor/tensor';
import {
    normalizeTrainingConfig,
    type NormalizedTrainingConfig,
    type TrainingConfig,
} from '@core/training/training-config';
import { TrainingDataError } from '@core/training/training.errors';
import type { BatchMetrics, EpochMetrics, EvaluationMetrics, TrainingResult } from '@core/training/training-metrics';
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
    readonly onEvaluationEnd?: (metrics: EvaluationMetrics, epoch: number) => void;
}

/** Optional advanced training integrations. */
export interface TrainerOptions {
    readonly scheduler?: LearningRateScheduler;
    readonly validationData?: TrainingDataSource;
    readonly precisionPolicy?: PrecisionPolicy;
    readonly lossScaler?: LossScaler;
}

/** Advanced supervised trainer with accumulation, scheduling and evaluation. */
export class Trainer {
    private readonly config: NormalizedTrainingConfig;
    private readonly scheduler: LearningRateScheduler | undefined;
    private readonly validationData: TrainingDataSource | undefined;
    private readonly precisionPolicy: PrecisionPolicy | undefined;
    private readonly lossScaler: LossScaler | undefined;
    private stateInternal: TrainingState = { epoch: 0, globalStep: 0, completed: false };

    public constructor(
        private readonly model: Module,
        private readonly optimizer: Optimizer,
        private readonly loss: Loss,
        config: TrainingConfig,
        options: TrainerOptions = {},
    ) {
        this.config = normalizeTrainingConfig(config);
        this.scheduler = options.scheduler;
        this.validationData = options.validationData;
        this.precisionPolicy = options.precisionPolicy;
        this.lossScaler = options.lossScaler;
    }

    public get state(): TrainingState {
        return { ...this.stateInternal };
    }

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
        this.stateInternal = { ...state };
    }

    public train(source: TrainingDataSource, callbacks: TrainerCallbacks = {}): TrainingResult {
        const loader: DataLoader<TrainingSample> = this.resolveDataLoader(source);
        this.assertUsableLoader(loader, 'Training');
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
            ...(finalMetrics.validation === undefined
                ? {}
                : { finalValidationLoss: finalMetrics.validation.averageLoss }),
        };
    }

    public evaluate(source: TrainingDataSource): EvaluationMetrics {
        const loader: DataLoader<TrainingSample> = this.resolveDataLoader(source);
        this.assertUsableLoader(loader, 'Evaluation');
        const wasTraining: boolean = this.model.training;
        this.model.eval();
        const startedAt: number = Date.now();
        let weightedLoss: number = 0;
        let samples: number = 0;
        let batches: number = 0;

        try {
            for (const batch of loader.batches(0)) {
                const { input, target } = this.stackBatch(batch);
                const modelInput: Tensor = this.precisionPolicy?.castInput(input) ?? input;
                const predictionRaw: Tensor = this.model.forward(modelInput);
                const prediction: Tensor = this.precisionPolicy?.castOutput(predictionRaw) ?? predictionRaw;
                const batchLoss: Tensor = this.loss.forward(prediction, target);
                const scalarLoss: number = batchLoss.get();
                if (!Number.isFinite(scalarLoss)) {
                    throw new TrainingDataError(
                        `Evaluation loss is not finite at batch ${batch.batchIndex}. Received ${scalarLoss}.`,
                    );
                }
                weightedLoss += scalarLoss * batch.samples.length;
                samples += batch.samples.length;
                batches += 1;
            }
        } finally {
            if (wasTraining) {
                this.model.train();
            }
        }

        if (samples === 0) {
            throw new TrainingDataError('Evaluation did not process any samples.');
        }
        return {
            samples,
            batches,
            averageLoss: weightedLoss / samples,
            durationMs: Math.max(0, Date.now() - startedAt),
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
        let optimizerSteps: number = 0;
        let accumulationCount: number = 0;

        this.optimizer.zeroGrad();

        for (const batch of loader.batches(epochNumber - 1)) {
            const isLastBatch: boolean = batch.batchIndex === loader.batchCount - 1;
            accumulationCount += 1;
            const shouldStep: boolean = accumulationCount >= this.config.gradientAccumulationSteps || isLastBatch;
            const batchMetrics: BatchMetrics = this.trainBatch(batch, epochNumber, shouldStep, accumulationCount);
            weightedLoss += batchMetrics.loss * batchMetrics.samples;
            processedSamples += batchMetrics.samples;
            batchCount += 1;
            if (batchMetrics.optimizerStep) {
                optimizerSteps += 1;
                accumulationCount = 0;
            }
            callbacks.onBatchEnd?.(batchMetrics);
        }

        if (processedSamples === 0) {
            throw new TrainingDataError(`Epoch ${epochNumber} did not process any samples.`);
        }

        let validation: EvaluationMetrics | undefined;
        if (this.validationData !== undefined && this.config.evaluateEveryEpoch) {
            validation = this.evaluate(this.validationData);
            callbacks.onEvaluationEnd?.(validation, epochNumber);
            this.model.train();
        }

        return {
            epoch: epochNumber,
            samples: processedSamples,
            batches: batchCount,
            averageLoss: weightedLoss / processedSamples,
            durationMs: Math.max(0, Date.now() - startedAt),
            optimizerSteps,
            learningRate: this.optimizer.learningRate,
            ...(validation === undefined ? {} : { validation }),
        };
    }

    private trainBatch(
        batch: DataBatch<TrainingSample>,
        epochNumber: number,
        shouldStep: boolean,
        accumulationCount: number,
    ): BatchMetrics {
        if (batch.samples.length === 0) {
            throw new TrainingDataError(`Training batch ${batch.batchIndex} cannot be empty.`);
        }
        const startedAt: number = Date.now();
        const { input, target } = this.stackBatch(batch);
        const modelInput: Tensor = this.precisionPolicy?.castInput(input) ?? input;
        const predictionRaw: Tensor = this.model.forward(modelInput);
        const prediction: Tensor = this.precisionPolicy?.castOutput(predictionRaw) ?? predictionRaw;
        const batchLoss: Tensor = this.loss.forward(prediction, target);
        const scalarLoss: number = batchLoss.get();
        if (!Number.isFinite(scalarLoss)) {
            throw new TrainingDataError(
                `Training loss is not finite at epoch ${epochNumber}, batch ${batch.batchIndex}. Received ${scalarLoss}.`,
            );
        }
        const backwardLoss: Tensor = this.lossScaler?.scaleLoss(batchLoss) ?? batchLoss;
        backwardLoss.backward();

        let gradientNorm: number | undefined;
        let skippedOptimizerStep: boolean = false;
        let lossScale: number | undefined = this.lossScaler?.scale;
        if (shouldStep) {
            const foundNonFinite: boolean = this.lossScaler?.unscale(this.optimizer.parameters) ?? false;
            if (foundNonFinite) {
                this.optimizer.zeroGrad();
                const update = this.lossScaler?.update(true);
                lossScale = update?.scale;
                skippedOptimizerStep = true;
            }
            if (!skippedOptimizerStep && accumulationCount > 1) {
                const scale: number = 1 / accumulationCount;
                const parameters = this.optimizer.parameters;
                for (let index: number = 0; index < parameters.length; index += 1) {
                    const parameter = parameters[index];
                    if (parameter === undefined) {
                        throw new TrainingDataError(`Missing optimizer parameter at index ${index}.`);
                    }
                    parameter.scaleGradient(scale);
                }
            }
            if (!skippedOptimizerStep) {
                if (this.config.maxGradientNorm !== undefined) {
                    gradientNorm = clipGradientsByGlobalNorm(
                        this.optimizer.parameters,
                        this.config.maxGradientNorm,
                    ).totalNorm;
                } else {
                    gradientNorm = computeGlobalGradientNorm(this.optimizer.parameters);
                }
                const nextStep: number = this.stateInternal.globalStep + 1;
                this.scheduler?.step(nextStep);
                this.optimizer.step();
                this.optimizer.zeroGrad();
                this.lossScaler?.update(false);
                lossScale = this.lossScaler?.scale;
                this.stateInternal = { epoch: this.stateInternal.epoch, globalStep: nextStep, completed: false };
            }
        }

        return {
            epoch: epochNumber,
            batch: batch.batchIndex,
            samples: batch.samples.length,
            loss: scalarLoss,
            durationMs: Math.max(0, Date.now() - startedAt),
            optimizerStep: shouldStep && !skippedOptimizerStep,
            learningRate: this.optimizer.learningRate,
            ...(gradientNorm === undefined ? {} : { gradientNorm }),
            ...(lossScale === undefined ? {} : { lossScale }),
            ...(skippedOptimizerStep ? { skippedOptimizerStep: true } : {}),
        };
    }

    private stackBatch(batch: DataBatch<TrainingSample>): { input: Tensor; target: Tensor } {
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
        return { input: Tensor.stack(inputs), target: Tensor.stack(targets) };
    }

    private assertUsableLoader(loader: DataLoader<TrainingSample>, label: string): void {
        if (loader.dataset.length === 0) {
            throw new TrainingDataError(`${label} dataset cannot be empty.`);
        }
        if (loader.batchCount === 0) {
            throw new TrainingDataError(
                `${label} DataLoader does not emit any batches. Disable dropLast or reduce batchSize.`,
            );
        }
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
