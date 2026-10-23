import type { TrainingDataSource, TrainerCallbacks, TrainingResult } from '@core/training';
import { Trainer } from '@core/training';
import { CrossEntropyLoss } from '@core/losses';
import { AdamW } from '@core/optim';
import { WarmupCosineScheduler } from '@core/optim/schedulers';
import type { CausalLanguageModel } from '@core/language-model/causal-language-model';

export interface LanguageModelPretrainingConfig {
    readonly epochs: number;
    readonly batchSize: number;
    readonly learningRate?: number;
    readonly weightDecay?: number;
    readonly warmupSteps?: number;
    readonly totalSteps: number;
    readonly gradientAccumulationSteps?: number;
    readonly maxGradientNorm?: number;
    readonly seed?: number;
}

/** Opinionated GPT-style pretraining orchestration using Edrion AdamW and warmup/cosine scheduling. */
export class LanguageModelPretrainer {
    private readonly trainer: Trainer;

    public constructor(
        model: CausalLanguageModel,
        config: LanguageModelPretrainingConfig,
        validationData?: TrainingDataSource,
    ) {
        const optimizer: AdamW = new AdamW(model.parameters(), {
            learningRate: config.learningRate ?? 3e-4,
            weightDecay: config.weightDecay ?? 0.1,
        });
        const warmupSteps: number = config.warmupSteps ?? Math.max(1, Math.floor(config.totalSteps * 0.03));
        const scheduler: WarmupCosineScheduler = new WarmupCosineScheduler(optimizer, {
            warmupSteps,
            totalSteps: Math.max(warmupSteps + 1, config.totalSteps),
        });
        this.trainer = new Trainer(
            model,
            optimizer,
            new CrossEntropyLoss(),
            {
                epochs: config.epochs,
                batchSize: config.batchSize,
                shuffle: true,
                seed: config.seed ?? 1337,
                gradientAccumulationSteps: config.gradientAccumulationSteps ?? 1,
                ...(config.maxGradientNorm === undefined ? {} : { maxGradientNorm: config.maxGradientNorm }),
                evaluateEveryEpoch: validationData !== undefined,
            },
            {
                scheduler,
                ...(validationData === undefined ? {} : { validationData }),
            },
        );
    }

    public train(source: TrainingDataSource, callbacks: TrainerCallbacks = {}): TrainingResult {
        return this.trainer.train(source, callbacks);
    }
}
