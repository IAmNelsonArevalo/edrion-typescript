import type { TrainingDataSource } from '@core/training';
import { Trainer } from '@core/training';
import { CrossEntropyLoss } from '@core/losses';
import { SGD } from '@core/optim';
import type { CausalLanguageModel } from '@core/language-model/causal-language-model';

export interface LanguageModelEvaluation {
    readonly averageLoss: number;
    readonly perplexity: number;
    readonly samples: number;
    readonly batches: number;
}

/** Evaluates next-token loss and perplexity without updating model parameters. */
export function evaluateLanguageModel(
    model: CausalLanguageModel,
    source: TrainingDataSource,
    batchSize: number = 1,
): LanguageModelEvaluation {
    const optimizer: SGD = new SGD(model.parameters(), { learningRate: 1e-12 });
    const trainer: Trainer = new Trainer(model, optimizer, new CrossEntropyLoss(), {
        epochs: 1,
        batchSize,
        shuffle: false,
    });
    const metrics = trainer.evaluate(source);
    const perplexity: number = Math.exp(Math.min(metrics.averageLoss, 80));
    return {
        averageLoss: metrics.averageLoss,
        perplexity,
        samples: metrics.samples,
        batches: metrics.batches,
    };
}
