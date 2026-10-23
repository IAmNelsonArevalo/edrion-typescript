import { describe, expect, it } from 'vitest';
import { MSELoss } from '@core/losses';
import { Linear } from '@core/nn';
import { SGD, WarmupCosineScheduler } from '@core/optim';
import { Tensor } from '@core/tensor';
import { Trainer, type TrainingSample } from '@core/training';

describe('advanced trainer', () => {
    it('supports accumulation and evaluation', () => {
        const model: Linear = new Linear(1, 1);
        const optimizer: SGD = new SGD(model.parameters(), { learningRate: 0.05 });
        const scheduler: WarmupCosineScheduler = new WarmupCosineScheduler(optimizer, {
            warmupSteps: 1,
            totalSteps: 4,
            minimumLearningRate: 0.005,
        });
        const samples: readonly TrainingSample[] = [
            { input: new Tensor([1], [1]), target: new Tensor([3], [1]) },
            { input: new Tensor([2], [1]), target: new Tensor([5], [1]) },
            { input: new Tensor([3], [1]), target: new Tensor([7], [1]) },
            { input: new Tensor([4], [1]), target: new Tensor([9], [1]) },
        ];
        const trainer: Trainer = new Trainer(
            model,
            optimizer,
            new MSELoss(),
            {
                epochs: 1,
                batchSize: 1,
                gradientAccumulationSteps: 2,
                maxGradientNorm: 10,
            },
            { scheduler, validationData: samples },
        );
        const result = trainer.train(samples);
        expect(result.epochs[0]?.optimizerSteps).toBe(2);
        expect(result.finalValidationLoss).toBeTypeOf('number');
        expect(trainer.state.globalStep).toBe(2);
    });
});
