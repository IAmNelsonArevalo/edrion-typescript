import { describe, expect, it } from 'vitest';

import { Linear, MSELoss, SGD, Tensor, Trainer, type TrainingSample } from '@core';

describe('Trainer', (): void => {
    it('should train a simple supervised model and expose metrics', (): void => {
        const model: Linear = new Linear(1, 1);
        const optimizer: SGD = new SGD(model.parameters(), {
            learningRate: 0.01,
        });
        const loss: MSELoss = new MSELoss();
        const trainer: Trainer = new Trainer(model, optimizer, loss, {
            epochs: 2,
            batchSize: 2,
        });
        const dataset: readonly TrainingSample[] = [
            {
                input: new Tensor([1], [1]),
                target: new Tensor([2], [1]),
            },
            {
                input: new Tensor([2], [1]),
                target: new Tensor([4], [1]),
            },
        ];

        const result = trainer.train(dataset);

        expect(result.epochs).toHaveLength(2);
        expect(result.finalLoss).toBeGreaterThanOrEqual(0);
        expect(trainer.state.epoch).toBe(2);
        expect(trainer.state.globalStep).toBe(2);
        expect(trainer.state.completed).toBe(true);
    });

    it('should restore trainer progress', (): void => {
        const model: Linear = new Linear(1, 1);
        const optimizer: SGD = new SGD(model.parameters(), {
            learningRate: 0.01,
        });
        const trainer: Trainer = new Trainer(model, optimizer, new MSELoss(), { epochs: 1 });

        trainer.restoreState({
            epoch: 10,
            globalStep: 42,
            completed: false,
        });

        expect(trainer.state.epoch).toBe(10);
        expect(trainer.state.globalStep).toBe(42);
    });
});
