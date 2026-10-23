import { describe, expect, it } from 'vitest';

import {
    ArrayDataset,
    DataLoader,
    Linear,
    MSELoss,
    SGD,
    Tensor,
    Trainer,
    type BatchMetrics,
    type TrainingSample,
} from '@core';

class RecordingLinear extends Linear {
    public readonly observedShapes: number[][] = [];

    public override forward(input: Tensor): Tensor {
        this.observedShapes.push([...input.shape]);
        return super.forward(input);
    }
}

describe('FEATURE-012 training data pipeline', (): void => {
    it('should execute one real batched forward pass per DataLoader batch', (): void => {
        const model: RecordingLinear = new RecordingLinear(1, 1);
        const optimizer: SGD = new SGD(model.parameters(), { learningRate: 0.01 });
        const trainer: Trainer = new Trainer(model, optimizer, new MSELoss(), {
            epochs: 1,
            batchSize: 2,
        });
        const samples: readonly TrainingSample[] = [
            { input: new Tensor([1], [1]), target: new Tensor([2], [1]) },
            { input: new Tensor([2], [1]), target: new Tensor([4], [1]) },
            { input: new Tensor([3], [1]), target: new Tensor([6], [1]) },
            { input: new Tensor([4], [1]), target: new Tensor([8], [1]) },
        ];
        const batches: BatchMetrics[] = [];

        const result = trainer.train(samples, {
            onBatchEnd: (metrics: BatchMetrics): void => {
                batches.push(metrics);
            },
        });

        expect(model.observedShapes).toEqual([
            [2, 1],
            [2, 1],
        ]);
        expect(batches).toHaveLength(2);
        expect(result.epochs[0]?.samples).toBe(4);
        expect(trainer.state.globalStep).toBe(2);
    });

    it('should accept a custom DataLoader as a training source', (): void => {
        const model: RecordingLinear = new RecordingLinear(1, 1);
        const optimizer: SGD = new SGD(model.parameters(), { learningRate: 0.01 });
        const trainer: Trainer = new Trainer(model, optimizer, new MSELoss(), { epochs: 1 });
        const dataset: ArrayDataset<TrainingSample> = new ArrayDataset([
            { input: new Tensor([1], [1]), target: new Tensor([2], [1]) },
            { input: new Tensor([2], [1]), target: new Tensor([4], [1]) },
            { input: new Tensor([3], [1]), target: new Tensor([6], [1]) },
        ]);
        const loader: DataLoader<TrainingSample> = new DataLoader(dataset, {
            batchSize: 2,
            dropLast: true,
        });

        const result = trainer.train(loader);

        expect(result.epochs[0]?.samples).toBe(2);
        expect(result.epochs[0]?.batches).toBe(1);
        expect(model.observedShapes).toEqual([[2, 1]]);
    });
});
