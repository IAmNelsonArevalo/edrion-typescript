import { describe, expect, it } from 'vitest';

import { MSELoss, Tensor } from '@core';

describe('MSELoss', (): void => {
    it('should compute mean squared error', (): void => {
        const lossFunction: MSELoss = new MSELoss();
        const prediction: Tensor = new Tensor([1, 3], [2]);
        const target: Tensor = new Tensor([0, 1], [2]);

        const loss: Tensor = lossFunction.forward(prediction, target);

        expect(loss.get()).toBeCloseTo(2.5, 6);
    });

    it('should propagate gradients', (): void => {
        const lossFunction: MSELoss = new MSELoss();
        const prediction: Tensor = new Tensor([1, 3], [2], {
            requiresGrad: true,
        });
        const target: Tensor = new Tensor([0, 1], [2]);

        const loss: Tensor = lossFunction.forward(prediction, target);

        loss.backward();

        expect(prediction.grad?.toFlatArray()).toEqual([1, 2]);
    });
});
