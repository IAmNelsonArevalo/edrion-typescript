import { describe, expect, it } from 'vitest';

import { CrossEntropyLoss, LossTargetError, Tensor } from '@core';

describe('CrossEntropyLoss', (): void => {
    it('should compute multiclass cross entropy', (): void => {
        const lossFunction: CrossEntropyLoss = new CrossEntropyLoss();
        const logits: Tensor = new Tensor([0, 0, 0], [1, 3]);
        const target: Tensor = new Tensor([0], [1]);

        const loss: Tensor = lossFunction.forward(logits, target);

        expect(loss.get()).toBeCloseTo(Math.log(3), 5);
    });

    it('should propagate softmax cross entropy gradients', (): void => {
        const lossFunction: CrossEntropyLoss = new CrossEntropyLoss();
        const logits: Tensor = new Tensor([0, 0, 0], [1, 3], {
            requiresGrad: true,
        });
        const target: Tensor = new Tensor([0], [1]);

        const loss: Tensor = lossFunction.forward(logits, target);

        loss.backward();

        const gradient: number[] | undefined = logits.grad?.toFlatArray();

        expect(gradient?.[0]).toBeCloseTo(-2 / 3, 5);
        expect(gradient?.[1]).toBeCloseTo(1 / 3, 5);
        expect(gradient?.[2]).toBeCloseTo(1 / 3, 5);
    });

    it('should reject invalid class targets', (): void => {
        const lossFunction: CrossEntropyLoss = new CrossEntropyLoss();
        const logits: Tensor = new Tensor([1, 2], [1, 2]);
        const target: Tensor = new Tensor([2], [1]);

        expect((): void => {
            lossFunction.forward(logits, target);
        }).toThrow(LossTargetError);
    });
});
