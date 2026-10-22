import { describe, expect, it } from 'vitest';

import { CrossEntropyLoss, Softmax, Tensor } from '@core';

describe('Numerical stability', (): void => {
    it('keeps softmax finite for large logits', (): void => {
        const logits: Tensor = new Tensor([1000, 1001, 1002], [1, 3]);
        const probabilities: Tensor = new Softmax(-1).forward(logits);
        const values: number[] = probabilities.toFlatArray();
        const total: number = values.reduce((sum: number, value: number): number => sum + value, 0);

        expect(values.every((value: number): boolean => Number.isFinite(value))).toBe(true);
        expect(total).toBeCloseTo(1, 5);
    });

    it('keeps cross entropy finite for large logits', (): void => {
        const logits: Tensor = new Tensor([1000, 1001, 1002], [1, 3], { requiresGrad: true });
        const target: Tensor = new Tensor([2], [1]);
        const loss: Tensor = new CrossEntropyLoss().forward(logits, target);

        expect(Number.isFinite(loss.get())).toBe(true);

        loss.backward();
        expect(logits.grad).toBeDefined();
    });
});
