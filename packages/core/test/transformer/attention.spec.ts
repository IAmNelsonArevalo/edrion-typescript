import { describe, expect, it } from 'vitest';
import { ScaledDotProductAttention, Tensor } from '@core';

describe('ScaledDotProductAttention', (): void => {
    it('should return [queryLength, valueDimension]', (): void => {
        const attention: ScaledDotProductAttention = new ScaledDotProductAttention();
        const query: Tensor = new Tensor([1, 0, 0, 1], [2, 2], { requiresGrad: true });
        const key: Tensor = new Tensor([1, 0, 0, 1], [2, 2], { requiresGrad: true });
        const value: Tensor = new Tensor([1, 2, 3, 4], [2, 2], { requiresGrad: true });
        const output: Tensor = attention.forward(query, key, value);
        expect(output.shape).toEqual([2, 2]);
        output.mean().backward();
        expect(query.grad).toBeDefined();
        expect(key.grad).toBeDefined();
        expect(value.grad).toBeDefined();
    });
});
