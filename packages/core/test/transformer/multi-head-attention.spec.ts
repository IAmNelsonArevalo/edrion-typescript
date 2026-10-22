import { describe, expect, it } from 'vitest';
import { MultiHeadAttention, Tensor } from '@core';

describe('MultiHeadAttention', (): void => {
    it('should preserve sequence and embedding dimensions', (): void => {
        const module: MultiHeadAttention = new MultiHeadAttention(4, 2);
        const input: Tensor = new Tensor([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [3, 4], { requiresGrad: true });
        const output: Tensor = module.forward(input);
        expect(output.shape).toEqual([3, 4]);
        output.mean().backward();
        expect(module.queryProjection.weight.grad).toBeDefined();
        expect(module.keyProjection.weight.grad).toBeDefined();
        expect(module.valueProjection.weight.grad).toBeDefined();
        expect(module.outputProjection.weight.grad).toBeDefined();
    });
});
