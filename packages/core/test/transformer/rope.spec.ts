import { describe, expect, it } from 'vitest';
import { ones, Tensor } from '@core';

describe('Rotary Position Embedding', (): void => {
    it('should preserve position zero and rotate later positions with autograd', (): void => {
        const input: Tensor = new Tensor([1, 2, 3, 4, 1, 2, 3, 4], [2, 4], { requiresGrad: true });
        const output: Tensor = input.applyRotaryPositionEmbedding(4);
        expect(output.shape).toEqual([2, 4]);
        expect(output.get(0, 0)).toBeCloseTo(1, 6);
        expect(output.get(0, 1)).toBeCloseTo(2, 6);
        expect(output.get(1, 0)).not.toBeCloseTo(1, 3);
        output.sum().backward();
        expect(input.grad).toBeDefined();
        expect(input.grad?.shape).toEqual([2, 4]);
    });

    it('should support batched tensors', (): void => {
        const input: Tensor = ones([2, 3, 4], { requiresGrad: true });
        const output: Tensor = input.applyRotaryPositionEmbedding(4);
        expect(output.shape).toEqual([2, 3, 4]);
        output.mean().backward();
        expect(input.grad).toBeDefined();
    });
});
