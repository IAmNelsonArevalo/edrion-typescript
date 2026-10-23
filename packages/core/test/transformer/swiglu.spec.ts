import { describe, expect, it } from 'vitest';
import { ones, SwiGLUFeedForward, Tensor } from '@core';

describe('SwiGLUFeedForward', (): void => {
    it('should preserve leading dimensions and propagate gradients', (): void => {
        const module: SwiGLUFeedForward = new SwiGLUFeedForward(4, 8);
        const input: Tensor = ones([2, 3, 4], { requiresGrad: true });
        const output: Tensor = module.forward(input);
        expect(output.shape).toEqual([2, 3, 4]);
        output.mean().backward();
        expect(input.grad).toBeDefined();
        expect(module.gateProjection.weight.grad).toBeDefined();
        expect(module.upProjection.weight.grad).toBeDefined();
        expect(module.downProjection.weight.grad).toBeDefined();
    });
});
