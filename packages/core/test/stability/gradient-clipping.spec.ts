import { describe, expect, it } from 'vitest';

import { Parameter, Tensor, clipGradientsByGlobalNorm, computeGlobalGradientNorm } from '@core';

describe('Gradient clipping', (): void => {
    it('clips gradients using a shared global norm', (): void => {
        const parameter: Parameter = new Parameter([3, 4], [2]);
        const loss: Tensor = parameter.pow(2).sum();
        loss.backward();

        const before: number = computeGlobalGradientNorm([parameter]);
        const result = clipGradientsByGlobalNorm([parameter], 5);
        const after: number = computeGlobalGradientNorm([parameter]);

        expect(before).toBeCloseTo(10);
        expect(result.clipped).toBe(true);
        expect(after).toBeCloseTo(5);
    });

    it('leaves gradients unchanged when already below the limit', (): void => {
        const parameter: Parameter = new Parameter([1], [1]);
        parameter.pow(2).sum().backward();

        const result = clipGradientsByGlobalNorm([parameter], 10);

        expect(result.clipped).toBe(false);
        expect(result.clipCoefficient).toBe(1);
    });
});
