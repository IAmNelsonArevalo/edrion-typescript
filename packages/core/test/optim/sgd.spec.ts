import { describe, expect, it } from 'vitest';

import { Parameter, SGD, Tensor } from '@core';

describe('SGD', (): void => {
    it('should update parameters from gradients', (): void => {
        const parameter: Parameter = new Parameter([1, 2], [2]);
        const optimizer: SGD = new SGD([parameter], {
            learningRate: 0.1,
        });

        const loss: Tensor = parameter.pow(2).sum();

        loss.backward();
        optimizer.step();

        const values: number[] = parameter.toFlatArray();

        expect(values[0]).toBeCloseTo(0.8, 6);
        expect(values[1]).toBeCloseTo(1.6, 6);
    });

    it('should clear gradients', (): void => {
        const parameter: Parameter = new Parameter([1], [1]);
        const optimizer: SGD = new SGD([parameter], {
            learningRate: 0.1,
        });

        parameter.pow(2).sum().backward();

        expect(parameter.grad).toBeDefined();

        optimizer.zeroGrad();

        expect(parameter.grad).toBeUndefined();
    });

    it('should support momentum', (): void => {
        const parameter: Parameter = new Parameter([1], [1]);
        const optimizer: SGD = new SGD([parameter], {
            learningRate: 0.1,
            momentum: 0.9,
        });

        parameter.pow(2).sum().backward();
        optimizer.step();
        optimizer.zeroGrad();

        parameter.pow(2).sum().backward();
        optimizer.step();

        expect(parameter.get(0)).toBeLessThan(0.8);
    });
});
