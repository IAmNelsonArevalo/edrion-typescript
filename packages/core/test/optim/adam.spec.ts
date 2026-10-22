import { describe, expect, it } from 'vitest';

import { Adam, Parameter, Tensor } from '@core';

describe('Adam', (): void => {
    it('should apply a bias-corrected Adam update', (): void => {
        const parameter: Parameter = new Parameter([1], [1]);
        const optimizer: Adam = new Adam([parameter], {
            learningRate: 0.1,
        });

        const loss: Tensor = parameter.pow(2).sum();

        loss.backward();
        optimizer.step();

        expect(parameter.get(0)).toBeCloseTo(0.9, 5);
        expect(optimizer.steps).toBe(1);
    });

    it('should preserve optimizer state across steps', (): void => {
        const parameter: Parameter = new Parameter([1], [1]);
        const optimizer: Adam = new Adam([parameter], {
            learningRate: 0.1,
        });

        for (let step: number = 0; step < 2; step += 1) {
            optimizer.zeroGrad();

            parameter.pow(2).sum().backward();

            optimizer.step();
        }

        expect(optimizer.steps).toBe(2);
        expect(parameter.get(0)).toBeLessThan(0.9);
    });
});
