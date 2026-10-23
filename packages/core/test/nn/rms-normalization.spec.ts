import { describe, expect, it } from 'vitest';
import { RMSNormalization, Tensor } from '@core';

describe('RMSNormalization', (): void => {
    it('should normalize the RMS of the last dimension', (): void => {
        const layer: RMSNormalization = new RMSNormalization(4, { epsilon: 1e-6 });
        const input: Tensor = new Tensor([1, 2, 3, 4], [1, 4], { requiresGrad: true });
        const output: Tensor = layer.forward(input);
        const values: Float32Array = output.toFloat32Array();
        let squareSum: number = 0;
        for (let index: number = 0; index < values.length; index += 1) {
            const value: number | undefined = values[index];
            if (value === undefined) throw new Error('Missing normalized value.');
            squareSum += value * value;
        }
        expect(Math.sqrt(squareSum / values.length)).toBeCloseTo(1, 5);
        output.sum().backward();
        expect(input.grad).toBeDefined();
        expect(layer.weight.grad).toBeDefined();
    });
});
