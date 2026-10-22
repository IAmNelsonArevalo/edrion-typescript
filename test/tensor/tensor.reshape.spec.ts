import { describe, expect, it } from 'vitest';
import { Tensor } from '@core';

describe('Tensor reshape', () => {
    it('should reshape a matrix into a vector', () => {
        const tensor = new Tensor([1, 2, 3, 4], [2, 2]),
            reshaped = tensor.reshape([4]);
        expect(reshaped.shape).toEqual([4]);
        expect(reshaped.toArray()).toEqual([1, 2, 3, 4]);
    });

    it('should reshape a vector into a matrix', () => {
        const tensor = new Tensor([1, 2, 3, 4, 5, 6], [6]),
            reshaped = tensor.reshape([2, 3]);
        expect(reshaped.toArray()).toEqual([
            [1, 2, 3],
            [4, 5, 6],
        ]);
    });

    it('should reject incompatible reshape sizes', () => {
        const tensor = Tensor.zeros([2, 2]);
        expect(() => tensor.reshape([3, 3])).toThrow('Cannot reshape tensor of size 4 into shape [3, 3] with size 9.');
    });

    it('should preserve dtype and backend', () => {
        const tensor = new Tensor([1, 2, 3, 4], [4], {
            backend: 'custom',
        });

        const reshaped = tensor.reshape([2, 2]);
        expect(reshaped.dtype).toBe('float32');
        expect(reshaped.backend).toBe('custom');
    });
});
