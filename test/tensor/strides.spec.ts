import { describe, expect, it } from 'vitest';
import { calculateTensorStrides } from '@core';

describe('Tensor strides', () => {
    it('should calculate vector strides', () => {
        expect(calculateTensorStrides([4])).toEqual([1]);
    });

    it('should calculate matrix strides', () => {
        expect(calculateTensorStrides([2, 3])).toEqual([3, 1]);
    });

    it('should calculate rank 3 strides', () => {
        expect(calculateTensorStrides([2, 3, 4])).toEqual([12, 4, 1]);
    });

    it('should return no strides for scalar tensors', () => {
        expect(calculateTensorStrides([])).toEqual([]);
    });
});
