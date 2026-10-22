import { describe, expect, it } from 'vitest';
import { calculateTensorSize, calculateTensorStrides, TensorShapeError } from '@core';

describe('Tensor shape', (): void => {
    it('should calculate vector size', (): void => {
        expect(calculateTensorSize([4])).toBe(4);
    });

    it('should calculate matrix size', (): void => {
        expect(calculateTensorSize([2, 3])).toBe(6);
    });

    it('should calculate scalar size', (): void => {
        expect(calculateTensorSize([])).toBe(1);
    });

    it('should calculate row-major strides', (): void => {
        expect(calculateTensorStrides([2, 3, 4])).toEqual([12, 4, 1]);
    });

    it('should create empty strides for scalar', (): void => {
        expect(calculateTensorStrides([])).toEqual([]);
    });

    it('should reject negative dimensions', (): void => {
        expect((): void => {
            calculateTensorSize([2, -3]);
        }).toThrow(TensorShapeError);
    });

    it('should reject zero dimensions', (): void => {
        expect((): void => {
            calculateTensorSize([2, 0]);
        }).toThrow(TensorShapeError);
    });

    it('should reject non-integer dimensions', (): void => {
        expect((): void => {
            calculateTensorSize([2, 1.5]);
        }).toThrow(TensorShapeError);
    });
});
