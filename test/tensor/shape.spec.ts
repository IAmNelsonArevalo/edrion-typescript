import { describe, expect, it } from 'vitest';
import { calculateTensorSize, tensorShapesEqual, validateTensorShape } from '@core';

describe('Tensor shape', () => {
    it('should calculate matrix size', () => {
        expect(calculateTensorSize([2, 3])).toBe(6);
    });

    it('should calculate rank 3 tensor size', () => {
        expect(calculateTensorSize([2, 3, 4])).toBe(24);
    });

    it('should treat empty shape as scalar', () => {
        expect(calculateTensorSize([])).toBe(1);
    });

    it('should support zero-sized dimensions', () => {
        expect(calculateTensorSize([2, 0, 3])).toBe(0);
    });

    it('should reject negative dimensions', () => {
        expect(() => validateTensorShape([2, -1])).toThrow('Tensor dimension 1 cannot be negative.');
    });

    it('should reject decimal dimensions', () => {
        expect(() => validateTensorShape([2, 1.5])).toThrow('Tensor dimension 1 must be an integer.');
    });

    it('should compare shapes', () => {
        expect(tensorShapesEqual([2, 3], [2, 3])).toBe(true);
        expect(tensorShapesEqual([2, 3], [3, 2])).toBe(false);
    });
});
