import { describe, expect, it } from 'vitest';
import { scalar, type Tensor } from '@core';

describe('Scalar Tensor', (): void => {
    it('should create scalar tensor', (): void => {
        const tensor: Tensor = scalar(42);
        expect(tensor.shape).toEqual([]);
        expect(tensor.strides).toEqual([]);
        expect(tensor.rank).toBe(0);
        expect(tensor.size).toBe(1);
        expect(tensor.isScalar).toBe(true);
        expect(tensor.get()).toBe(42);
    });

    it('should update scalar tensor', (): void => {
        const tensor: Tensor = scalar(10);
        tensor.set([], 25);
        expect(tensor.get()).toBe(25);
    });

    it('should serialize scalar as a number', (): void => {
        const tensor: Tensor = scalar(5);
        expect(tensor.toArray()).toBe(5);
    });
});
