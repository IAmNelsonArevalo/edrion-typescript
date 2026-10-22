import { describe, expect, it } from 'vitest';
import { full, ones, scalar, type Tensor, zeros } from '@core';

describe('Tensor factories', (): void => {
    it('should create zeros', (): void => {
        const tensor: Tensor = zeros([2, 2]);
        expect(tensor.toFlatArray()).toEqual([0, 0, 0, 0]);
    });

    it('should create ones', (): void => {
        const tensor: Tensor = ones([2, 2]);
        expect(tensor.toFlatArray()).toEqual([1, 1, 1, 1]);
    });

    it('should create a filled tensor', (): void => {
        const tensor: Tensor = full([3], 7);
        expect(tensor.toFlatArray()).toEqual([7, 7, 7]);
    });

    it('should create a scalar', (): void => {
        const tensor: Tensor = scalar(3.5);
        expect(tensor.get()).toBeCloseTo(3.5);
    });
});
