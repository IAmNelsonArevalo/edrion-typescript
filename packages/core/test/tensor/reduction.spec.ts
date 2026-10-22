import { describe, expect, it } from 'vitest';

import { Tensor } from '@core';

describe('Tensor reductions', (): void => {
    it('should sum all values', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4], [2, 2]);

        expect(tensor.sum().get()).toBe(10);
    });

    it('should calculate mean', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4], [2, 2]);

        expect(tensor.mean().get()).toBe(2.5);
    });

    it('should reduce along an axis', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);

        expect(tensor.sum(0).toFlatArray()).toEqual([5, 7, 9]);

        expect(tensor.sum(1).toFlatArray()).toEqual([6, 15]);
    });

    it('should support negative axes and keepDims', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);

        const result: Tensor = tensor.mean(-1, true);

        expect(result.shape).toEqual([2, 1]);

        expect(result.toFlatArray()).toEqual([2, 5]);
    });
});
