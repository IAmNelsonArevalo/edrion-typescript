import { describe, expect, it } from 'vitest';
import { Tensor, full, ones, scalar, zeros } from '@core';

describe('Tensor factories', () => {
    it('should create a zero tensor', () => {
        const tensor = Tensor.zeros([2, 3]);
        expect(tensor.toArray()).toEqual([
            [0, 0, 0],
            [0, 0, 0],
        ]);
    });

    it('should create a ones tensor', () => {
        const tensor = Tensor.ones([2, 2]);
        expect(tensor.toArray()).toEqual([
            [1, 1],
            [1, 1],
        ]);
    });

    it('should create a full tensor', () => {
        const tensor = Tensor.full([2, 2], 5);
        expect(tensor.toArray()).toEqual([
            [5, 5],
            [5, 5],
        ]);
    });

    it('should create scalar tensors', () => {
        const tensor = Tensor.scalar(7);

        expect(tensor.shape).toEqual([]);
        expect(tensor.rank).toBe(0);
        expect(tensor.size).toBe(1);
        expect(tensor.isScalar).toBe(true);
        expect(tensor.get()).toBe(7);
        expect(tensor.toArray()).toBe(7);
    });

    it('should expose functional factories', () => {
        expect(zeros([2]).toArray()).toEqual([0, 0]);
        expect(ones([2]).toArray()).toEqual([1, 1]);
        expect(full([2], 8).toArray()).toEqual([8, 8]);
        expect(scalar(10).toArray()).toBe(10);
    });
});
