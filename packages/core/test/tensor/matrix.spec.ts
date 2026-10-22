import { describe, expect, it } from 'vitest';

import { Tensor, TensorShapeError } from '@core';

describe('Tensor matrix operations', (): void => {
    it('should transpose a matrix', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);

        const result: Tensor = tensor.transpose();

        expect(result.shape).toEqual([3, 2]);

        expect(result.toArray()).toEqual([
            [1, 4],
            [2, 5],
            [3, 6],
        ]);
    });

    it('should multiply matrices', (): void => {
        const left: Tensor = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
        const right: Tensor = new Tensor([7, 8, 9, 10, 11, 12], [3, 2]);

        const result: Tensor = left.matmul(right);

        expect(result.shape).toEqual([2, 2]);

        expect(result.toFlatArray()).toEqual([58, 64, 139, 154]);
    });

    it('should reject incompatible matrix shapes', (): void => {
        const left: Tensor = new Tensor([1, 2, 3, 4], [2, 2]);
        const right: Tensor = new Tensor([1, 2, 3, 4, 5, 6], [3, 2]);

        expect((): void => {
            left.matmul(right);
        }).toThrow(TensorShapeError);
    });
});
