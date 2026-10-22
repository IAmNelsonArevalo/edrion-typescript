import { describe, expect, it } from 'vitest';

import { Tensor } from '@core';

describe('Tensor operations', (): void => {
    it('should add tensors', (): void => {
        const left: Tensor = new Tensor([1, 2, 3], [3]);
        const right: Tensor = new Tensor([4, 5, 6], [3]);

        expect(left.add(right).toFlatArray()).toEqual([5, 7, 9]);
    });

    it('should support scalar operands', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3], [3]);

        expect(tensor.mul(2).add(1).toFlatArray()).toEqual([3, 5, 7]);
    });

    it('should support broadcasting', (): void => {
        const matrix: Tensor = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
        const bias: Tensor = new Tensor([10, 20, 30], [3]);

        expect(matrix.add(bias).toArray()).toEqual([
            [11, 22, 33],
            [14, 25, 36],
        ]);
    });

    it('should apply unary operations', (): void => {
        const tensor: Tensor = new Tensor([1, 4, 9], [3]);

        expect(tensor.sqrt().toFlatArray()).toEqual([1, 2, 3]);

        expect(tensor.neg().toFlatArray()).toEqual([-1, -4, -9]);
    });

    it('should calculate power', (): void => {
        const tensor: Tensor = new Tensor([2, 3, 4], [3]);

        expect(tensor.pow(2).toFlatArray()).toEqual([4, 9, 16]);
    });
});
