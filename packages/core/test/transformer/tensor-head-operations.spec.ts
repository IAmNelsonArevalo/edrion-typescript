import { describe, expect, it } from 'vitest';
import { Tensor } from '@core';

describe('Transformer tensor head operations', (): void => {
    it('should slice and concatenate columns with autograd', (): void => {
        const input: Tensor = new Tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 4], { requiresGrad: true });
        const left: Tensor = input.sliceColumns(0, 2);
        const right: Tensor = input.sliceColumns(2, 4);
        const combined: Tensor = Tensor.concatColumns([left, right]);
        expect(combined.toFlatArray()).toEqual(input.toFlatArray());
        combined.sum().backward();
        expect(input.grad?.toFlatArray()).toEqual([1, 1, 1, 1, 1, 1, 1, 1]);
    });
});
