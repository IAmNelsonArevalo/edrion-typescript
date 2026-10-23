import { describe, expect, it } from 'vitest';

import { Tensor, TensorShapeError } from '@core';

describe('Tensor.stack', (): void => {
    it('should stack tensors along a new leading batch dimension', (): void => {
        const first: Tensor = new Tensor([1, 2], [2]);
        const second: Tensor = new Tensor([3, 4], [2]);

        const stacked: Tensor = Tensor.stack([first, second]);

        expect(stacked.shape).toEqual([2, 2]);
        expect(Array.from(stacked.toFloat32Array())).toEqual([1, 2, 3, 4]);
    });

    it('should propagate gradients to each stacked tensor', (): void => {
        const first: Tensor = new Tensor([1, 2], [2], { requiresGrad: true });
        const second: Tensor = new Tensor([3, 4], [2], { requiresGrad: true });

        Tensor.stack([first, second]).sum().backward();

        expect(Array.from(first.grad?.toFloat32Array() ?? [])).toEqual([1, 1]);
        expect(Array.from(second.grad?.toFloat32Array() ?? [])).toEqual([1, 1]);
    });

    it('should reject mismatched shapes', (): void => {
        const first: Tensor = new Tensor([1, 2], [2]);
        const second: Tensor = new Tensor([1, 2, 3], [3]);

        expect((): Tensor => Tensor.stack([first, second])).toThrow(TensorShapeError);
    });
});
