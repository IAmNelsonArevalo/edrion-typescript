import { describe, expect, it } from 'vitest';
import { Tensor } from '@core/tensor';

describe('FEATURE-011 batched matrix operations', (): void => {
    it('multiplies rank-3 tensors batch by batch', (): void => {
        const left: Tensor = new Tensor(
            new Float32Array([
                1, 2, 3, 4,
                5, 6, 7, 8,
            ]),
            [2, 2, 2],
        );
        const right: Tensor = new Tensor(
            new Float32Array([
                1, 0, 0, 1,
                2, 0, 0, 2,
            ]),
            [2, 2, 2],
        );

        const output: Tensor = left.matmul(right);

        expect(output.shape).toEqual([2, 2, 2]);
        expect(output.toFlatArray()).toEqual([1, 2, 3, 4, 10, 12, 14, 16]);
    });

    it('propagates gradients through batched matmul', (): void => {
        const left: Tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [1, 2, 2], { requiresGrad: true });
        const right: Tensor = new Tensor(new Float32Array([1, 0, 0, 1]), [1, 2, 2], { requiresGrad: true });

        left.matmul(right).sum().backward();

        expect(left.grad?.toFlatArray()).toEqual([1, 1, 1, 1]);
        expect(right.grad?.toFlatArray()).toEqual([4, 4, 6, 6]);
    });

    it('transposes the final two dimensions of a batch', (): void => {
        const tensor: Tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [1, 2, 3]);
        const transposed: Tensor = tensor.transpose();

        expect(transposed.shape).toEqual([1, 3, 2]);
        expect(transposed.toFlatArray()).toEqual([1, 4, 2, 5, 3, 6]);
    });
});
