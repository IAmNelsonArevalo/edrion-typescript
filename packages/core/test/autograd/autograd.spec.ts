import { describe, expect, it } from 'vitest';

import { InvalidGradientError, Tensor } from '@core';

describe('Autograd', (): void => {
    it('should calculate a simple gradient', (): void => {
        const x: Tensor = new Tensor([2], [], {
            requiresGrad: true,
        });

        const y: Tensor = x.pow(2).add(x.mul(3)).add(1);

        y.backward();

        expect(x.grad?.get()).toBeCloseTo(7);
    });

    it('should calculate gradients through mean', (): void => {
        const x: Tensor = new Tensor([1, 2, 3, 4], [4], {
            requiresGrad: true,
        });

        const loss: Tensor = x.mul(2).mean();

        loss.backward();

        expect(x.grad?.toFlatArray()).toEqual([0.5, 0.5, 0.5, 0.5]);
    });

    it('should reduce broadcast gradients', (): void => {
        const x: Tensor = new Tensor([1, 2, 3, 4, 5, 6], [2, 3], {
            requiresGrad: true,
        });

        const bias: Tensor = new Tensor([10, 20, 30], [3], {
            requiresGrad: true,
        });

        const loss: Tensor = x.add(bias).sum();

        loss.backward();

        expect(x.grad?.toFlatArray()).toEqual([1, 1, 1, 1, 1, 1]);

        expect(bias.grad?.toFlatArray()).toEqual([2, 2, 2]);
    });

    it('should accumulate gradients from multiple graph paths', (): void => {
        const x: Tensor = new Tensor([3], [], {
            requiresGrad: true,
        });

        const y: Tensor = x.mul(x).add(x);

        y.backward();

        expect(x.grad?.get()).toBeCloseTo(7);
    });

    it('should clear gradients', (): void => {
        const x: Tensor = new Tensor([2], [], {
            requiresGrad: true,
        });

        x.pow(2).backward();

        expect(x.grad?.get()).toBeCloseTo(4);

        x.zeroGrad();

        expect(x.grad).toBeUndefined();
    });

    it('should detach a tensor from autograd', (): void => {
        const x: Tensor = new Tensor([1, 2], [2], {
            requiresGrad: true,
        });

        const detached: Tensor = x.detach();

        expect(detached.requiresGrad).toBe(false);

        expect(detached.toFlatArray()).toEqual([1, 2]);
    });

    it('should require an explicit gradient for non scalar backward', (): void => {
        const x: Tensor = new Tensor([1, 2], [2], {
            requiresGrad: true,
        });

        expect((): void => {
            x.backward();
        }).toThrow(InvalidGradientError);
    });

    it('should accept explicit backward gradient', (): void => {
        const x: Tensor = new Tensor([1, 2], [2], {
            requiresGrad: true,
        });

        const y: Tensor = x.mul(3);

        y.backward(new Tensor([2, 4], [2]));

        expect(x.grad?.toFlatArray()).toEqual([6, 12]);
    });

    it('should differentiate matrix multiplication', (): void => {
        const left: Tensor = new Tensor([1, 2, 3, 4], [2, 2], {
            requiresGrad: true,
        });
        const right: Tensor = new Tensor([5, 6, 7, 8], [2, 2], {
            requiresGrad: true,
        });

        const loss: Tensor = left.matmul(right).sum();

        loss.backward();

        expect(left.grad?.toFlatArray()).toEqual([11, 15, 11, 15]);

        expect(right.grad?.toFlatArray()).toEqual([4, 4, 6, 6]);
    });

    it('should preserve gradients through reshape and clone', (): void => {
        const x: Tensor = new Tensor([1, 2, 3, 4], [4], {
            requiresGrad: true,
        });

        const loss: Tensor = x.reshape([2, 2]).clone().sum();

        loss.backward();

        expect(x.grad?.toFlatArray()).toEqual([1, 1, 1, 1]);
    });
});
