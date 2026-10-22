import { describe, expect, it } from 'vitest';

import { GELU, ReLU, Sigmoid, Softmax, Tanh, Tensor } from '@core';

describe('Neural-network activations', (): void => {
    it('should apply ReLU with gradients', (): void => {
        const input: Tensor = new Tensor([-2, 0, 3], [3], {
            requiresGrad: true,
        });

        const output: Tensor = new ReLU().forward(input);

        expect(output.toFlatArray()).toEqual([0, 0, 3]);

        output.sum().backward();

        expect(input.grad?.toFlatArray()).toEqual([0, 0, 1]);
    });

    it('should apply sigmoid', (): void => {
        const output: Tensor = new Sigmoid().forward(new Tensor([0], [1]));

        expect(output.get(0)).toBeCloseTo(0.5, 5);
    });

    it('should apply tanh', (): void => {
        const output: Tensor = new Tanh().forward(new Tensor([0], [1]));

        expect(output.get(0)).toBeCloseTo(0, 5);
    });

    it('should apply GELU', (): void => {
        const output: Tensor = new GELU().forward(new Tensor([0], [1]));

        expect(output.get(0)).toBeCloseTo(0, 5);
    });

    it('should normalize softmax probabilities', (): void => {
        const output: Tensor = new Softmax(-1).forward(new Tensor([1, 2, 3], [3]));

        const probabilitySum: number = output
            .toFlatArray()
            .reduce((sum: number, value: number): number => sum + value, 0);

        expect(probabilitySum).toBeCloseTo(1, 5);
    });
});
