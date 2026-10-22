import { describe, expect, it } from 'vitest';

import { LayerShapeError, Linear, Tensor } from '@core';

describe('Linear', (): void => {
    it('should transform a vector', (): void => {
        const layer: Linear = new Linear(2, 3);

        layer.weight.set([0, 0], 1);
        layer.weight.set([0, 1], 2);
        layer.weight.set([0, 2], 3);
        layer.weight.set([1, 0], 4);
        layer.weight.set([1, 1], 5);
        layer.weight.set([1, 2], 6);

        if (layer.bias === undefined) {
            throw new Error('Expected Linear bias.');
        }

        layer.bias.set([0], 1);
        layer.bias.set([1], 1);
        layer.bias.set([2], 1);

        const input: Tensor = new Tensor([1, 2], [2]);

        const output: Tensor = layer.forward(input);

        expect(output.shape).toEqual([3]);
        expect(output.toFlatArray()).toEqual([10, 13, 16]);
    });

    it('should transform a batch', (): void => {
        const layer: Linear = new Linear(2, 2, {
            bias: false,
        });

        layer.weight.set([0, 0], 1);
        layer.weight.set([0, 1], 0);
        layer.weight.set([1, 0], 0);
        layer.weight.set([1, 1], 1);

        const input: Tensor = new Tensor([1, 2, 3, 4], [2, 2]);

        expect(layer.forward(input).toArray()).toEqual([
            [1, 2],
            [3, 4],
        ]);
    });

    it('should propagate gradients to input and parameters', (): void => {
        const layer: Linear = new Linear(2, 2);
        const input: Tensor = new Tensor([1, 2], [1, 2], {
            requiresGrad: true,
        });

        layer.forward(input).sum().backward();

        expect(input.grad).toBeDefined();
        expect(layer.weight.grad).toBeDefined();
        expect(layer.bias?.grad).toBeDefined();
    });

    it('should reject incompatible feature dimensions', (): void => {
        const layer: Linear = new Linear(3, 2);
        const input: Tensor = new Tensor([1, 2], [2]);

        expect((): Tensor => layer.forward(input)).toThrow(LayerShapeError);
    });
});
