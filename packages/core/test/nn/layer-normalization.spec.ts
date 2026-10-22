import { describe, expect, it } from 'vitest';

import { LayerNormalization, LayerShapeError, Tensor } from '@core';

describe('LayerNormalization', (): void => {
    it('should normalize the last dimension', (): void => {
        const layer: LayerNormalization = new LayerNormalization(2, {
            affine: false,
        });

        const input: Tensor = new Tensor([1, 3, 2, 4], [2, 2]);

        const output: Tensor = layer.forward(input);
        const values: number[] = output.toFlatArray();

        expect(values[0]).toBeCloseTo(-1, 4);
        expect(values[1]).toBeCloseTo(1, 4);
        expect(values[2]).toBeCloseTo(-1, 4);
        expect(values[3]).toBeCloseTo(1, 4);
    });

    it('should expose affine parameters', (): void => {
        const layer: LayerNormalization = new LayerNormalization(3);

        expect(layer.parameters()).toHaveLength(2);
        expect(layer.weight?.toFlatArray()).toEqual([1, 1, 1]);
        expect(layer.bias?.toFlatArray()).toEqual([0, 0, 0]);
    });

    it('should propagate gradients', (): void => {
        const layer: LayerNormalization = new LayerNormalization(2);

        const input: Tensor = new Tensor([1, 3], [1, 2], {
            requiresGrad: true,
        });

        layer.forward(input).sum().backward();

        expect(input.grad).toBeDefined();
        expect(layer.weight?.grad).toBeDefined();
        expect(layer.bias?.grad).toBeDefined();
    });

    it('should reject an incompatible final dimension', (): void => {
        const layer: LayerNormalization = new LayerNormalization(3);

        const input: Tensor = new Tensor([1, 2], [2]);

        expect((): Tensor => layer.forward(input)).toThrow(LayerShapeError);
    });
});
