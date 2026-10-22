import type { TensorShape } from '@core/tensor/shape/shape';
import { validateTensorShape } from '@core/tensor/shape/shape';
import { TensorShapeError } from '@core/tensor/tensor.errors';

/** Row-major tensor strides. */
export type TensorStrides = readonly number[];

/** Calculates row-major strides for a tensor. */
export function calculateTensorStrides(shape: TensorShape): TensorStrides {
    validateTensorShape(shape);

    const rank: number = shape.length;

    if (rank === 0) {
        return [];
    }

    const strides: number[] = new Array<number>(rank);
    let stride: number = 1;

    for (let index: number = rank - 1; index >= 0; index -= 1) {
        const dimension: number | undefined = shape[index];

        if (dimension === undefined) {
            throw new TensorShapeError(`Missing tensor dimension at index ${index}.`);
        }

        strides[index] = stride;
        stride *= dimension;
    }

    return strides;
}
