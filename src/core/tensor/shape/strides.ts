import type { TensorShape } from '@core/tensor/shape/shape';
import { validateTensorShape } from '@core/tensor/shape/shape';

/**
 * Tensor strides used for row-major indexing.
 *
 * Example:
 *
 * shape:
 * [2, 3, 4]
 *
 * strides:
 * [12, 4, 1]
 */
export type TensorStrides = readonly number[];

/**
 * Calculates row-major tensor strides.
 */
export function calculateTensorStrides(shape: TensorShape): number[] {
    validateTensorShape(shape);

    if (shape.length === 0) return [];

    const strides = new Array<number>(shape.length);
    let stride = 1;

    for (let index = shape.length - 1; index >= 0; index -= 1) {
        strides[index] = stride;
        stride *= shape[index] ?? 0;
    }

    return strides;
}
