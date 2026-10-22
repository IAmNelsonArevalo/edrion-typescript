import { Tensor, type TensorOptions } from '@core/tensor/tensor';
import type { TensorShape } from '@core/tensor/shape';

/**
 * Creates a Tensor filled with zeros.
 */
export function zeros(shape: TensorShape, options: TensorOptions = {}): Tensor {
    return Tensor.zeros(shape, options);
}

/**
 * Creates a Tensor filled with ones.
 */
export function ones(shape: TensorShape, options: TensorOptions = {}): Tensor {
    return Tensor.ones(shape, options);
}

/**
 * Creates a Tensor filled with a specific value.
 */
export function full(shape: TensorShape, value: number, options: TensorOptions = {}): Tensor {
    return Tensor.full(shape, value, options);
}

/**
 * Creates a scalar Tensor.
 */
export function scalar(value: number, options: TensorOptions = {}): Tensor {
    return Tensor.scalar(value, options);
}
