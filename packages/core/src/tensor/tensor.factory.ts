import { calculateTensorSize, type TensorShape } from '@core/tensor/shape';
import { Tensor, type TensorOptions } from '@core/tensor/tensor';

/** Creates a tensor filled with zeros. */
export function zeros(shape: TensorShape, options: TensorOptions = {}): Tensor {
    return full(shape, 0, options);
}

/** Creates a tensor filled with ones. */
export function ones(shape: TensorShape, options: TensorOptions = {}): Tensor {
    return full(shape, 1, options);
}

/** Creates a tensor filled with one value. */
export function full(shape: TensorShape, value: number, options: TensorOptions = {}): Tensor {
    const size: number = calculateTensorSize(shape);
    const data: Float32Array = new Float32Array(size);
    data.fill(value);
    return new Tensor(data, shape, options);
}

/** Creates a scalar tensor. */
export function scalar(value: number, options: TensorOptions = {}): Tensor {
    const data: Float32Array = new Float32Array([value]);
    return new Tensor(data, [], options);
}
