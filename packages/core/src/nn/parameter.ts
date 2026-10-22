import type { DType } from '@core/tensor/dtype';
import type { TensorShape } from '@core/tensor/shape';
import type { TensorData } from '@core/tensor/storage';
import { Tensor } from '@core/tensor/tensor';

/**
 * Parameter construction options.
 */
export interface ParameterOptions {
    readonly dtype?: DType;
    readonly backend?: string;
}

/**
 * Trainable Tensor owned by a neural-network Module.
 */
export class Parameter extends Tensor {
    public constructor(data: TensorData, shape: TensorShape, options: ParameterOptions = {}) {
        super(data, shape, {
            ...options,
            requiresGrad: true,
        });
    }
}
