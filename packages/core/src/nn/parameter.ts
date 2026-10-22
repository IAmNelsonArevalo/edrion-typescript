import { unravelTensorIndex } from '@core/tensor/operations/broadcasting';
import type { DType } from '@core/tensor/dtype';
import type { TensorShape } from '@core/tensor/shape';
import type { TensorData } from '@core/tensor/storage';
import { TensorDataError } from '@core/tensor/tensor.errors';
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

    /**
     * Replaces parameter values without creating an autograd operation.
     *
     * Optimizers use this method to apply parameter updates while
     * preserving the Parameter object and its registration identity.
     */
    public assign(data: TensorData): void {
        if (data.length !== this.size) {
            throw new TensorDataError(
                `Parameter assignment length ${data.length} does not match parameter size ${this.size}.`,
            );
        }

        const shape: TensorShape = this.shape;

        for (let offset: number = 0; offset < data.length; offset += 1) {
            const value: number | undefined = data[offset];

            if (value === undefined) {
                throw new TensorDataError(`Missing parameter assignment value at flat offset ${offset}.`);
            }

            const indices: number[] = unravelTensorIndex(offset, shape);

            this.set(indices, value);
        }
    }
}
