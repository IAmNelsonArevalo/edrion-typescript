import type { DType } from '@core/tensor/dtype';
import { Tensor } from '@core/tensor/tensor';
import { SerializationError } from '@core/serialization/serialization.errors';

/** JSON-safe tensor representation. */
export interface SerializedTensor {
    readonly shape: readonly number[];
    readonly dtype: DType;
    readonly backend: string;
    readonly data: readonly number[];
}

/** Converts tensors to and from JSON-safe representations. */
export class TensorSerializer {
    public static serialize(tensor: Tensor): SerializedTensor {
        return {
            shape: tensor.shape,
            dtype: tensor.dtype,
            backend: tensor.backend,
            data: tensor.toFlatArray(),
        };
    }

    public static deserialize(serialized: SerializedTensor): Tensor {
        if (serialized.data.length === 0) {
            throw new SerializationError('Serialized tensor data cannot be empty.');
        }

        return new Tensor(serialized.data, serialized.shape, {
            dtype: serialized.dtype,
            backend: serialized.backend,
        });
    }
}
