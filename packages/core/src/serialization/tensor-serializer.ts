import type { DType } from '@core/tensor/dtype';
import { Tensor } from '@core/tensor/tensor';
import { SerializationError } from '@core/serialization/serialization.errors';
import { assertFiniteTensor } from '@core/stability/numerical-validation';

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
        assertFiniteTensor(tensor, 'Serialized tensor');

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

        for (let index: number = 0; index < serialized.data.length; index += 1) {
            const value: number | undefined = serialized.data[index];

            if (value === undefined || !Number.isFinite(value)) {
                throw new SerializationError(`Serialized tensor contains a non-finite value at flat offset ${index}.`);
            }
        }

        return new Tensor(serialized.data, serialized.shape, {
            dtype: serialized.dtype,
            backend: serialized.backend,
        });
    }
}
