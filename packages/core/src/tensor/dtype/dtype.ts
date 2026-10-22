import { TensorDTypeError } from '@core/tensor/tensor.errors';

/** Data types currently supported by Edrion tensors. */
export const SUPPORTED_DTYPES = ['float32'] as const;

/** Supported tensor data type. */
export type DType = (typeof SUPPORTED_DTYPES)[number];

/** Default Edrion tensor data type. */
export const DEFAULT_DTYPE: DType = 'float32';

/** Determines whether a value represents a supported tensor data type. */
export function isSupportedDType(value: string): value is DType {
    return (SUPPORTED_DTYPES as readonly string[]).includes(value);
}

/** Ensures a tensor data type is supported. */
export function assertSupportedDType(value: string): asserts value is DType {
    if (!isSupportedDType(value)) {
        throw new TensorDTypeError(
            `Unsupported tensor dtype "${value}". Supported dtypes: ${SUPPORTED_DTYPES.join(', ')}.`,
        );
    }
}
