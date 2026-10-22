import { TensorDTypeError } from '@core/tensor/tensor.errors';

/**
 * Data types currently supported by Edrion tensors.
 *
 * Additional types such as float64, int32 and bool
 * may be introduced in future releases.
 */
export const SUPPORTED_DTYPES = ['float32'] as const;

/**
 * Tensor data type.
 */
export type DType = (typeof SUPPORTED_DTYPES)[number];

/**
 * Default tensor data type.
 */
export const DEFAULT_DTYPE: DType = 'float32';

/**
 * Determines whether a value represents
 * a supported Edrion tensor data type.
 */
export function isSupportedDType(dtype: string): dtype is DType {
    return (SUPPORTED_DTYPES as readonly string[]).includes(dtype);
}

/**
 * Validates a tensor data type.
 */
export function assertSupportedDType(dtype: string): asserts dtype is DType {
    if (!isSupportedDType(dtype)) {
        throw new TensorDTypeError(`Unsupported tensor dtype "${dtype}".`);
    }
}
