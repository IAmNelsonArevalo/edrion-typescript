export { DEFAULT_DTYPE, SUPPORTED_DTYPES, assertSupportedDType, isSupportedDType, type DType } from './dtype';
export {
    calculateTensorSize,
    calculateTensorStrides,
    normalizeTensorShape,
    tensorShapesEqual,
    validateTensorShape,
    type TensorShape,
    type TensorStrides,
} from './shape';
export { TensorStorage, type TensorData } from './storage';
export { Tensor, type TensorOptions } from './tensor';
export { full, ones, scalar, zeros } from './tensor.factory';
export { buildNestedTensorArray, calculateTensorOffset, type TensorArray } from './tensor.utils';
export { TensorDataError, TensorDTypeError, TensorError, TensorIndexError, TensorReshapeError, TensorShapeError } from './tensor.errors';
