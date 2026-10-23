export {
    DEFAULT_DTYPE,
    SUPPORTED_DTYPES,
    assertSupportedDType,
    isSupportedDType,
    quantizeBFloat16,
    quantizeFloat16,
    quantizeToDType,
    type DType,
} from './dtype';

export {
    broadcastShapes,
    binaryOperation,
    matmulOperation,
    meanOperation,
    powerOperation,
    reduceBroadcastGradient,
    resolveBroadcastOffset,
    sumOperation,
    transposeMatrix,
    unaryOperation,
    unravelTensorIndex,
    type BinaryOperation,
    type ReductionOperation,
    type ReductionResult,
    type UnaryOperation,
} from './operations';

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

export { Tensor, type TensorOperand, type TensorOptions } from './tensor';

export { full, ones, scalar, zeros } from './tensor.factory';

export { buildNestedTensorArray, calculateTensorOffset, type TensorArray } from './tensor.utils';

export {
    TensorDataError,
    TensorDTypeError,
    TensorError,
    TensorIndexError,
    TensorReshapeError,
    TensorShapeError,
} from './tensor.errors';
