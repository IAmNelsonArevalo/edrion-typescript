import { calculateTensorOffset } from '@core/tensor/tensor.utils';
import { calculateTensorSize, calculateTensorStrides, type TensorShape } from '@core/tensor/shape';
import { unravelTensorIndex } from '@core/tensor/operations/broadcasting';
import { Tensor } from '@core/tensor/tensor';
import { TensorShapeError } from '@core/tensor/tensor.errors';
import { assertFiniteTensor } from '@core/stability/numerical-validation';

/** Normalizes a possibly-negative tensor axis. */
export function normalizeAxis(axis: number, rank: number): number {
    if (!Number.isInteger(axis)) {
        throw new TensorShapeError(`Tensor axis must be an integer. Received ${axis}.`);
    }

    const normalizedAxis: number = axis < 0 ? axis + rank : axis;

    if (normalizedAxis < 0 || normalizedAxis >= rank) {
        throw new TensorShapeError(`Tensor axis ${axis} is outside rank ${rank}.`);
    }

    return normalizedAxis;
}

/**
 * Builds a detached keep-dims tensor containing the maximum value for each
 * slice along one axis. Detaching the max is safe for stabilized softmax and
 * log-sum-exp because the subtraction is algebraically shift-invariant.
 */
export function createDetachedAxisMax(input: Tensor, axis: number): Tensor {
    if (input.rank === 0) {
        throw new TensorShapeError('Axis maximum requires a tensor with rank greater than zero.');
    }

    assertFiniteTensor(input, 'Axis-maximum input');

    const normalizedAxis: number = normalizeAxis(axis, input.rank);
    const inputShape: TensorShape = input.shape;
    const outputShape: number[] = [...inputShape];
    outputShape[normalizedAxis] = 1;

    const outputSize: number = calculateTensorSize(outputShape);
    const outputData: Float32Array = new Float32Array(outputSize);
    outputData.fill(Number.NEGATIVE_INFINITY);

    const outputStrides = calculateTensorStrides(outputShape);
    const inputData: Float32Array = input.toFloat32Array();

    for (let flatIndex: number = 0; flatIndex < inputData.length; flatIndex += 1) {
        const value: number | undefined = inputData[flatIndex];

        if (value === undefined) {
            throw new TensorShapeError(`Missing tensor value at flat offset ${flatIndex}.`);
        }

        const coordinates: number[] = unravelTensorIndex(flatIndex, inputShape);
        coordinates[normalizedAxis] = 0;

        const outputOffset: number = calculateTensorOffset(coordinates, outputShape, outputStrides);
        const currentMaximum: number | undefined = outputData[outputOffset];

        if (currentMaximum === undefined) {
            throw new TensorShapeError(`Missing axis maximum at flat offset ${outputOffset}.`);
        }

        if (value > currentMaximum) {
            outputData[outputOffset] = value;
        }
    }

    return new Tensor(outputData, outputShape, {
        dtype: input.dtype,
        backend: input.backend,
        requiresGrad: false,
    });
}
