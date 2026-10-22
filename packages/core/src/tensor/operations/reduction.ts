import { calculateTensorSize, calculateTensorStrides, type TensorShape, type TensorStrides } from '@core/tensor/shape';
import { TensorIndexError, TensorShapeError } from '@core/tensor/tensor.errors';
import { unravelTensorIndex } from '@core/tensor/operations/broadcasting';

/**
 * Supported reduction operations.
 */
export type ReductionOperation = 'sum' | 'mean';

export interface ReductionResult {
    readonly data: Float32Array;
    readonly shape: TensorShape;
    readonly normalizedAxis: number | undefined;
    readonly reductionSize: number;
}

/**
 * Calculates a tensor sum.
 */
export function sumOperation(
    data: Float32Array,
    shape: TensorShape,
    axis?: number,
    keepDims: boolean = false,
): ReductionResult {
    return reductionOperation(data, shape, 'sum', axis, keepDims);
}

/**
 * Calculates a tensor mean.
 */
export function meanOperation(
    data: Float32Array,
    shape: TensorShape,
    axis?: number,
    keepDims: boolean = false,
): ReductionResult {
    return reductionOperation(data, shape, 'mean', axis, keepDims);
}

function reductionOperation(
    data: Float32Array,
    shape: TensorShape,
    operation: ReductionOperation,
    axis?: number,
    keepDims: boolean = false,
): ReductionResult {
    const inputSize: number = calculateTensorSize(shape);

    if (data.length !== inputSize) {
        throw new TensorShapeError(`Tensor data length ${data.length} does not match shape [${shape.join(', ')}].`);
    }

    if (axis === undefined) {
        let total: number = 0;

        for (let index: number = 0; index < data.length; index += 1) {
            const value: number | undefined = data[index];

            if (value === undefined) {
                throw new TensorIndexError(`Tensor value at flat offset ${index} does not exist.`);
            }

            total += value;
        }

        if (operation === 'mean') {
            total /= inputSize;
        }

        const resultShape: TensorShape = keepDims ? new Array<number>(shape.length).fill(1) : [];

        return {
            data: new Float32Array([total]),
            shape: resultShape,
            normalizedAxis: undefined,
            reductionSize: inputSize,
        };
    }

    const normalizedAxis: number = normalizeAxis(axis, shape.length);
    const axisDimension: number = readDimension(shape, normalizedAxis);
    const resultShape: TensorShape = buildReductionShape(shape, normalizedAxis, keepDims);
    const resultSize: number = calculateTensorSize(resultShape);
    const result: Float32Array = new Float32Array(resultSize);
    const resultStrides: TensorStrides = calculateTensorStrides(resultShape);

    for (let inputOffset: number = 0; inputOffset < inputSize; inputOffset += 1) {
        const inputIndices: number[] = unravelTensorIndex(inputOffset, shape);
        const resultIndices: number[] = buildReductionIndices(inputIndices, normalizedAxis, keepDims);
        const resultOffset: number = calculateOffset(resultIndices, resultStrides);
        const value: number | undefined = data[inputOffset];

        if (value === undefined) {
            throw new TensorIndexError(`Tensor value at flat offset ${inputOffset} does not exist.`);
        }

        const currentValue: number | undefined = result[resultOffset];

        if (currentValue === undefined) {
            throw new TensorIndexError(`Reduction result offset ${resultOffset} does not exist.`);
        }

        result[resultOffset] = currentValue + value;
    }

    if (operation === 'mean') {
        for (let resultOffset: number = 0; resultOffset < result.length; resultOffset += 1) {
            const value: number | undefined = result[resultOffset];

            if (value === undefined) {
                throw new TensorIndexError(`Reduction result at offset ${resultOffset} does not exist.`);
            }

            result[resultOffset] = value / axisDimension;
        }
    }

    return {
        data: result,
        shape: resultShape,
        normalizedAxis,
        reductionSize: axisDimension,
    };
}

function normalizeAxis(axis: number, rank: number): number {
    if (!Number.isInteger(axis)) {
        throw new TensorShapeError(`Reduction axis must be an integer. Received ${axis}.`);
    }

    const normalizedAxis: number = axis < 0 ? rank + axis : axis;

    if (normalizedAxis < 0 || normalizedAxis >= rank) {
        throw new TensorShapeError(`Reduction axis ${axis} is outside tensor rank ${rank}.`);
    }

    return normalizedAxis;
}

function buildReductionShape(shape: TensorShape, axis: number, keepDims: boolean): TensorShape {
    const result: number[] = [];

    for (let currentAxis: number = 0; currentAxis < shape.length; currentAxis += 1) {
        const dimension: number = readDimension(shape, currentAxis);

        if (currentAxis === axis) {
            if (keepDims) {
                result.push(1);
            }

            continue;
        }

        result.push(dimension);
    }

    return result;
}

function buildReductionIndices(inputIndices: readonly number[], axis: number, keepDims: boolean): number[] {
    const result: number[] = [];

    for (let currentAxis: number = 0; currentAxis < inputIndices.length; currentAxis += 1) {
        const index: number | undefined = inputIndices[currentAxis];

        if (index === undefined) {
            throw new TensorIndexError(`Missing tensor index at axis ${currentAxis}.`);
        }

        if (currentAxis === axis) {
            if (keepDims) {
                result.push(0);
            }

            continue;
        }

        result.push(index);
    }

    return result;
}

function calculateOffset(indices: readonly number[], strides: TensorStrides): number {
    if (indices.length !== strides.length) {
        throw new TensorShapeError(`Index rank ${indices.length} does not match stride rank ${strides.length}.`);
    }

    let offset: number = 0;

    for (let axis: number = 0; axis < indices.length; axis += 1) {
        const index: number | undefined = indices[axis];
        const stride: number | undefined = strides[axis];

        if (index === undefined || stride === undefined) {
            throw new TensorIndexError(`Unable to resolve reduction index at axis ${axis}.`);
        }

        offset += index * stride;
    }

    return offset;
}

function readDimension(shape: TensorShape, axis: number): number {
    const dimension: number | undefined = shape[axis];

    if (dimension === undefined) {
        throw new TensorShapeError(`Missing tensor dimension at axis ${axis}.`);
    }

    return dimension;
}
