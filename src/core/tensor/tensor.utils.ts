import { TensorIndexError } from '@core/tensor/tensor.errors';
import type { TensorShape, TensorStrides } from '@core/tensor/shape';

/**
 * Recursive JavaScript representation of a Tensor.
 */
export type TensorArray = number | TensorArray[];

/**
 * Calculates the linear storage offset represented
 * by multidimensional tensor indices.
 */
export function calculateTensorOffset(shape: TensorShape, strides: TensorStrides, indices: readonly number[]): number {
    if (indices.length !== shape.length) throw new TensorIndexError(`Tensor rank is ${shape.length}, but ${indices.length} indices were provided.`);

    let offset = 0;

    for (let dimensionIndex = 0; dimensionIndex < shape.length; dimensionIndex += 1) {
        const index = indices[dimensionIndex],
            dimensionSize = shape[dimensionIndex],
            stride = strides[dimensionIndex];

        if (index === undefined || dimensionSize === undefined || stride === undefined)
            throw new TensorIndexError(`Unable to resolve tensor dimension ${dimensionIndex}.`);

        if (!Number.isInteger(index))
            throw new TensorIndexError(`Tensor index for dimension ${dimensionIndex} must be an integer. Received ${index}.`);

        if (index < 0 || index >= dimensionSize)
            throw new TensorIndexError(`Tensor index ${index} is out of bounds for dimension ${dimensionIndex} with size ${dimensionSize}.`);

        offset += index * stride;
    }

    return offset;
}

/**
 * Converts flat tensor storage into a nested
 * JavaScript array matching the tensor shape.
 */
export function buildNestedTensorArray(values: readonly number[], shape: TensorShape): TensorArray {
    if (shape.length === 0) {
        const value = values[0];

        if (value === undefined) throw new Error('Scalar tensor does not contain a value.');

        return value;
    }

    return buildDimension(values, shape, 0, 0).value;
}

interface BuildDimensionResult {
    value: TensorArray;
    nextOffset: number;
}

function buildDimension(values: readonly number[], shape: TensorShape, dimension: number, offset: number): BuildDimensionResult {
    const dimensionSize = shape[dimension];

    if (dimensionSize === undefined) {
        const value = values[offset];

        if (value === undefined) throw new Error(`Unable to resolve tensor value at offset ${offset}.`);

        return {
            value,
            nextOffset: offset + 1,
        };
    }

    const result: TensorArray[] = [];
    let currentOffset = offset;

    for (let index = 0; index < dimensionSize; index += 1) {
        const child = buildDimension(values, shape, dimension + 1, currentOffset);
        result.push(child.value);
        currentOffset = child.nextOffset;
    }

    return {
        value: result,
        nextOffset: currentOffset,
    };
}
