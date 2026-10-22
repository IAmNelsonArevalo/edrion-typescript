import type { TensorShape, TensorStrides } from '@core/tensor/shape';
import { TensorIndexError, TensorShapeError } from '@core/tensor/tensor.errors';

/** Recursive JavaScript representation of tensor data. */
export type TensorArray = number | TensorArray[];

/** Calculates a flat row-major storage offset from tensor coordinates. */
export function calculateTensorOffset(indices: readonly number[], shape: TensorShape, strides: TensorStrides): number {
    const rank: number = shape.length;

    if (indices.length !== rank) {
        throw new TensorIndexError(`Expected ${rank} tensor indices but received ${indices.length}.`);
    }

    if (strides.length !== rank) {
        throw new TensorShapeError(`Tensor strides rank ${strides.length} does not match shape rank ${rank}.`);
    }

    if (rank === 0) {
        return 0;
    }

    let offset: number = 0;

    for (let axis: number = 0; axis < rank; axis += 1) {
        const index: number | undefined = indices[axis];
        const dimension: number | undefined = shape[axis];
        const stride: number | undefined = strides[axis];

        if (index === undefined || dimension === undefined || stride === undefined) {
            throw new TensorIndexError(`Unable to resolve tensor index information for axis ${axis}.`);
        }

        if (!Number.isInteger(index)) {
            throw new TensorIndexError(`Tensor index at axis ${axis} must be an integer. Received ${index}.`);
        }

        if (index < 0 || index >= dimension) {
            throw new TensorIndexError(`Tensor index ${index} is outside axis ${axis} with dimension ${dimension}.`);
        }

        offset += index * stride;
    }

    return offset;
}

/** Builds a nested JavaScript representation from flat row-major data. */
export function buildNestedTensorArray(values: readonly number[], shape: TensorShape): TensorArray {
    if (shape.length === 0) {
        const scalarValue: number | undefined = values[0];

        if (scalarValue === undefined) {
            throw new TensorShapeError('Scalar tensor does not contain a value.');
        }

        return scalarValue;
    }

    return buildNestedDimension(values, shape, 0, 0);
}

function buildNestedDimension(
    values: readonly number[],
    shape: TensorShape,
    axis: number,
    startOffset: number,
): TensorArray[] {
    const dimension: number | undefined = shape[axis];

    if (dimension === undefined) {
        throw new TensorShapeError(`Missing tensor dimension at axis ${axis}.`);
    }

    const result: TensorArray[] = [];
    const isLastAxis: boolean = axis === shape.length - 1;

    if (isLastAxis) {
        for (let index: number = 0; index < dimension; index += 1) {
            const flatOffset: number = startOffset + index;
            const value: number | undefined = values[flatOffset];

            if (value === undefined) {
                throw new TensorShapeError(`Unable to read tensor value at flat offset ${flatOffset}.`);
            }

            result.push(value);
        }

        return result;
    }

    let childSize: number = 1;

    for (let childAxis: number = axis + 1; childAxis < shape.length; childAxis += 1) {
        const childDimension: number | undefined = shape[childAxis];

        if (childDimension === undefined) {
            throw new TensorShapeError(`Missing tensor dimension at axis ${childAxis}.`);
        }

        childSize *= childDimension;
    }

    for (let index: number = 0; index < dimension; index += 1) {
        const childOffset: number = startOffset + index * childSize;
        const child: TensorArray[] = buildNestedDimension(values, shape, axis + 1, childOffset);
        result.push(child);
    }

    return result;
}
