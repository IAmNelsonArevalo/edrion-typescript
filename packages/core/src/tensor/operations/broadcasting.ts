import { calculateTensorSize, calculateTensorStrides, type TensorShape, type TensorStrides } from '@core/tensor/shape';
import { TensorIndexError, TensorShapeError } from '@core/tensor/tensor.errors';

/**
 * Calculates the broadcasted shape for two operands.
 */
export function broadcastShapes(left: TensorShape, right: TensorShape): TensorShape {
    const resultRank: number = Math.max(left.length, right.length);
    const result: number[] = new Array<number>(resultRank);

    for (let axisFromEnd: number = 0; axisFromEnd < resultRank; axisFromEnd += 1) {
        const leftIndex: number = left.length - 1 - axisFromEnd;
        const rightIndex: number = right.length - 1 - axisFromEnd;
        const resultIndex: number = resultRank - 1 - axisFromEnd;

        const leftDimension: number = leftIndex >= 0 ? readDimension(left, leftIndex) : 1;
        const rightDimension: number = rightIndex >= 0 ? readDimension(right, rightIndex) : 1;

        if (leftDimension !== rightDimension && leftDimension !== 1 && rightDimension !== 1) {
            throw new TensorShapeError(`Cannot broadcast shapes [${left.join(', ')}] and [${right.join(', ')}].`);
        }

        result[resultIndex] = Math.max(leftDimension, rightDimension);
    }

    return result;
}

/**
 * Converts a flat row-major offset to tensor coordinates.
 */
export function unravelTensorIndex(offset: number, shape: TensorShape): number[] {
    const size: number = calculateTensorSize(shape);

    if (!Number.isInteger(offset) || offset < 0 || offset >= size) {
        throw new TensorIndexError(`Tensor offset ${offset} is outside the valid range 0-${size - 1}.`);
    }

    if (shape.length === 0) {
        return [];
    }

    const strides: TensorStrides = calculateTensorStrides(shape);
    const indices: number[] = new Array<number>(shape.length);
    let remainder: number = offset;

    for (let axis: number = 0; axis < shape.length; axis += 1) {
        const stride: number = readStride(strides, axis);
        const index: number = Math.floor(remainder / stride);

        indices[axis] = index;
        remainder %= stride;
    }

    return indices;
}

/**
 * Resolves the source storage offset for an operand being broadcast
 * into the supplied result coordinates.
 */
export function resolveBroadcastOffset(
    resultIndices: readonly number[],
    sourceShape: TensorShape,
    sourceStrides: TensorStrides,
): number {
    if (sourceShape.length !== sourceStrides.length) {
        throw new TensorShapeError(
            `Tensor shape rank ${sourceShape.length} does not match stride rank ${sourceStrides.length}.`,
        );
    }

    if (sourceShape.length > resultIndices.length) {
        throw new TensorShapeError(
            `Source rank ${sourceShape.length} cannot be broadcast to result rank ${resultIndices.length}.`,
        );
    }

    const rankOffset: number = resultIndices.length - sourceShape.length;
    let sourceOffset: number = 0;

    for (let sourceAxis: number = 0; sourceAxis < sourceShape.length; sourceAxis += 1) {
        const sourceDimension: number = readDimension(sourceShape, sourceAxis);
        const sourceStride: number = readStride(sourceStrides, sourceAxis);
        const resultAxis: number = rankOffset + sourceAxis;
        const resultIndex: number | undefined = resultIndices[resultAxis];

        if (resultIndex === undefined) {
            throw new TensorIndexError(`Missing broadcast result index at axis ${resultAxis}.`);
        }

        const sourceIndex: number = sourceDimension === 1 ? 0 : resultIndex;

        if (sourceIndex < 0 || sourceIndex >= sourceDimension) {
            throw new TensorIndexError(
                `Broadcast index ${sourceIndex} is outside source axis ${sourceAxis} with dimension ${sourceDimension}.`,
            );
        }

        sourceOffset += sourceIndex * sourceStride;
    }

    return sourceOffset;
}

/**
 * Reduces a gradient from a broadcasted result shape back to
 * the original operand shape.
 */
export function reduceBroadcastGradient(
    gradient: Float32Array,
    gradientShape: TensorShape,
    targetShape: TensorShape,
): Float32Array {
    const gradientSize: number = calculateTensorSize(gradientShape);

    if (gradient.length !== gradientSize) {
        throw new TensorShapeError(
            `Gradient data length ${gradient.length} does not match shape [${gradientShape.join(', ')}].`,
        );
    }

    broadcastShapes(targetShape, gradientShape);

    const targetSize: number = calculateTensorSize(targetShape);
    const targetStrides: TensorStrides = calculateTensorStrides(targetShape);
    const reduced: Float32Array = new Float32Array(targetSize);

    for (let offset: number = 0; offset < gradientSize; offset += 1) {
        const resultIndices: number[] = unravelTensorIndex(offset, gradientShape);
        const targetOffset: number = resolveBroadcastOffset(resultIndices, targetShape, targetStrides);
        const value: number | undefined = gradient[offset];

        if (value === undefined) {
            throw new TensorIndexError(`Gradient value at offset ${offset} does not exist.`);
        }

        const currentValue: number | undefined = reduced[targetOffset];

        if (currentValue === undefined) {
            throw new TensorIndexError(`Reduced gradient offset ${targetOffset} does not exist.`);
        }

        reduced[targetOffset] = currentValue + value;
    }

    return reduced;
}

function readDimension(shape: TensorShape, axis: number): number {
    const dimension: number | undefined = shape[axis];

    if (dimension === undefined) {
        throw new TensorShapeError(`Missing tensor dimension at axis ${axis}.`);
    }

    return dimension;
}

function readStride(strides: TensorStrides, axis: number): number {
    const stride: number | undefined = strides[axis];

    if (stride === undefined) {
        throw new TensorShapeError(`Missing tensor stride at axis ${axis}.`);
    }

    return stride;
}
