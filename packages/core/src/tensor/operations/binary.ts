import { calculateTensorSize, calculateTensorStrides, type TensorShape, type TensorStrides } from '@core/tensor/shape';
import { TensorDataError, TensorIndexError } from '@core/tensor/tensor.errors';
import { broadcastShapes, resolveBroadcastOffset, unravelTensorIndex } from '@core/tensor/operations/broadcasting';

/**
 * Supported element-wise binary operations.
 */
export type BinaryOperation = 'add' | 'sub' | 'mul' | 'div';

/**
 * Executes an element-wise binary operation with broadcasting.
 */
export function binaryOperation(
    leftData: Float32Array,
    leftShape: TensorShape,
    rightData: Float32Array,
    rightShape: TensorShape,
    operation: BinaryOperation,
): {
    readonly data: Float32Array;
    readonly shape: TensorShape;
} {
    const resultShape: TensorShape = broadcastShapes(leftShape, rightShape);
    const resultSize: number = calculateTensorSize(resultShape);
    const leftStrides: TensorStrides = calculateTensorStrides(leftShape);
    const rightStrides: TensorStrides = calculateTensorStrides(rightShape);
    const result: Float32Array = new Float32Array(resultSize);

    for (let offset: number = 0; offset < resultSize; offset += 1) {
        const resultIndices: number[] = unravelTensorIndex(offset, resultShape);
        const leftOffset: number = resolveBroadcastOffset(resultIndices, leftShape, leftStrides);
        const rightOffset: number = resolveBroadcastOffset(resultIndices, rightShape, rightStrides);

        const leftValue: number = readValue(leftData, leftOffset);
        const rightValue: number = readValue(rightData, rightOffset);

        result[offset] = applyBinaryOperation(leftValue, rightValue, operation);
    }

    return {
        data: result,
        shape: resultShape,
    };
}

function applyBinaryOperation(left: number, right: number, operation: BinaryOperation): number {
    switch (operation) {
        case 'add':
            return left + right;

        case 'sub':
            return left - right;

        case 'mul':
            return left * right;

        case 'div':
            return left / right;

        default: {
            const exhaustiveOperation: never = operation;

            throw new TensorDataError(`Unsupported binary operation "${String(exhaustiveOperation)}".`);
        }
    }
}

function readValue(data: Float32Array, offset: number): number {
    const value: number | undefined = data[offset];

    if (value === undefined) {
        throw new TensorIndexError(`Tensor value at flat offset ${offset} does not exist.`);
    }

    return value;
}
