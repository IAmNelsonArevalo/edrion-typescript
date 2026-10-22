import { TensorDataError, TensorIndexError } from '@core/tensor/tensor.errors';

/**
 * Supported element-wise unary operations.
 */
export type UnaryOperation = 'neg' | 'exp' | 'log' | 'sqrt';

/**
 * Executes an element-wise unary operation.
 */
export function unaryOperation(data: Float32Array, operation: UnaryOperation): Float32Array {
    const result: Float32Array = new Float32Array(data.length);

    for (let index: number = 0; index < data.length; index += 1) {
        const value: number | undefined = data[index];

        if (value === undefined) {
            throw new TensorIndexError(`Tensor value at flat offset ${index} does not exist.`);
        }

        result[index] = applyUnaryOperation(value, operation);
    }

    return result;
}

/**
 * Raises every tensor element to a scalar exponent.
 */
export function powerOperation(data: Float32Array, exponent: number): Float32Array {
    const result: Float32Array = new Float32Array(data.length);

    for (let index: number = 0; index < data.length; index += 1) {
        const value: number | undefined = data[index];

        if (value === undefined) {
            throw new TensorIndexError(`Tensor value at flat offset ${index} does not exist.`);
        }

        result[index] = Math.pow(value, exponent);
    }

    return result;
}

function applyUnaryOperation(value: number, operation: UnaryOperation): number {
    switch (operation) {
        case 'neg':
            return -value;

        case 'exp':
            return Math.exp(value);

        case 'log':
            return Math.log(value);

        case 'sqrt':
            return Math.sqrt(value);

        default: {
            const exhaustiveOperation: never = operation;

            throw new TensorDataError(`Unsupported unary operation "${String(exhaustiveOperation)}".`);
        }
    }
}
