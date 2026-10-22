import { TensorShapeError } from '@core/tensor/tensor.errors';

/** Tensor dimensions. Empty shape [] represents a scalar tensor. */
export type TensorShape = readonly number[];

/** Validates a tensor shape. */
export function validateTensorShape(shape: TensorShape): void {
    for (let index: number = 0; index < shape.length; index += 1) {
        const dimension: number | undefined = shape[index];

        if (dimension === undefined) {
            throw new TensorShapeError(`Missing tensor dimension at index ${index}.`);
        }

        if (!Number.isInteger(dimension)) {
            throw new TensorShapeError(`Tensor dimension at index ${index} must be an integer. Received ${dimension}.`);
        }

        if (dimension <= 0) {
            throw new TensorShapeError(
                `Tensor dimension at index ${index} must be greater than zero. Received ${dimension}.`,
            );
        }
    }
}

/** Returns a normalized immutable tensor shape. */
export function normalizeTensorShape(shape: TensorShape): TensorShape {
    validateTensorShape(shape);
    return [...shape];
}

/** Calculates the number of scalar values required by a tensor shape. */
export function calculateTensorSize(shape: TensorShape): number {
    validateTensorShape(shape);

    if (shape.length === 0) {
        return 1;
    }

    let size: number = 1;

    for (let index: number = 0; index < shape.length; index += 1) {
        const dimension: number | undefined = shape[index];

        if (dimension === undefined) {
            throw new TensorShapeError(`Missing tensor dimension at index ${index}.`);
        }

        size *= dimension;
    }

    return size;
}

/** Determines whether two tensor shapes are structurally equal. */
export function tensorShapesEqual(left: TensorShape, right: TensorShape): boolean {
    if (left.length !== right.length) {
        return false;
    }

    for (let index: number = 0; index < left.length; index += 1) {
        const leftDimension: number | undefined = left[index];
        const rightDimension: number | undefined = right[index];

        if (leftDimension === undefined || rightDimension === undefined) {
            return false;
        }

        if (leftDimension !== rightDimension) {
            return false;
        }
    }

    return true;
}
