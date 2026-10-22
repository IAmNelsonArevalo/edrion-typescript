import { TensorShapeError } from '@core/tensor/tensor.errors';

/**
 * Shape describing the dimensions of a tensor.
 *
 * Examples:
 *
 * Scalar:
 * []
 *
 * Vector:
 * [3]
 *
 * Matrix:
 * [2, 3]
 *
 * Rank 3:
 * [2, 3, 4]
 */
export type TensorShape = readonly number[];

/**
 * Validates a tensor shape.
 *
 * Dimensions must be non-negative integers.
 *
 * An empty shape is valid and represents
 * a scalar tensor.
 */
export function validateTensorShape(shape: TensorShape): void {
    for (let dimensionIndex = 0; dimensionIndex < shape.length; dimensionIndex += 1) {
        const dimension = shape[dimensionIndex];

        if (dimension === undefined) throw new TensorShapeError(`Missing tensor dimension at index ${dimensionIndex}.`);

        if (!Number.isInteger(dimension)) throw new TensorShapeError(`Tensor dimension ${dimensionIndex} must be an integer. Received ${dimension}.`);

        if (dimension < 0) throw new TensorShapeError(`Tensor dimension ${dimensionIndex} cannot be negative. Received ${dimension}.`);
    }
}

/**
 * Calculates the total number of elements
 * represented by a shape.
 *
 * An empty shape represents a scalar and
 * therefore contains exactly one element.
 */
export function calculateTensorSize(shape: TensorShape): number {
    validateTensorShape(shape);
    if (shape.length === 0) return 1;
    return shape.reduce((size, dimension) => size * dimension, 1);
}

/**
 * Creates an owned mutable copy of a shape.
 */
export function normalizeTensorShape(shape: TensorShape): number[] {
    validateTensorShape(shape);
    return [...shape];
}

/**
 * Determines whether two shapes are equal.
 */
export function tensorShapesEqual(first: TensorShape, second: TensorShape): boolean {
    if (first.length !== second.length) return false;

    return first.every((dimension, index) => dimension === second[index]);
}
