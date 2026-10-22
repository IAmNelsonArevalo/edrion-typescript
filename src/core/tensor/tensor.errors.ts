/**
 * Base class for Tensor-related errors.
 */
export class TensorError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'TensorError';
    }
}

/**
 * Error produced by an invalid tensor shape.
 */
export class TensorShapeError extends TensorError {
    constructor(message: string) {
        super(message);
        this.name = 'TensorShapeError';
    }
}

/**
 * Error produced when tensor data is invalid
 * for the requested shape.
 */
export class TensorDataError extends TensorError {
    constructor(message: string) {
        super(message);
        this.name = 'TensorDataError';
    }
}

/**
 * Error produced by an invalid tensor index.
 */
export class TensorIndexError extends TensorError {
    constructor(message: string) {
        super(message);
        this.name = 'TensorIndexError';
    }
}

/**
 * Error produced by an invalid reshape operation.
 */
export class TensorReshapeError extends TensorError {
    constructor(message: string) {
        super(message);
        this.name = 'TensorReshapeError';
    }
}

/**
 * Error produced by an unsupported tensor dtype.
 */
export class TensorDTypeError extends TensorError {
    constructor(message: string) {
        super(message);
        this.name = 'TensorDTypeError';
    }
}
