/** Base error for every Tensor-related failure. */
export class TensorError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'TensorError';
    }
}

/** Thrown when a tensor shape is invalid. */
export class TensorShapeError extends TensorError {
    public constructor(message: string) {
        super(message);
        this.name = 'TensorShapeError';
    }
}

/** Thrown when tensor data does not match the tensor structure. */
export class TensorDataError extends TensorError {
    public constructor(message: string) {
        super(message);
        this.name = 'TensorDataError';
    }
}

/** Thrown when tensor indexing is invalid. */
export class TensorIndexError extends TensorError {
    public constructor(message: string) {
        super(message);
        this.name = 'TensorIndexError';
    }
}

/** Thrown when a tensor reshape is invalid. */
export class TensorReshapeError extends TensorError {
    public constructor(message: string) {
        super(message);
        this.name = 'TensorReshapeError';
    }
}

/** Thrown when an unsupported tensor data type is requested. */
export class TensorDTypeError extends TensorError {
    public constructor(message: string) {
        super(message);
        this.name = 'TensorDTypeError';
    }
}
