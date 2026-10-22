/** Base error for Edrion serialization failures. */
export class SerializationError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'SerializationError';
    }
}

/** Raised when serialized model state is invalid. */
export class ModelStateError extends SerializationError {
    public constructor(message: string) {
        super(message);
        this.name = 'ModelStateError';
    }
}
