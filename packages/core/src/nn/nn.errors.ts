/**
 * Base error for neural-network related failures.
 */
export class NeuralNetworkError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'NeuralNetworkError';
    }
}

/**
 * Thrown when a module registration contract is violated.
 */
export class ModuleRegistrationError extends NeuralNetworkError {
    public constructor(message: string) {
        super(message);
        this.name = 'ModuleRegistrationError';
    }
}

/**
 * Thrown when a neural-network layer receives an incompatible shape.
 */
export class LayerShapeError extends NeuralNetworkError {
    public constructor(message: string) {
        super(message);
        this.name = 'LayerShapeError';
    }
}

/**
 * Thrown when an embedding index is invalid.
 */
export class EmbeddingIndexError extends NeuralNetworkError {
    public constructor(message: string) {
        super(message);
        this.name = 'EmbeddingIndexError';
    }
}
