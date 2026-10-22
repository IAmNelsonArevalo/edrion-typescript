import { NeuralNetworkError } from '@core/nn/nn.errors';

/**
 * Base error for Transformer-specific failures.
 */
export class TransformerError extends NeuralNetworkError {
    public constructor(message: string) {
        super(message);
        this.name = 'TransformerError';
    }
}

/**
 * Thrown when Transformer configuration is invalid.
 */
export class TransformerConfigurationError extends TransformerError {
    public constructor(message: string) {
        super(message);
        this.name = 'TransformerConfigurationError';
    }
}

/**
 * Thrown when attention tensor shapes are incompatible.
 */
export class AttentionShapeError extends TransformerError {
    public constructor(message: string) {
        super(message);
        this.name = 'AttentionShapeError';
    }
}
