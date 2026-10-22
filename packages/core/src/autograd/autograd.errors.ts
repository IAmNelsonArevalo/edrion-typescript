/**
 * Base error for automatic differentiation failures.
 */
export class AutogradError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'AutogradError';
    }
}

/**
 * Thrown when a supplied backward gradient is invalid.
 */
export class InvalidGradientError extends AutogradError {
    public constructor(message: string) {
        super(message);
        this.name = 'InvalidGradientError';
    }
}
