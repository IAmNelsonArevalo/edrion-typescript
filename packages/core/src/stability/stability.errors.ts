/** Base error for numerical-safety failures. */
export class NumericalSafetyError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'NumericalSafetyError';
    }
}

/** Thrown when tensor values contain NaN or Infinity. */
export class NonFiniteTensorError extends NumericalSafetyError {
    public constructor(message: string) {
        super(message);
        this.name = 'NonFiniteTensorError';
    }
}

/** Thrown when gradients contain NaN or Infinity. */
export class NonFiniteGradientError extends NumericalSafetyError {
    public constructor(message: string) {
        super(message);
        this.name = 'NonFiniteGradientError';
    }
}

/** Thrown when gradient clipping configuration is invalid. */
export class GradientClippingError extends NumericalSafetyError {
    public constructor(message: string) {
        super(message);
        this.name = 'GradientClippingError';
    }
}
