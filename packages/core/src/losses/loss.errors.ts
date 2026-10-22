/**
 * Base error for loss-function failures.
 */
export class LossError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'LossError';
    }
}

/**
 * Thrown when prediction and target shapes are incompatible.
 */
export class LossShapeError extends LossError {
    public constructor(message: string) {
        super(message);
        this.name = 'LossShapeError';
    }
}

/**
 * Thrown when a class target is invalid for cross entropy.
 */
export class LossTargetError extends LossError {
    public constructor(message: string) {
        super(message);
        this.name = 'LossTargetError';
    }
}
