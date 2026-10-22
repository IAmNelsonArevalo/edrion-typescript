/**
 * Base error for optimizer failures.
 */
export class OptimizerError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'OptimizerError';
    }
}

/**
 * Thrown when optimizer configuration is invalid.
 */
export class OptimizerConfigurationError extends OptimizerError {
    public constructor(message: string) {
        super(message);
        this.name = 'OptimizerConfigurationError';
    }
}
