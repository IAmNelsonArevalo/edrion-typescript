/** Base error for Edrion training failures. */
export class TrainingError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'TrainingError';
    }
}

/** Raised when trainer configuration is invalid. */
export class TrainingConfigError extends TrainingError {
    public constructor(message: string) {
        super(message);
        this.name = 'TrainingConfigError';
    }
}

/** Raised when a training dataset is invalid. */
export class TrainingDataError extends TrainingError {
    public constructor(message: string) {
        super(message);
        this.name = 'TrainingDataError';
    }
}
