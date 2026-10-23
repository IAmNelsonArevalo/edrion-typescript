/** Base error for Edrion dataset and data-loading failures. */
export class DataPipelineError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'DataPipelineError';
    }
}

/** Raised when a dataset index is invalid. */
export class DatasetIndexError extends DataPipelineError {
    public constructor(message: string) {
        super(message);
        this.name = 'DatasetIndexError';
    }
}

/** Raised when a DataLoader configuration is invalid. */
export class DataLoaderConfigError extends DataPipelineError {
    public constructor(message: string) {
        super(message);
        this.name = 'DataLoaderConfigError';
    }
}
