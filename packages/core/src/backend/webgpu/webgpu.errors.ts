/** Base error for the Edrion WebGPU backend. */
export class WebGPUBackendError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'WebGPUBackendError';
    }
}

/** Raised when WebGPU is unavailable in the current runtime. */
export class WebGPUUnavailableError extends WebGPUBackendError {
    public constructor() {
        super('WebGPU is not available in the current runtime. A compatible navigator.gpu implementation is required.');
        this.name = 'WebGPUUnavailableError';
    }
}
