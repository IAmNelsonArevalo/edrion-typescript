import type { Module } from '@core/nn/module';
import type { Parameter } from '@core/nn/parameter';
import type { Tensor } from '@core/tensor/tensor';
import { NonFiniteGradientError, NonFiniteTensorError } from '@core/stability/stability.errors';

/** Ensures a scalar number is finite. */
export function assertFiniteNumber(value: number, label: string): void {
    if (!Number.isFinite(value)) {
        throw new NonFiniteTensorError(`${label} must be finite. Received ${value}.`);
    }
}

/** Ensures every tensor value is finite. */
export function assertFiniteTensor(tensor: Tensor, label: string = 'Tensor'): void {
    const values: Float32Array = tensor.toFloat32Array();

    for (let index: number = 0; index < values.length; index += 1) {
        const value: number | undefined = values[index];

        if (value === undefined) {
            throw new NonFiniteTensorError(`${label} value at flat offset ${index} does not exist.`);
        }

        if (!Number.isFinite(value)) {
            throw new NonFiniteTensorError(`${label} contains non-finite value ${value} at flat offset ${index}.`);
        }
    }
}

/** Ensures the accumulated gradient of a tensor is finite when present. */
export function assertFiniteGradient(tensor: Tensor, label: string = 'Tensor gradient'): void {
    const gradient: Tensor | undefined = tensor.grad;

    if (gradient === undefined) {
        return;
    }

    const values: Float32Array = gradient.toFloat32Array();

    for (let index: number = 0; index < values.length; index += 1) {
        const value: number | undefined = values[index];

        if (value === undefined) {
            throw new NonFiniteGradientError(`${label} value at flat offset ${index} does not exist.`);
        }

        if (!Number.isFinite(value)) {
            throw new NonFiniteGradientError(`${label} contains non-finite value ${value} at flat offset ${index}.`);
        }
    }
}

/** Ensures all registered model parameters and gradients are finite. */
export function assertFiniteModel(model: Module): void {
    const parameters: readonly Parameter[] = model.parameters();

    for (let index: number = 0; index < parameters.length; index += 1) {
        const parameter: Parameter | undefined = parameters[index];

        if (parameter === undefined) {
            throw new NonFiniteTensorError(`Model parameter at index ${index} does not exist.`);
        }

        assertFiniteTensor(parameter, `Model parameter ${index}`);
        assertFiniteGradient(parameter, `Model parameter ${index} gradient`);
    }
}
