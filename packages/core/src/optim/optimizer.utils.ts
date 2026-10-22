import type { Parameter } from '@core/nn/parameter';
import { OptimizerConfigurationError, OptimizerError } from '@core/optim/optimizer.errors';
import type { Tensor } from '@core/tensor/tensor';

/** Validates a finite positive optimizer value. */
export function assertPositiveFinite(value: number, name: string): void {
    if (!Number.isFinite(value) || value <= 0) {
        throw new OptimizerConfigurationError(`${name} must be a finite number greater than zero. Received ${value}.`);
    }
}

/** Validates a finite value in the inclusive range [0, 1). */
export function assertUnitIntervalExclusiveOne(value: number, name: string): void {
    if (!Number.isFinite(value) || value < 0 || value >= 1) {
        throw new OptimizerConfigurationError(
            `${name} must be a finite number in the range [0, 1). Received ${value}.`,
        );
    }
}

/** Reads and validates the current gradient for a parameter. */
export function getParameterGradient(parameter: Parameter): Float32Array | undefined {
    const gradient: Tensor | undefined = parameter.grad;

    if (gradient === undefined) {
        return undefined;
    }

    if (gradient.size !== parameter.size) {
        throw new OptimizerError(
            `Parameter gradient size ${gradient.size} does not match parameter size ${parameter.size}.`,
        );
    }

    return gradient.toFloat32Array();
}
