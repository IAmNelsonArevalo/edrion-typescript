import type { Parameter } from '@core/nn/parameter';
import { GradientClippingError, NonFiniteGradientError } from '@core/stability/stability.errors';

/** Result produced by global-norm gradient clipping. */
export interface GradientClippingResult {
    readonly totalNorm: number;
    readonly maxNorm: number;
    readonly clipCoefficient: number;
    readonly clipped: boolean;
}

/** Computes the L2 norm across all available parameter gradients. */
export function computeGlobalGradientNorm(parameters: readonly Parameter[]): number {
    let squaredNorm: number = 0;

    for (let parameterIndex: number = 0; parameterIndex < parameters.length; parameterIndex += 1) {
        const parameter: Parameter | undefined = parameters[parameterIndex];

        if (parameter === undefined) {
            throw new GradientClippingError(`Missing parameter at index ${parameterIndex}.`);
        }

        const gradient = parameter.grad;

        if (gradient === undefined) {
            continue;
        }

        const values: Float32Array = gradient.toFloat32Array();

        for (let valueIndex: number = 0; valueIndex < values.length; valueIndex += 1) {
            const value: number | undefined = values[valueIndex];

            if (value === undefined) {
                throw new NonFiniteGradientError(
                    `Missing gradient value at parameter ${parameterIndex}, offset ${valueIndex}.`,
                );
            }

            if (!Number.isFinite(value)) {
                throw new NonFiniteGradientError(
                    `Gradient contains non-finite value ${value} at parameter ${parameterIndex}, offset ${valueIndex}.`,
                );
            }

            squaredNorm += value * value;
        }
    }

    const totalNorm: number = Math.sqrt(squaredNorm);

    if (!Number.isFinite(totalNorm)) {
        throw new NonFiniteGradientError(`Global gradient norm is not finite. Received ${totalNorm}.`);
    }

    return totalNorm;
}

/** Clips parameter gradients using one shared global L2 norm. */
export function clipGradientsByGlobalNorm(parameters: readonly Parameter[], maxNorm: number): GradientClippingResult {
    if (!Number.isFinite(maxNorm) || maxNorm <= 0) {
        throw new GradientClippingError(`maxNorm must be a finite number greater than zero. Received ${maxNorm}.`);
    }

    const totalNorm: number = computeGlobalGradientNorm(parameters);

    if (totalNorm <= maxNorm || totalNorm === 0) {
        return {
            totalNorm,
            maxNorm,
            clipCoefficient: 1,
            clipped: false,
        };
    }

    const clipCoefficient: number = maxNorm / totalNorm;

    for (let index: number = 0; index < parameters.length; index += 1) {
        const parameter: Parameter | undefined = parameters[index];

        if (parameter === undefined) {
            throw new GradientClippingError(`Missing parameter at index ${index}.`);
        }

        parameter.scaleGradient(clipCoefficient);
    }

    return {
        totalNorm,
        maxNorm,
        clipCoefficient,
        clipped: true,
    };
}
