import type { Parameter } from '@core/nn/parameter';
import { Optimizer } from '@core/optim/optimizer';
import { OptimizerError } from '@core/optim/optimizer.errors';
import {
    assertPositiveFinite,
    assertUnitIntervalExclusiveOne,
    getParameterGradient,
} from '@core/optim/optimizer.utils';

/**
 * Stochastic gradient descent configuration.
 */
export interface SGDOptions {
    readonly learningRate: number;
    readonly momentum?: number;
}

/**
 * Stochastic gradient descent optimizer with optional momentum.
 */
export class SGD extends Optimizer {
    public readonly momentum: number;

    private readonly velocity: Map<Parameter, Float32Array> = new Map<Parameter, Float32Array>();

    public constructor(parameters: readonly Parameter[], options: SGDOptions) {
        super(parameters, options.learningRate);

        assertPositiveFinite(options.learningRate, 'SGD learningRate');

        const momentum: number = options.momentum ?? 0;
        assertUnitIntervalExclusiveOne(momentum, 'SGD momentum');

        this.momentum = momentum;
    }

    /**
     * Applies one SGD optimization step.
     */
    public override step(): void {
        for (let parameterIndex: number = 0; parameterIndex < this.parametersInternal.length; parameterIndex += 1) {
            const parameter: Parameter | undefined = this.parametersInternal[parameterIndex];

            if (parameter === undefined) {
                throw new OptimizerError(`Missing SGD parameter at index ${parameterIndex}.`);
            }

            const gradient: Float32Array | undefined = getParameterGradient(parameter);

            if (gradient === undefined) {
                continue;
            }

            const values: Float32Array = parameter.toFloat32Array();
            const updatedValues: Float32Array = new Float32Array(parameter.size);

            let velocity: Float32Array | undefined = this.velocity.get(parameter);

            if (this.momentum > 0 && velocity === undefined) {
                velocity = new Float32Array(parameter.size);
                this.velocity.set(parameter, velocity);
            }

            for (let valueIndex: number = 0; valueIndex < parameter.size; valueIndex += 1) {
                const currentValue: number | undefined = values[valueIndex];
                const gradientValue: number | undefined = gradient[valueIndex];

                if (currentValue === undefined || gradientValue === undefined) {
                    throw new OptimizerError(
                        `Unable to resolve SGD data at parameter index ${parameterIndex}, value index ${valueIndex}.`,
                    );
                }

                let update: number = gradientValue;

                if (velocity !== undefined) {
                    const previousVelocity: number | undefined = velocity[valueIndex];

                    if (previousVelocity === undefined) {
                        throw new OptimizerError(`Missing SGD velocity at value index ${valueIndex}.`);
                    }

                    const nextVelocity: number = this.momentum * previousVelocity + gradientValue;

                    velocity[valueIndex] = nextVelocity;
                    update = nextVelocity;
                }

                updatedValues[valueIndex] = currentValue - this.learningRate * update;
            }

            parameter.assign(updatedValues);
        }
    }
}
