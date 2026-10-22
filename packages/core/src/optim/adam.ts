import type { Parameter } from '@core/nn/parameter';
import { Optimizer } from '@core/optim/optimizer';
import { OptimizerError } from '@core/optim/optimizer.errors';
import {
    assertPositiveFinite,
    assertUnitIntervalExclusiveOne,
    getParameterGradient,
} from '@core/optim/optimizer.utils';

/** Adam optimizer configuration. */
export interface AdamOptions {
    readonly learningRate: number;
    readonly beta1?: number;
    readonly beta2?: number;
    readonly epsilon?: number;
}

interface AdamParameterState {
    readonly firstMoment: Float32Array;
    readonly secondMoment: Float32Array;
}

/**
 * Adam optimizer with bias-corrected first and second moments.
 */
export class Adam extends Optimizer {
    public readonly learningRate: number;
    public readonly beta1: number;
    public readonly beta2: number;
    public readonly epsilon: number;

    private readonly states: Map<Parameter, AdamParameterState> = new Map<Parameter, AdamParameterState>();

    private stepCount: number = 0;

    public constructor(parameters: readonly Parameter[], options: AdamOptions) {
        super(parameters);

        assertPositiveFinite(options.learningRate, 'Adam learningRate');

        const beta1: number = options.beta1 ?? 0.9;
        const beta2: number = options.beta2 ?? 0.999;
        const epsilon: number = options.epsilon ?? 1e-8;

        assertUnitIntervalExclusiveOne(beta1, 'Adam beta1');
        assertUnitIntervalExclusiveOne(beta2, 'Adam beta2');
        assertPositiveFinite(epsilon, 'Adam epsilon');

        this.learningRate = options.learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    /** Returns the number of completed Adam steps. */
    public get steps(): number {
        return this.stepCount;
    }

    /**
     * Applies one Adam optimization step.
     */
    public override step(): void {
        this.stepCount += 1;

        const firstBiasCorrection: number = 1 - Math.pow(this.beta1, this.stepCount);
        const secondBiasCorrection: number = 1 - Math.pow(this.beta2, this.stepCount);

        for (let parameterIndex: number = 0; parameterIndex < this.parametersInternal.length; parameterIndex += 1) {
            const parameter: Parameter | undefined = this.parametersInternal[parameterIndex];

            if (parameter === undefined) {
                throw new OptimizerError(`Missing Adam parameter at index ${parameterIndex}.`);
            }

            const gradient: Float32Array | undefined = getParameterGradient(parameter);

            if (gradient === undefined) {
                continue;
            }

            const values: Float32Array = parameter.toFloat32Array();
            const updatedValues: Float32Array = new Float32Array(parameter.size);

            const state: AdamParameterState = this.getState(parameter);

            for (let valueIndex: number = 0; valueIndex < parameter.size; valueIndex += 1) {
                const currentValue: number | undefined = values[valueIndex];
                const gradientValue: number | undefined = gradient[valueIndex];
                const previousFirstMoment: number | undefined = state.firstMoment[valueIndex];
                const previousSecondMoment: number | undefined = state.secondMoment[valueIndex];

                if (
                    currentValue === undefined ||
                    gradientValue === undefined ||
                    previousFirstMoment === undefined ||
                    previousSecondMoment === undefined
                ) {
                    throw new OptimizerError(
                        `Unable to resolve Adam data at parameter index ${parameterIndex}, value index ${valueIndex}.`,
                    );
                }

                const firstMoment: number = this.beta1 * previousFirstMoment + (1 - this.beta1) * gradientValue;

                const secondMoment: number =
                    this.beta2 * previousSecondMoment + (1 - this.beta2) * gradientValue * gradientValue;

                state.firstMoment[valueIndex] = firstMoment;
                state.secondMoment[valueIndex] = secondMoment;

                const correctedFirstMoment: number = firstMoment / firstBiasCorrection;
                const correctedSecondMoment: number = secondMoment / secondBiasCorrection;

                const denominator: number = Math.sqrt(correctedSecondMoment) + this.epsilon;

                updatedValues[valueIndex] = currentValue - (this.learningRate * correctedFirstMoment) / denominator;
            }

            parameter.assign(updatedValues);
        }
    }

    private getState(parameter: Parameter): AdamParameterState {
        const existingState: AdamParameterState | undefined = this.states.get(parameter);

        if (existingState !== undefined) {
            return existingState;
        }

        const state: AdamParameterState = {
            firstMoment: new Float32Array(parameter.size),
            secondMoment: new Float32Array(parameter.size),
        };

        this.states.set(parameter, state);
        return state;
    }
}
