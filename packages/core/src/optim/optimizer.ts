import type { Parameter } from '@core/nn/parameter';
import { OptimizerError } from '@core/optim/optimizer.errors';
import { assertPositiveFinite } from '@core/optim/optimizer.utils';

/**
 * Base optimizer for trainable Edrion parameters.
 */
export abstract class Optimizer {
    protected readonly parametersInternal: readonly Parameter[];
    protected learningRateInternal: number;

    protected constructor(parameters: readonly Parameter[], learningRate: number) {
        this.parametersInternal = [...parameters];
        assertPositiveFinite(learningRate, 'Optimizer learningRate');
        this.learningRateInternal = learningRate;
    }

    /** Returns the parameters managed by this optimizer. */
    public get parameters(): readonly Parameter[] {
        return [...this.parametersInternal];
    }

    /** Returns the current learning rate. */
    public get learningRate(): number {
        return this.learningRateInternal;
    }

    /** Updates the learning rate used by subsequent optimization steps. */
    public setLearningRate(learningRate: number): void {
        assertPositiveFinite(learningRate, 'Optimizer learningRate');
        this.learningRateInternal = learningRate;
    }

    /** Applies one optimization step. */
    public abstract step(): void;

    /** Clears accumulated gradients on every managed parameter. */
    public zeroGrad(): void {
        for (let index: number = 0; index < this.parametersInternal.length; index += 1) {
            const parameter: Parameter | undefined = this.parametersInternal[index];

            if (parameter === undefined) {
                throw new OptimizerError(`Missing optimizer parameter at index ${index}.`);
            }

            parameter.zeroGrad();
        }
    }
}
