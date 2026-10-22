import type { Parameter } from '@core/nn/parameter';
import { OptimizerError } from '@core/optim/optimizer.errors';

/**
 * Base optimizer for trainable Edrion parameters.
 */
export abstract class Optimizer {
    protected readonly parametersInternal: readonly Parameter[];

    protected constructor(parameters: readonly Parameter[]) {
        this.parametersInternal = [...parameters];
    }

    /**
     * Returns the parameters managed by this optimizer.
     */
    public get parameters(): readonly Parameter[] {
        return [...this.parametersInternal];
    }

    /**
     * Applies one optimization step.
     */
    public abstract step(): void;

    /**
     * Clears accumulated gradients on every managed parameter.
     */
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
