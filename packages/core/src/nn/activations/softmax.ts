import { Module } from '@core/nn/module';
import { Tensor } from '@core/tensor/tensor';

/**
 * Softmax activation along one tensor axis.
 *
 * Numerical stabilization with max-subtraction is introduced
 * in FEATURE-009 together with the numerical-safety layer.
 */
export class Softmax extends Module {
    public constructor(public readonly axis: number = -1) {
        super();
    }

    public override forward(input: Tensor): Tensor {
        const exponentials: Tensor = input.exp();
        const denominator: Tensor = exponentials.sum(this.axis, true);

        return exponentials.div(denominator);
    }
}
