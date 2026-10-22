import { Module } from '@core/nn/module';
import { createDetachedAxisMax } from '@core/stability/stable-reductions';
import { Tensor } from '@core/tensor/tensor';

/** Softmax activation along one tensor axis using max-subtraction stability. */
export class Softmax extends Module {
    public constructor(public readonly axis: number = -1) {
        super();
    }

    public override forward(input: Tensor): Tensor {
        const maxima: Tensor = createDetachedAxisMax(input, this.axis);
        const shifted: Tensor = input.sub(maxima);
        const exponentials: Tensor = shifted.exp();
        const denominator: Tensor = exponentials.sum(this.axis, true);

        return exponentials.div(denominator);
    }
}
