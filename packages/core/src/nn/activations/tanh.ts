import { Module } from '@core/nn/module';
import { Tensor } from '@core/tensor/tensor';

/**
 * Hyperbolic tangent activation.
 */
export class Tanh extends Module {
    public override forward(input: Tensor): Tensor {
        const sigmoid: Tensor = input.mul(2).neg().exp().add(1).pow(-1);

        return sigmoid.mul(2).sub(1);
    }
}
