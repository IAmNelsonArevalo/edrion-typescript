import { Module } from '@core/nn/module';
import { Tensor } from '@core/tensor/tensor';

/**
 * Logistic sigmoid activation.
 */
export class Sigmoid extends Module {
    public override forward(input: Tensor): Tensor {
        return input.neg().exp().add(1).pow(-1);
    }
}
