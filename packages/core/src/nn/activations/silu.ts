import { Module } from '@core/nn/module';
import { Tensor } from '@core/tensor/tensor';

/** SiLU / Swish activation: x * sigmoid(x). */
export class SiLU extends Module {
    public override forward(input: Tensor): Tensor {
        const sigmoid: Tensor = input.neg().exp().add(1).pow(-1);
        return input.mul(sigmoid);
    }
}
