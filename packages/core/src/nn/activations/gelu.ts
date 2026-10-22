import { Module } from '@core/nn/module';
import { Tensor } from '@core/tensor/tensor';

/**
 * Gaussian Error Linear Unit using the standard tanh approximation.
 */
export class GELU extends Module {
    public override forward(input: Tensor): Tensor {
        const cubic: Tensor = input.pow(3).mul(0.044715);
        const inner: Tensor = input.add(cubic).mul(Math.sqrt(2 / Math.PI));

        const sigmoid: Tensor = inner.mul(2).neg().exp().add(1).pow(-1);

        const tanh: Tensor = sigmoid.mul(2).sub(1);

        return input.mul(0.5).mul(tanh.add(1));
    }
}
