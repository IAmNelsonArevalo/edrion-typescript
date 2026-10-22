import { Module } from '@core/nn/module';
import { Tensor } from '@core/tensor/tensor';

/**
 * Rectified Linear Unit activation.
 */
export class ReLU extends Module {
    public override forward(input: Tensor): Tensor {
        return input.relu();
    }
}
