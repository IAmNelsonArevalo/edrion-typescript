import type { Tensor } from '@core/tensor/tensor';

/**
 * Base contract for differentiable loss functions.
 */
export abstract class Loss {
    /**
     * Computes a scalar loss tensor from predictions and targets.
     */
    public abstract forward(prediction: Tensor, target: Tensor): Tensor;
}
