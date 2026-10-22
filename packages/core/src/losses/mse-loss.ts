import { Loss } from '@core/losses/loss';
import { LossShapeError } from '@core/losses/loss.errors';
import { tensorShapesEqual } from '@core/tensor/shape';
import { Tensor } from '@core/tensor/tensor';

/**
 * Mean squared error loss.
 *
 * Computes mean((prediction - target)^2).
 */
export class MSELoss extends Loss {
    public override forward(prediction: Tensor, target: Tensor): Tensor {
        if (!tensorShapesEqual(prediction.shape, target.shape)) {
            throw new LossShapeError(
                `MSELoss requires equal shapes. Received prediction [${prediction.shape.join(', ')}] and target [${target.shape.join(', ')}].`,
            );
        }

        return prediction.sub(target).pow(2).mean();
    }
}
