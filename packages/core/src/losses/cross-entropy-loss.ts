import { Loss } from '@core/losses/loss';
import { LossShapeError, LossTargetError } from '@core/losses/loss.errors';
import type { TensorShape } from '@core/tensor/shape';
import { tensorShapesEqual } from '@core/tensor/shape';
import { createDetachedAxisMax } from '@core/stability/stable-reductions';
import { assertFiniteTensor } from '@core/stability/numerical-validation';
import { Tensor } from '@core/tensor/tensor';

/**
 * Cross entropy loss over logits using integer class targets.
 *
 * The final logits axis is interpreted as the class dimension.
 * Target shape must equal the logits shape without that final axis.
 *
 * Examples:
 * - logits [classes] -> target []
 * - logits [batch, classes] -> target [batch]
 * - logits [batch, sequence, classes] -> target [batch, sequence]
 *
 * Uses max-subtracted log-sum-exp for numerical stability.
 */
export class CrossEntropyLoss extends Loss {
    public override forward(logits: Tensor, target: Tensor): Tensor {
        if (logits.rank === 0) {
            throw new LossShapeError('CrossEntropyLoss requires logits with at least one dimension.');
        }

        const logitsShape: TensorShape = logits.shape;
        const classCount: number = readClassCount(logitsShape);
        const expectedTargetShape: TensorShape = logitsShape.slice(0, -1);

        if (!tensorShapesEqual(target.shape, expectedTargetShape)) {
            throw new LossShapeError(
                `CrossEntropyLoss expected target shape [${expectedTargetShape.join(', ')}] for logits shape [${logitsShape.join(', ')}], but received [${target.shape.join(', ')}].`,
            );
        }

        if (target.requiresGrad) {
            throw new LossTargetError('CrossEntropyLoss class targets must not require gradients.');
        }

        assertFiniteTensor(logits, 'Cross entropy logits');

        const oneHot: Tensor = createOneHotTarget(target, logitsShape, classCount, logits.backend);

        const maxima: Tensor = createDetachedAxisMax(logits, -1);
        const shifted: Tensor = logits.sub(maxima);
        const exponentials: Tensor = shifted.exp();
        const logSumExp: Tensor = exponentials.sum(-1, true).log().add(maxima);
        const logProbabilities: Tensor = logits.sub(logSumExp);

        return logProbabilities.mul(oneHot).sum(-1).neg().mean();
    }
}

function readClassCount(shape: TensorShape): number {
    const classAxis: number = shape.length - 1;
    const classCount: number | undefined = shape[classAxis];

    if (classCount === undefined) {
        throw new LossShapeError(`Unable to read class dimension at axis ${classAxis}.`);
    }

    return classCount;
}

function createOneHotTarget(target: Tensor, logitsShape: TensorShape, classCount: number, backend: string): Tensor {
    const targetValues: number[] = target.toFlatArray();
    const oneHotData: Float32Array = new Float32Array(targetValues.length * classCount);

    for (let sampleIndex: number = 0; sampleIndex < targetValues.length; sampleIndex += 1) {
        const classIndex: number | undefined = targetValues[sampleIndex];

        if (classIndex === undefined) {
            throw new LossTargetError(`Missing cross entropy target at flat index ${sampleIndex}.`);
        }

        if (!Number.isInteger(classIndex)) {
            throw new LossTargetError(
                `Cross entropy target at flat index ${sampleIndex} must be an integer class index. Received ${classIndex}.`,
            );
        }

        if (classIndex < 0 || classIndex >= classCount) {
            throw new LossTargetError(
                `Cross entropy target class ${classIndex} at flat index ${sampleIndex} is outside the valid range 0-${classCount - 1}.`,
            );
        }

        const offset: number = sampleIndex * classCount + classIndex;
        oneHotData[offset] = 1;
    }

    return new Tensor(oneHotData, logitsShape, {
        backend,
        requiresGrad: false,
    });
}
