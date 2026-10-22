import { Softmax } from '@core/nn/activations/softmax';
import { AttentionShapeError } from '@core/nn/transformer/transformer.errors';
import { Tensor } from '@core/tensor/tensor';

/**
 * Scaled dot-product attention for one sequence or a batch of sequences.
 *
 * Supported shapes:
 * - [queryLength, headDimension]
 * - [batch, queryLength, headDimension]
 */
export class ScaledDotProductAttention {
    private readonly softmax: Softmax = new Softmax(-1);

    public forward(query: Tensor, key: Tensor, value: Tensor, mask?: Tensor): Tensor {
        assertAttentionInputs(query, key, value);

        const queryShape: readonly number[] = query.shape;
        const keyShape: readonly number[] = key.shape;
        const headDimension: number = readDimension(queryShape, query.rank - 1, 'query head dimension');
        const queryLength: number = readDimension(queryShape, query.rank - 2, 'query sequence length');
        const keyLength: number = readDimension(keyShape, key.rank - 2, 'key sequence length');
        let scores: Tensor = query.matmul(key.transpose()).div(Math.sqrt(headDimension));

        if (mask !== undefined) {
            if (mask.rank !== 2 && mask.rank !== query.rank) {
                throw new AttentionShapeError(
                    `Attention mask must have rank 2 or ${query.rank}. Received rank ${mask.rank}.`,
                );
            }

            const maskShape: readonly number[] = mask.shape;
            const maskRows: number = readDimension(maskShape, mask.rank - 2, 'mask row dimension');
            const maskColumns: number = readDimension(maskShape, mask.rank - 1, 'mask column dimension');

            if (maskRows !== queryLength || maskColumns !== keyLength) {
                throw new AttentionShapeError(
                    `Attention mask final dimensions [${maskRows}, ${maskColumns}] must equal [${queryLength}, ${keyLength}].`,
                );
            }

            scores = scores.add(mask);
        }

        const probabilities: Tensor = this.softmax.forward(scores);
        return probabilities.matmul(value);
    }
}

function assertAttentionInputs(query: Tensor, key: Tensor, value: Tensor): void {
    if (query.rank !== key.rank || query.rank !== value.rank || (query.rank !== 2 && query.rank !== 3)) {
        throw new AttentionShapeError(
            `Scaled dot-product attention expects query, key and value with matching rank 2 or 3. Received ${query.rank}, ${key.rank}, ${value.rank}.`,
        );
    }

    const queryShape: readonly number[] = query.shape;
    const keyShape: readonly number[] = key.shape;
    const valueShape: readonly number[] = value.shape;
    const queryDimension: number = readDimension(queryShape, query.rank - 1, 'query dimension');
    const keyDimension: number = readDimension(keyShape, key.rank - 1, 'key dimension');
    const keyLength: number = readDimension(keyShape, key.rank - 2, 'key length');
    const valueLength: number = readDimension(valueShape, value.rank - 2, 'value length');

    if (queryDimension !== keyDimension) {
        throw new AttentionShapeError(`Query dimension ${queryDimension} must equal key dimension ${keyDimension}.`);
    }

    if (keyLength !== valueLength) {
        throw new AttentionShapeError(
            `Key sequence length ${keyLength} must equal value sequence length ${valueLength}.`,
        );
    }

    if (query.rank === 3) {
        const queryBatch: number = readDimension(queryShape, 0, 'query batch');
        const keyBatch: number = readDimension(keyShape, 0, 'key batch');
        const valueBatch: number = readDimension(valueShape, 0, 'value batch');

        if (queryBatch !== keyBatch || queryBatch !== valueBatch) {
            throw new AttentionShapeError(
                `Attention batch dimensions must match. Received ${queryBatch}, ${keyBatch}, ${valueBatch}.`,
            );
        }
    }

    if (query.backend !== key.backend || query.backend !== value.backend) {
        throw new AttentionShapeError('Query, key and value must use the same backend.');
    }
}

function readDimension(shape: readonly number[], axis: number, label: string): number {
    const value: number | undefined = shape[axis];

    if (value === undefined) {
        throw new AttentionShapeError(`Unable to resolve ${label}.`);
    }

    return value;
}
