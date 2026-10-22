import { Softmax } from '@core/nn/activations/softmax';
import { AttentionShapeError } from '@core/nn/transformer/transformer.errors';
import { Tensor } from '@core/tensor/tensor';

/**
 * Scaled dot-product attention operating on rank-2 tensors.
 *
 * Shapes:
 * query: [queryLength, headDimension]
 * key:   [keyLength, headDimension]
 * value: [keyLength, valueDimension]
 */
export class ScaledDotProductAttention {
    private readonly softmax: Softmax = new Softmax(-1);

    public forward(query: Tensor, key: Tensor, value: Tensor, mask?: Tensor): Tensor {
        assertAttentionInputs(query, key, value);

        const queryShape: readonly number[] = query.shape;
        const keyShape: readonly number[] = key.shape;
        const headDimension: number | undefined = queryShape[1];
        const keyLength: number | undefined = keyShape[0];
        const queryLength: number | undefined = queryShape[0];

        if (headDimension === undefined || keyLength === undefined || queryLength === undefined) {
            throw new AttentionShapeError('Unable to resolve attention dimensions.');
        }

        let scores: Tensor = query.matmul(key.transpose()).div(Math.sqrt(headDimension));

        if (mask !== undefined) {
            if (mask.rank !== 2) {
                throw new AttentionShapeError(`Attention mask must have rank 2. Received rank ${mask.rank}.`);
            }

            const maskShape: readonly number[] = mask.shape;
            const maskRows: number | undefined = maskShape[0];
            const maskColumns: number | undefined = maskShape[1];

            if (maskRows !== queryLength || maskColumns !== keyLength) {
                throw new AttentionShapeError(
                    `Attention mask shape [${maskShape.join(', ')}] must equal [${queryLength}, ${keyLength}].`,
                );
            }

            scores = scores.add(mask);
        }

        const probabilities: Tensor = this.softmax.forward(scores);
        return probabilities.matmul(value);
    }
}

function assertAttentionInputs(query: Tensor, key: Tensor, value: Tensor): void {
    if (query.rank !== 2 || key.rank !== 2 || value.rank !== 2) {
        throw new AttentionShapeError(
            `Scaled dot-product attention expects rank-2 query, key and value tensors. Received ${query.rank}, ${key.rank}, ${value.rank}.`,
        );
    }

    const queryShape: readonly number[] = query.shape;
    const keyShape: readonly number[] = key.shape;
    const valueShape: readonly number[] = value.shape;
    const queryDimension: number | undefined = queryShape[1];
    const keyDimension: number | undefined = keyShape[1];
    const keyLength: number | undefined = keyShape[0];
    const valueLength: number | undefined = valueShape[0];

    if (
        queryDimension === undefined ||
        keyDimension === undefined ||
        keyLength === undefined ||
        valueLength === undefined
    ) {
        throw new AttentionShapeError('Unable to resolve attention input dimensions.');
    }

    if (queryDimension !== keyDimension) {
        throw new AttentionShapeError(`Query dimension ${queryDimension} must equal key dimension ${keyDimension}.`);
    }

    if (keyLength !== valueLength) {
        throw new AttentionShapeError(
            `Key sequence length ${keyLength} must equal value sequence length ${valueLength}.`,
        );
    }

    if (query.backend !== key.backend || query.backend !== value.backend) {
        throw new AttentionShapeError('Query, key and value must use the same backend.');
    }
}
