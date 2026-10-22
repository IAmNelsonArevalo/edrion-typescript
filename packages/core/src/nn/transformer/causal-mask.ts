import { Tensor } from '@core/tensor/tensor';
import { TransformerConfigurationError } from '@core/nn/transformer/transformer.errors';

/**
 * Large negative value used by additive causal attention masks.
 */
export const CAUSAL_MASK_VALUE: number = -1_000_000_000;

/**
 * Creates an additive causal mask with shape [sequenceLength, sequenceLength].
 * Visible positions contain 0 while future positions contain a large negative value.
 */
export function createCausalMask(sequenceLength: number, backend: string = 'cpu'): Tensor {
    if (!Number.isInteger(sequenceLength) || sequenceLength <= 0) {
        throw new TransformerConfigurationError(
            `sequenceLength must be a positive integer. Received ${sequenceLength}.`,
        );
    }

    const values: Float32Array = new Float32Array(sequenceLength * sequenceLength);

    for (let row: number = 0; row < sequenceLength; row += 1) {
        for (let column: number = 0; column < sequenceLength; column += 1) {
            if (column > row) {
                values[row * sequenceLength + column] = CAUSAL_MASK_VALUE;
            }
        }
    }

    return new Tensor(values, [sequenceLength, sequenceLength], { backend });
}
