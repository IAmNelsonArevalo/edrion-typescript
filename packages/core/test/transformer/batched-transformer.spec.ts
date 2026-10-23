import { describe, expect, it } from 'vitest';
import { Tensor } from '@core/tensor';
import { MultiHeadAttention, TransformerModel } from '@core/nn/transformer';

describe('FEATURE-011 batched transformer', (): void => {
    it('runs multi-head attention over [batch, sequence, embedding]', (): void => {
        const attention: MultiHeadAttention = new MultiHeadAttention(4, 2);
        const input: Tensor = new Tensor(
            new Float32Array([
                1, 0, 0, 1,
                0, 1, 1, 0,
                1, 1, 0, 0,
                0, 0, 1, 1,
            ]),
            [2, 2, 4],
            { requiresGrad: true },
        );

        const output: Tensor = attention.forward(input);

        expect(output.shape).toEqual([2, 2, 4]);
        output.sum().backward();
        expect(input.grad?.shape).toEqual([2, 2, 4]);
    });

    it('produces batched vocabulary logits', (): void => {
        const model: TransformerModel = new TransformerModel({
            vocabularySize: 16,
            contextLength: 4,
            embeddingDimension: 4,
            numberOfLayers: 1,
            numberOfHeads: 2,
            feedForwardDimension: 8,
        });
        const tokens: Tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);

        const logits: Tensor = model.forward(tokens);

        expect(logits.shape).toEqual([2, 3, 16]);
    });
});
