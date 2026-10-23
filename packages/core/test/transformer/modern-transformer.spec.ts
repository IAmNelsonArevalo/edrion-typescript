import { describe, expect, it } from 'vitest';
import { RMSNormalization, SwiGLUFeedForward, Tensor, TransformerModel } from '@core';

describe('FEATURE-014 modern Transformer architecture', (): void => {
    it('should use RoPE, RMSNorm and SwiGLU by default', (): void => {
        const model: TransformerModel = new TransformerModel({
            vocabularySize: 16,
            contextLength: 8,
            embeddingDimension: 8,
            numberOfLayers: 2,
            numberOfHeads: 2,
            feedForwardDimension: 16,
            architecture: 'modern',
        });
        expect(model.config.architecture).toBe('modern');
        expect(model.positionEmbedding).toBeUndefined();
        expect(model.finalNormalization).toBeInstanceOf(RMSNormalization);
        expect(model.blocks[0]?.feedForward).toBeInstanceOf(SwiGLUFeedForward);
        expect(model.blocks[0]?.attention.rotaryEmbedding).toBe(true);

        const tokens: Tensor = new Tensor([1, 2, 3, 4, 4, 3, 2, 1], [2, 4]);
        const logits: Tensor = model.forward(tokens);
        expect(logits.shape).toEqual([2, 4, 16]);
        logits.mean().backward();
        expect(model.tokenEmbedding.weight.grad).toBeDefined();
    });

    it('should preserve the legacy architecture when explicitly requested', (): void => {
        const model: TransformerModel = new TransformerModel({
            vocabularySize: 16,
            contextLength: 8,
            embeddingDimension: 8,
            numberOfLayers: 1,
            numberOfHeads: 2,
            feedForwardDimension: 16,
            architecture: 'legacy',
        });
        expect(model.positionEmbedding).toBeDefined();
        expect(model.blocks[0]?.attention.rotaryEmbedding).toBe(false);
        expect(model.forward(new Tensor([1, 2, 3], [3])).shape).toEqual([3, 16]);
    });
});
