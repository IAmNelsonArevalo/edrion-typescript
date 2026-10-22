import { describe, expect, it } from 'vitest';
import { Tensor, TransformerConfigurationError, TransformerModel } from '@core';

describe('TransformerModel', (): void => {
    it('should produce vocabulary logits', (): void => {
        const model: TransformerModel = new TransformerModel({
            vocabularySize: 8,
            contextLength: 6,
            embeddingDimension: 4,
            numberOfLayers: 2,
            numberOfHeads: 2,
            feedForwardDimension: 8,
        });
        const tokens: Tensor = new Tensor([1, 2, 3], [3]);
        const logits: Tensor = model.forward(tokens);
        expect(logits.shape).toEqual([3, 8]);
        expect(model.blocks.length).toBe(2);
        expect(model.parameters().length).toBeGreaterThan(0);
    });

    it('should reject incompatible head configuration', (): void => {
        expect((): void => {
            new TransformerModel({
                vocabularySize: 8,
                contextLength: 6,
                embeddingDimension: 5,
                numberOfLayers: 1,
                numberOfHeads: 2,
                feedForwardDimension: 10,
            });
        }).toThrow(TransformerConfigurationError);
    });
});
