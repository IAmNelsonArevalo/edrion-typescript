import { describe, expect, it } from 'vitest';
import { Tensor, TransformerModel } from '@core';

describe('FEATURE-016 KV cache', (): void => {
    it('should match full-prefix logits during incremental inference', (): void => {
        const model: TransformerModel = new TransformerModel({
            vocabularySize: 16,
            contextLength: 8,
            embeddingDimension: 8,
            numberOfLayers: 2,
            numberOfHeads: 2,
            feedForwardDimension: 16,
            architecture: 'modern',
        });
        model.eval();
        const ids: readonly number[] = [1, 3, 5, 7];
        const cache = model.createKeyValueCache();

        for (let index: number = 0; index < ids.length; index += 1) {
            const tokenId: number | undefined = ids[index];
            if (tokenId === undefined) {
                throw new Error(`Missing token ${index}.`);
            }
            const incremental = model.forwardIncremental(new Tensor([tokenId], [1]), cache).logits;
            const prefix = new Tensor(ids.slice(0, index + 1), [index + 1]);
            const full = model.forward(prefix);
            for (let vocabularyIndex: number = 0; vocabularyIndex < 16; vocabularyIndex += 1) {
                expect(incremental.get(0, vocabularyIndex)).toBeCloseTo(full.get(index, vocabularyIndex), 4);
            }
        }

        expect(cache.sequenceLength).toBe(ids.length);
        expect(cache.getLayer(0)?.key.shape).toEqual([ids.length, 8]);
        expect(cache.getLayer(0)?.value.shape).toEqual([ids.length, 8]);
    });

    it('should enforce context length and support clearing', (): void => {
        const model: TransformerModel = new TransformerModel({
            vocabularySize: 8,
            contextLength: 2,
            embeddingDimension: 4,
            numberOfLayers: 1,
            numberOfHeads: 1,
            feedForwardDimension: 8,
            architecture: 'modern',
        });
        const cache = model.createKeyValueCache();
        model.forwardIncremental(new Tensor([1], [1]), cache);
        model.forwardIncremental(new Tensor([2], [1]), cache);
        expect((): void => {
            model.forwardIncremental(new Tensor([3], [1]), cache);
        }).toThrow();
        cache.clear();
        expect(cache.sequenceLength).toBe(0);
        expect(cache.getLayer(0)).toBeUndefined();
    });
});
