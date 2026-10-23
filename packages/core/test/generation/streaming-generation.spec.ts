import { describe, expect, it } from 'vitest';
import { TextGenerator, TransformerModel, type Tokenizer } from '@core';

class TestTokenizer implements Tokenizer {
    public readonly vocabularySize: number = 8;
    public readonly padTokenId: number = 0;
    public readonly unknownTokenId: number = 1;
    public readonly beginningOfSequenceTokenId: number = 2;
    public readonly endOfSequenceTokenId: number = 3;

    public encode(text: string): readonly number[] {
        return text.length === 0 ? [] : [4, 5];
    }

    public decode(tokenIds: readonly number[]): string {
        return tokenIds.join(' ');
    }
}

describe('FEATURE-016 streaming generation', (): void => {
    it('should stream token events and report KV-cache inference metrics', (): void => {
        const model: TransformerModel = new TransformerModel({
            vocabularySize: 8,
            contextLength: 8,
            embeddingDimension: 8,
            numberOfLayers: 1,
            numberOfHeads: 2,
            feedForwardDimension: 16,
            architecture: 'modern',
        });
        model.eval();
        const generator: TextGenerator = new TextGenerator(model, new TestTokenizer());
        const iterator = generator.stream('hello', {
            maxNewTokens: 3,
            temperature: 0,
            stopOnEndOfSequence: false,
        });
        const events: number[] = [];
        let item = iterator.next();
        while (!item.done) {
            events.push(item.value.tokenId);
            item = iterator.next();
        }

        expect(events).toHaveLength(3);
        expect(item.value.generatedTokenIds).toHaveLength(3);
        expect(item.value.metrics.cacheEnabled).toBe(true);
        expect(item.value.metrics.generatedTokenCount).toBe(3);
        expect(item.value.metrics.modelForwardCalls).toBe(4);
        expect(item.value.metrics.tokensPerSecond).toBeGreaterThan(0);
    });

    it('should preserve full-context generation for legacy architecture', (): void => {
        const model: TransformerModel = new TransformerModel({
            vocabularySize: 8,
            contextLength: 8,
            embeddingDimension: 8,
            numberOfLayers: 1,
            numberOfHeads: 2,
            feedForwardDimension: 16,
            architecture: 'legacy',
        });
        const generator: TextGenerator = new TextGenerator(model, new TestTokenizer());
        const result = generator.generate('hello', { maxNewTokens: 2, temperature: 0, stopOnEndOfSequence: false });
        expect(result.generatedTokenIds).toHaveLength(2);
        expect(result.metrics.cacheEnabled).toBe(false);
        expect(result.metrics.modelForwardCalls).toBe(2);
    });
});
