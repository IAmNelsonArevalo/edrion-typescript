import { describe, expect, it } from 'vitest';
import {
    BpeTokenizer,
    CausalLanguageModel,
    CausalLanguageModelDataset,
    createLanguageModelConfig,
    evaluateLanguageModel,
} from '@core/index';

describe('Edrion v2 language-model foundation', () => {
    it('creates a causal next-token dataset', () => {
        const ids = Array.from({ length: 20 }, (_value: unknown, index: number): number => index % 8);
        const dataset = new CausalLanguageModelDataset(ids, { sequenceLength: 4, stride: 4 });
        expect(dataset.length).toBeGreaterThan(0);
        expect(dataset.get(0).input.shape).toEqual([4]);
        expect(dataset.get(0).target.shape).toEqual([4]);
    });

    it('runs a modern GPT-style LM forward pass', () => {
        const tokenizer = new BpeTokenizer({ targetVocabularySize: 32, minimumPairFrequency: 2 });
        tokenizer.fit(['edrion aprende lenguaje ', 'edrion genera texto ', 'lenguaje con edrion ']);
        const model = new CausalLanguageModel(
            createLanguageModelConfig('tiny', tokenizer.vocabularySize, {
                contextLength: 8,
                embeddingDimension: 16,
                numberOfLayers: 2,
                numberOfHeads: 2,
                feedForwardDimension: 48,
            }),
        );
        const dataset = CausalLanguageModelDataset.fromTexts(
            ['edrion aprende lenguaje edrion genera texto'],
            tokenizer,
            { sequenceLength: 4 },
        );
        const sample = dataset.get(0);
        expect(model.forward(sample.input).shape).toEqual([4, tokenizer.vocabularySize]);
        expect(Number.isFinite(model.loss(sample.input, sample.target).get())).toBe(true);
    });

    it('reports finite perplexity', () => {
        const tokenizer = new BpeTokenizer({ targetVocabularySize: 24, minimumPairFrequency: 2 });
        tokenizer.fit(['aaaa bbbb cccc ', 'aaaa bbbb ']);
        const model = new CausalLanguageModel(
            createLanguageModelConfig('tiny', tokenizer.vocabularySize, {
                contextLength: 4,
                embeddingDimension: 8,
                numberOfLayers: 1,
                numberOfHeads: 2,
                feedForwardDimension: 24,
            }),
        );
        const dataset = CausalLanguageModelDataset.fromTexts(['aaaa bbbb cccc aaaa bbbb'], tokenizer, {
            sequenceLength: 4,
        });
        const evaluation = evaluateLanguageModel(model, dataset, 1);
        expect(Number.isFinite(evaluation.perplexity)).toBe(true);
        expect(evaluation.perplexity).toBeGreaterThan(0);
    });
});
