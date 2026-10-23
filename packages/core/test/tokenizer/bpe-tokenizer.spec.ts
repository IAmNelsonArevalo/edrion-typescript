import { describe, expect, it } from 'vitest';

import { ArrayDataset } from '@core/data';
import { BpeTokenizer, TokenizedTextDataset, TokenizerError } from '@core/tokenizer';

describe('BpeTokenizer', (): void => {
    it('learns deterministic merges from repeated text', (): void => {
        const tokenizer: BpeTokenizer = new BpeTokenizer({
            lowercase: true,
            targetVocabularySize: 20,
            minimumPairFrequency: 2,
        });

        tokenizer.fit(['banana banana', 'banana bandana']);

        expect(tokenizer.learnedMerges.length).toBeGreaterThan(0);
        expect(tokenizer.stats?.documents).toBe(2);
        expect(tokenizer.vocabularySize).toBeLessThanOrEqual(20);
    });

    it('round-trips Unicode text without losing spaces or punctuation', (): void => {
        const tokenizer: BpeTokenizer = new BpeTokenizer({
            lowercase: false,
            targetVocabularySize: 64,
            minimumPairFrequency: 1,
        });
        const text: string = '¡Hola, Edrion! IA en Bogotá 🚀';

        tokenizer.fit([text, text]);
        const ids: readonly number[] = tokenizer.encode(text);

        expect(tokenizer.decode(ids)).toBe(text);
    });

    it('supports BOS and EOS special tokens', (): void => {
        const tokenizer: BpeTokenizer = new BpeTokenizer({ targetVocabularySize: 32, minimumPairFrequency: 1 });
        tokenizer.fit(['edrion aprende', 'edrion aprende']);

        const ids: readonly number[] = tokenizer.encode('edrion', true);

        expect(ids[0]).toBe(tokenizer.beginningOfSequenceTokenId);
        expect(ids[ids.length - 1]).toBe(tokenizer.endOfSequenceTokenId);
        expect(tokenizer.decode(ids)).toBe('edrion');
    });

    it('serializes vocabulary and merge rules deterministically', (): void => {
        const tokenizer: BpeTokenizer = new BpeTokenizer({ targetVocabularySize: 30, minimumPairFrequency: 1 });
        tokenizer.fit(['edrion aprende lenguaje', 'edrion aprende ia']);
        const serialized: string = tokenizer.serialize();
        const restored: BpeTokenizer = BpeTokenizer.deserialize(serialized);
        const text: string = 'edrion aprende';

        expect(restored.encode(text)).toEqual(tokenizer.encode(text));
        expect(restored.decode(restored.encode(text))).toBe(text);
        expect(restored.learnedMerges).toEqual(tokenizer.learnedMerges);
    });

    it('trains directly from an Edrion Dataset', (): void => {
        const tokenizer: BpeTokenizer = new BpeTokenizer({ targetVocabularySize: 40, minimumPairFrequency: 1 });
        const dataset: ArrayDataset<string> = new ArrayDataset<string>(['uno dos', 'uno tres']);

        tokenizer.fitDataset(dataset);

        expect(tokenizer.stats?.documents).toBe(2);
        expect(tokenizer.encode('uno').length).toBeGreaterThan(0);
    });

    it('provides a lazy tokenized dataset adapter', (): void => {
        const tokenizer: BpeTokenizer = new BpeTokenizer({ targetVocabularySize: 40, minimumPairFrequency: 1 });
        const source: ArrayDataset<string> = new ArrayDataset<string>(['edrion', 'typescript']);
        tokenizer.fit(['edrion', 'typescript']);
        const dataset: TokenizedTextDataset = new TokenizedTextDataset(source, tokenizer, true);

        expect(dataset.length).toBe(2);
        expect(dataset.get(0)[0]).toBe(tokenizer.beginningOfSequenceTokenId);
    });

    it('uses unknown token ids for unseen Unicode characters', (): void => {
        const tokenizer: BpeTokenizer = new BpeTokenizer({ targetVocabularySize: 20, minimumPairFrequency: 1 });
        tokenizer.fit(['abc abc']);

        expect(tokenizer.encode('z')).toEqual([tokenizer.unknownTokenId]);
    });

    it('rejects an empty corpus', (): void => {
        const tokenizer: BpeTokenizer = new BpeTokenizer();

        expect((): void => tokenizer.fit([])).toThrow(TokenizerError);
    });

    it('rejects a target vocabulary smaller than the required alphabet', (): void => {
        const tokenizer: BpeTokenizer = new BpeTokenizer({ targetVocabularySize: 5, minimumPairFrequency: 1 });

        expect((): void => tokenizer.fit(['abcdef'])).toThrow(/smaller than the required initial vocabulary/u);
    });

    it('is deterministic across independent training runs', (): void => {
        const corpus: readonly string[] = ['abab abab', 'abab baba'];
        const left: BpeTokenizer = new BpeTokenizer({ targetVocabularySize: 20, minimumPairFrequency: 1 });
        const right: BpeTokenizer = new BpeTokenizer({ targetVocabularySize: 20, minimumPairFrequency: 1 });

        left.fit(corpus);
        right.fit(corpus);

        expect(right.toState()).toEqual(left.toState());
    });
});
