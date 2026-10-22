import { describe, expect, it } from 'vitest';
import { BasicTokenizer } from '@core';

describe('BasicTokenizer', (): void => {
    it('should fit, encode and decode text', (): void => {
        const tokenizer: BasicTokenizer = new BasicTokenizer();
        tokenizer.fit(['Edrion aprende lenguaje.', 'Edrion genera texto.']);
        const ids: readonly number[] = tokenizer.encode('Edrion aprende lenguaje.');
        expect(ids.length).toBe(4);
        expect(tokenizer.decode(ids)).toBe('edrion aprende lenguaje.');
    });

    it('should map unknown tokens to the unknown token id', (): void => {
        const tokenizer: BasicTokenizer = new BasicTokenizer();
        tokenizer.fit(['edrion']);
        expect(tokenizer.encode('desconocido')).toEqual([tokenizer.unknownTokenId]);
    });

    it('should round-trip serialized tokenizer state', (): void => {
        const tokenizer: BasicTokenizer = new BasicTokenizer();
        tokenizer.fit(['edrion aprende']);
        const restored: BasicTokenizer = BasicTokenizer.deserialize(tokenizer.serialize());
        expect(restored.vocabularySize).toBe(tokenizer.vocabularySize);
        expect(restored.encode('edrion aprende')).toEqual(tokenizer.encode('edrion aprende'));
    });
});
