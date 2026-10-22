export interface Tokenizer {
    readonly vocabularySize: number;
    readonly padTokenId: number;
    readonly unknownTokenId: number;
    readonly beginningOfSequenceTokenId: number;
    readonly endOfSequenceTokenId: number;

    encode(text: string, addSpecialTokens?: boolean): readonly number[];
    decode(tokenIds: readonly number[], skipSpecialTokens?: boolean): string;
}
