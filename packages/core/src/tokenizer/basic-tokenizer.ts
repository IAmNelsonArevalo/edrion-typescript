import type { Tokenizer } from '@core/tokenizer/tokenizer.interface';
import { TokenizerError } from '@core/tokenizer/tokenizer.errors';
import { Vocabulary, type VocabularyState } from '@core/tokenizer/vocabulary';

export interface BasicTokenizerOptions {
    readonly lowercase?: boolean;
}

export interface BasicTokenizerState {
    readonly version: 1;
    readonly lowercase: boolean;
    readonly vocabulary: VocabularyState;
}

const PAD_TOKEN: string = '<pad>';
const UNKNOWN_TOKEN: string = '<unk>';
const BOS_TOKEN: string = '<bos>';
const EOS_TOKEN: string = '<eos>';

const SPECIAL_TOKENS: readonly string[] = [PAD_TOKEN, UNKNOWN_TOKEN, BOS_TOKEN, EOS_TOKEN];

export class BasicTokenizer implements Tokenizer {
    private readonly vocabulary: Vocabulary;
    private readonly lowercase: boolean;

    public constructor(options: BasicTokenizerOptions = {}, vocabulary?: Vocabulary) {
        this.lowercase = options.lowercase ?? true;
        this.vocabulary = vocabulary ?? new Vocabulary(SPECIAL_TOKENS);
        this.ensureSpecialTokens();
    }

    public get vocabularySize(): number {
        return this.vocabulary.size;
    }

    public get padTokenId(): number {
        return this.requireTokenId(PAD_TOKEN);
    }

    public get unknownTokenId(): number {
        return this.requireTokenId(UNKNOWN_TOKEN);
    }

    public get beginningOfSequenceTokenId(): number {
        return this.requireTokenId(BOS_TOKEN);
    }

    public get endOfSequenceTokenId(): number {
        return this.requireTokenId(EOS_TOKEN);
    }

    public fit(texts: readonly string[]): void {
        for (let textIndex: number = 0; textIndex < texts.length; textIndex += 1) {
            const text: string | undefined = texts[textIndex];
            if (text === undefined) {
                throw new TokenizerError(`Training text at index ${textIndex} does not exist.`);
            }
            const tokens: readonly string[] = this.tokenize(text);
            for (let tokenIndex: number = 0; tokenIndex < tokens.length; tokenIndex += 1) {
                const token: string | undefined = tokens[tokenIndex];
                if (token === undefined) {
                    throw new TokenizerError(`Token at index ${tokenIndex} does not exist.`);
                }
                this.vocabulary.add(token);
            }
        }
    }

    public encode(text: string, addSpecialTokens: boolean = false): readonly number[] {
        const tokens: readonly string[] = this.tokenize(text);
        const ids: number[] = [];
        if (addSpecialTokens) {
            ids.push(this.beginningOfSequenceTokenId);
        }
        for (let index: number = 0; index < tokens.length; index += 1) {
            const token: string | undefined = tokens[index];
            if (token === undefined) {
                throw new TokenizerError(`Token at index ${index} does not exist.`);
            }
            const id: number = this.vocabulary.getId(token) ?? this.unknownTokenId;
            ids.push(id);
        }
        if (addSpecialTokens) {
            ids.push(this.endOfSequenceTokenId);
        }
        return ids;
    }

    public decode(tokenIds: readonly number[], skipSpecialTokens: boolean = true): string {
        const tokens: string[] = [];
        for (let index: number = 0; index < tokenIds.length; index += 1) {
            const id: number | undefined = tokenIds[index];
            if (id === undefined) {
                throw new TokenizerError(`Token id at index ${index} does not exist.`);
            }
            const token: string = this.vocabulary.getToken(id);
            if (skipSpecialTokens && SPECIAL_TOKENS.includes(token)) {
                continue;
            }
            tokens.push(token);
        }
        return detokenize(tokens);
    }

    public toState(): BasicTokenizerState {
        return {
            version: 1,
            lowercase: this.lowercase,
            vocabulary: this.vocabulary.toState(),
        };
    }

    public serialize(): string {
        return JSON.stringify(this.toState());
    }

    public static fromState(state: BasicTokenizerState): BasicTokenizer {
        if (state.version !== 1) {
            throw new TokenizerError(`Unsupported tokenizer state version ${String(state.version)}.`);
        }
        return new BasicTokenizer({ lowercase: state.lowercase }, Vocabulary.fromState(state.vocabulary));
    }

    public static deserialize(serialized: string): BasicTokenizer {
        let parsed: unknown;
        try {
            parsed = JSON.parse(serialized) as unknown;
        } catch (error: unknown) {
            throw new TokenizerError(
                `Unable to deserialize tokenizer state: ${error instanceof Error ? error.message : String(error)}.`,
            );
        }
        if (!isBasicTokenizerState(parsed)) {
            throw new TokenizerError('Serialized tokenizer state has an invalid structure.');
        }
        return BasicTokenizer.fromState(parsed);
    }

    private tokenize(text: string): readonly string[] {
        const normalized: string = this.lowercase ? text.toLowerCase() : text;
        const matches: RegExpMatchArray | null = normalized.match(/[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu);
        return matches === null ? [] : [...matches];
    }

    private requireTokenId(token: string): number {
        const id: number | undefined = this.vocabulary.getId(token);
        if (id === undefined) {
            throw new TokenizerError(`Required special token "${token}" is missing.`);
        }
        return id;
    }

    private ensureSpecialTokens(): void {
        for (let index: number = 0; index < SPECIAL_TOKENS.length; index += 1) {
            const token: string | undefined = SPECIAL_TOKENS[index];
            if (token === undefined || this.vocabulary.getId(token) === undefined) {
                throw new TokenizerError('Tokenizer vocabulary does not contain all required special tokens.');
            }
        }
    }
}

function detokenize(tokens: readonly string[]): string {
    let text: string = '';
    const punctuation: RegExp = /^[.,!?;:%)\]}]$/u;
    const opening: RegExp = /^[([{]$/u;
    for (let index: number = 0; index < tokens.length; index += 1) {
        const token: string | undefined = tokens[index];
        if (token === undefined) {
            throw new TokenizerError(`Decoded token at index ${index} does not exist.`);
        }
        if (text.length === 0 || punctuation.test(token)) {
            text += token;
            continue;
        }
        const previous: string | undefined = tokens[index - 1];
        text += previous !== undefined && opening.test(previous) ? token : ` ${token}`;
    }
    return text;
}

function isBasicTokenizerState(value: unknown): value is BasicTokenizerState {
    if (typeof value !== 'object' || value === null) {
        return false;
    }
    const candidate: Record<string, unknown> = value as Record<string, unknown>;
    const vocabulary: unknown = candidate['vocabulary'];
    if (candidate['version'] !== 1 || typeof candidate['lowercase'] !== 'boolean') {
        return false;
    }
    if (typeof vocabulary !== 'object' || vocabulary === null) {
        return false;
    }
    const vocabularyRecord: Record<string, unknown> = vocabulary as Record<string, unknown>;
    const tokens: unknown = vocabularyRecord['tokens'];
    return Array.isArray(tokens) && tokens.every((token: unknown): boolean => typeof token === 'string');
}
