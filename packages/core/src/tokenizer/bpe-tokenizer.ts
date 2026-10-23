import type { Dataset } from '@core/data/dataset';
import type { Tokenizer } from '@core/tokenizer/tokenizer.interface';
import { TokenizerError } from '@core/tokenizer/tokenizer.errors';
import { Vocabulary, type VocabularyState } from '@core/tokenizer/vocabulary';

const PAD_TOKEN: string = '<pad>';
const UNKNOWN_TOKEN: string = '<unk>';
const BOS_TOKEN: string = '<bos>';
const EOS_TOKEN: string = '<eos>';

const SPECIAL_TOKENS: readonly string[] = [PAD_TOKEN, UNKNOWN_TOKEN, BOS_TOKEN, EOS_TOKEN];

export interface BpeTokenizerOptions {
    readonly lowercase?: boolean;
    readonly targetVocabularySize?: number;
    readonly minimumPairFrequency?: number;
}

export interface BpeMerge {
    readonly left: string;
    readonly right: string;
    readonly token: string;
}

export interface BpeTrainingStats {
    readonly documents: number;
    readonly initialAlphabetSize: number;
    readonly mergesLearned: number;
    readonly vocabularySize: number;
}

export interface BpeTokenizerState {
    readonly version: 1;
    readonly lowercase: boolean;
    readonly targetVocabularySize: number;
    readonly minimumPairFrequency: number;
    readonly vocabulary: VocabularyState;
    readonly merges: readonly BpeMerge[];
}

interface PairFrequency {
    readonly left: string;
    readonly right: string;
    count: number;
}

/**
 * Deterministic Unicode Byte-Pair-Encoding style tokenizer.
 *
 * The initial alphabet is composed of Unicode code points observed in the
 * training corpus. The most frequent adjacent pair is repeatedly merged until
 * the target vocabulary size is reached or no pair satisfies the configured
 * minimum frequency.
 *
 * Edrion deliberately keeps this implementation framework independent. It is
 * suitable for building the first Edrion language-model vocabulary while a
 * future tokenizer backend can add byte-level normalization and larger-scale
 * training without changing the Tokenizer contract.
 */
export class BpeTokenizer implements Tokenizer {
    private readonly lowercase: boolean;
    private readonly targetVocabularySize: number;
    private readonly minimumPairFrequency: number;
    private vocabulary: Vocabulary;
    private merges: BpeMerge[];
    private trainingStats: BpeTrainingStats | undefined;

    public constructor(options: BpeTokenizerOptions = {}, state?: BpeTokenizerState) {
        this.lowercase = options.lowercase ?? state?.lowercase ?? true;
        this.targetVocabularySize = options.targetVocabularySize ?? state?.targetVocabularySize ?? 8192;
        this.minimumPairFrequency = options.minimumPairFrequency ?? state?.minimumPairFrequency ?? 2;
        validateConfiguration(this.targetVocabularySize, this.minimumPairFrequency);

        this.vocabulary = state === undefined ? new Vocabulary(SPECIAL_TOKENS) : Vocabulary.fromState(state.vocabulary);
        this.merges = state === undefined ? [] : state.merges.map((merge: BpeMerge): BpeMerge => ({ ...merge }));
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

    public get learnedMerges(): readonly BpeMerge[] {
        return this.merges.map((merge: BpeMerge): BpeMerge => ({ ...merge }));
    }

    public get stats(): BpeTrainingStats | undefined {
        return this.trainingStats === undefined ? undefined : { ...this.trainingStats };
    }

    /** Trains a new BPE vocabulary from an in-memory corpus. */
    public fit(texts: readonly string[]): void {
        if (texts.length === 0) {
            throw new TokenizerError('BPE training corpus cannot be empty.');
        }

        const sequences: string[][] = new Array<string[]>(texts.length);
        const alphabet: Set<string> = new Set<string>();

        for (let textIndex: number = 0; textIndex < texts.length; textIndex += 1) {
            const text: string | undefined = texts[textIndex];
            if (text === undefined) {
                throw new TokenizerError(`Training text at index ${textIndex} does not exist.`);
            }

            const symbols: string[] = this.toInitialSymbols(text);
            sequences[textIndex] = symbols;

            for (let symbolIndex: number = 0; symbolIndex < symbols.length; symbolIndex += 1) {
                const symbol: string | undefined = symbols[symbolIndex];
                if (symbol === undefined) {
                    throw new TokenizerError(`Training symbol at index ${symbolIndex} does not exist.`);
                }
                alphabet.add(symbol);
            }
        }

        const initialVocabularySize: number = SPECIAL_TOKENS.length + alphabet.size;
        if (initialVocabularySize > this.targetVocabularySize) {
            throw new TokenizerError(
                `BPE targetVocabularySize ${this.targetVocabularySize} is smaller than the required initial vocabulary ${initialVocabularySize}.`,
            );
        }

        this.vocabulary = new Vocabulary(SPECIAL_TOKENS);
        this.merges = [];

        const sortedAlphabet: string[] = [...alphabet].sort(compareStrings);
        for (let index: number = 0; index < sortedAlphabet.length; index += 1) {
            const symbol: string | undefined = sortedAlphabet[index];
            if (symbol === undefined) {
                throw new TokenizerError(`Alphabet symbol at index ${index} does not exist.`);
            }
            this.vocabulary.add(symbol);
        }

        while (this.vocabulary.size < this.targetVocabularySize) {
            const bestPair: PairFrequency | undefined = findBestPair(sequences);
            if (bestPair === undefined || bestPair.count < this.minimumPairFrequency) {
                break;
            }

            const mergedToken: string = `${bestPair.left}${bestPair.right}`;
            const merge: BpeMerge = {
                left: bestPair.left,
                right: bestPair.right,
                token: mergedToken,
            };

            this.merges.push(merge);
            this.vocabulary.add(mergedToken);

            for (let sequenceIndex: number = 0; sequenceIndex < sequences.length; sequenceIndex += 1) {
                const sequence: string[] | undefined = sequences[sequenceIndex];
                if (sequence === undefined) {
                    throw new TokenizerError(`Training sequence at index ${sequenceIndex} does not exist.`);
                }
                sequences[sequenceIndex] = applyMerge(sequence, merge);
            }
        }

        this.trainingStats = {
            documents: texts.length,
            initialAlphabetSize: alphabet.size,
            mergesLearned: this.merges.length,
            vocabularySize: this.vocabulary.size,
        };
    }

    /** Trains from any Edrion Dataset<string> without coupling Core to I/O. */
    public fitDataset(dataset: Dataset<string>): void {
        if (dataset.length === 0) {
            throw new TokenizerError('BPE training dataset cannot be empty.');
        }

        const texts: string[] = new Array<string>(dataset.length);
        for (let index: number = 0; index < dataset.length; index += 1) {
            texts[index] = dataset.get(index);
        }
        this.fit(texts);
    }

    public encode(text: string, addSpecialTokens: boolean = false): readonly number[] {
        let symbols: string[] = this.toInitialSymbols(text);

        for (let mergeIndex: number = 0; mergeIndex < this.merges.length; mergeIndex += 1) {
            const merge: BpeMerge | undefined = this.merges[mergeIndex];
            if (merge === undefined) {
                throw new TokenizerError(`BPE merge at index ${mergeIndex} does not exist.`);
            }
            symbols = applyMerge(symbols, merge);
        }

        const ids: number[] = [];
        if (addSpecialTokens) {
            ids.push(this.beginningOfSequenceTokenId);
        }

        for (let index: number = 0; index < symbols.length; index += 1) {
            const symbol: string | undefined = symbols[index];
            if (symbol === undefined) {
                throw new TokenizerError(`Encoded symbol at index ${index} does not exist.`);
            }
            ids.push(this.vocabulary.getId(symbol) ?? this.unknownTokenId);
        }

        if (addSpecialTokens) {
            ids.push(this.endOfSequenceTokenId);
        }

        return ids;
    }

    public encodeBatch(texts: readonly string[], addSpecialTokens: boolean = false): readonly (readonly number[])[] {
        const encoded: Array<readonly number[]> = new Array<readonly number[]>(texts.length);
        for (let index: number = 0; index < texts.length; index += 1) {
            const text: string | undefined = texts[index];
            if (text === undefined) {
                throw new TokenizerError(`Batch text at index ${index} does not exist.`);
            }
            encoded[index] = this.encode(text, addSpecialTokens);
        }
        return encoded;
    }

    public decode(tokenIds: readonly number[], skipSpecialTokens: boolean = true): string {
        let output: string = '';

        for (let index: number = 0; index < tokenIds.length; index += 1) {
            const id: number | undefined = tokenIds[index];
            if (id === undefined) {
                throw new TokenizerError(`Token id at index ${index} does not exist.`);
            }
            const token: string = this.vocabulary.getToken(id);
            if (skipSpecialTokens && SPECIAL_TOKENS.includes(token)) {
                continue;
            }
            output += token;
        }

        return output;
    }

    public toState(): BpeTokenizerState {
        return {
            version: 1,
            lowercase: this.lowercase,
            targetVocabularySize: this.targetVocabularySize,
            minimumPairFrequency: this.minimumPairFrequency,
            vocabulary: this.vocabulary.toState(),
            merges: this.learnedMerges,
        };
    }

    public serialize(): string {
        return JSON.stringify(this.toState());
    }

    public static fromState(state: BpeTokenizerState): BpeTokenizer {
        validateBpeTokenizerState(state);
        return new BpeTokenizer({}, state);
    }

    public static deserialize(serialized: string): BpeTokenizer {
        let parsed: unknown;
        try {
            parsed = JSON.parse(serialized) as unknown;
        } catch (error: unknown) {
            throw new TokenizerError(
                `Unable to deserialize BPE tokenizer state: ${error instanceof Error ? error.message : String(error)}.`,
            );
        }

        if (!isBpeTokenizerState(parsed)) {
            throw new TokenizerError('Serialized BPE tokenizer state has an invalid structure.');
        }

        return BpeTokenizer.fromState(parsed);
    }

    private toInitialSymbols(text: string): string[] {
        const normalized: string = this.lowercase ? text.toLowerCase() : text;
        return Array.from(normalized);
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
                throw new TokenizerError('BPE vocabulary does not contain all required special tokens.');
            }
        }
    }
}

function validateConfiguration(targetVocabularySize: number, minimumPairFrequency: number): void {
    if (!Number.isInteger(targetVocabularySize) || targetVocabularySize <= SPECIAL_TOKENS.length) {
        throw new TokenizerError(
            `BPE targetVocabularySize must be an integer greater than ${SPECIAL_TOKENS.length}. Received ${targetVocabularySize}.`,
        );
    }
    if (!Number.isInteger(minimumPairFrequency) || minimumPairFrequency <= 0) {
        throw new TokenizerError(
            `BPE minimumPairFrequency must be a positive integer. Received ${minimumPairFrequency}.`,
        );
    }
}

function findBestPair(sequences: readonly (readonly string[])[]): PairFrequency | undefined {
    const frequencies: Map<string, PairFrequency> = new Map<string, PairFrequency>();

    for (let sequenceIndex: number = 0; sequenceIndex < sequences.length; sequenceIndex += 1) {
        const sequence: readonly string[] | undefined = sequences[sequenceIndex];
        if (sequence === undefined) {
            throw new TokenizerError(`BPE sequence at index ${sequenceIndex} does not exist.`);
        }

        for (let index: number = 0; index + 1 < sequence.length; index += 1) {
            const left: string | undefined = sequence[index];
            const right: string | undefined = sequence[index + 1];
            if (left === undefined || right === undefined) {
                throw new TokenizerError(`Unable to resolve BPE pair at sequence ${sequenceIndex}, offset ${index}.`);
            }

            const key: string = JSON.stringify([left, right]);
            const current: PairFrequency | undefined = frequencies.get(key);
            if (current === undefined) {
                frequencies.set(key, { left, right, count: 1 });
            } else {
                current.count += 1;
            }
        }
    }

    let best: PairFrequency | undefined;
    for (const candidate of frequencies.values()) {
        if (
            best === undefined ||
            candidate.count > best.count ||
            (candidate.count === best.count && comparePairs(candidate, best) < 0)
        ) {
            best = candidate;
        }
    }

    return best;
}

function comparePairs(left: PairFrequency, right: PairFrequency): number {
    const leftComparison: number = compareStrings(left.left, right.left);
    return leftComparison !== 0 ? leftComparison : compareStrings(left.right, right.right);
}

function compareStrings(left: string, right: string): number {
    return left < right ? -1 : left > right ? 1 : 0;
}

function applyMerge(symbols: readonly string[], merge: BpeMerge): string[] {
    const output: string[] = [];

    for (let index: number = 0; index < symbols.length; ) {
        const current: string | undefined = symbols[index];
        if (current === undefined) {
            throw new TokenizerError(`BPE symbol at index ${index} does not exist.`);
        }

        const next: string | undefined = symbols[index + 1];
        if (current === merge.left && next === merge.right) {
            output.push(merge.token);
            index += 2;
        } else {
            output.push(current);
            index += 1;
        }
    }

    return output;
}

function validateBpeTokenizerState(state: BpeTokenizerState): void {
    if (state.version !== 1) {
        throw new TokenizerError(`Unsupported BPE tokenizer state version ${String(state.version)}.`);
    }
    validateConfiguration(state.targetVocabularySize, state.minimumPairFrequency);
}

function isBpeTokenizerState(value: unknown): value is BpeTokenizerState {
    if (typeof value !== 'object' || value === null) {
        return false;
    }

    const candidate: Record<string, unknown> = value as Record<string, unknown>;
    const vocabulary: unknown = candidate['vocabulary'];
    const merges: unknown = candidate['merges'];

    if (
        candidate['version'] !== 1 ||
        typeof candidate['lowercase'] !== 'boolean' ||
        typeof candidate['targetVocabularySize'] !== 'number' ||
        typeof candidate['minimumPairFrequency'] !== 'number' ||
        typeof vocabulary !== 'object' ||
        vocabulary === null ||
        !Array.isArray(merges)
    ) {
        return false;
    }

    const vocabularyRecord: Record<string, unknown> = vocabulary as Record<string, unknown>;
    const tokens: unknown = vocabularyRecord['tokens'];
    if (!Array.isArray(tokens) || !tokens.every((token: unknown): boolean => typeof token === 'string')) {
        return false;
    }

    return merges.every((merge: unknown): boolean => {
        if (typeof merge !== 'object' || merge === null) {
            return false;
        }
        const record: Record<string, unknown> = merge as Record<string, unknown>;
        return typeof record['left'] === 'string' && typeof record['right'] === 'string' && typeof record['token'] === 'string';
    });
}
