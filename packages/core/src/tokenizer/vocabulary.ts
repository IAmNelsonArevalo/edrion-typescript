import { VocabularyError } from '@core/tokenizer/tokenizer.errors';

export interface VocabularyState {
    readonly tokens: readonly string[];
}

export class Vocabulary {
    private readonly tokenToId: Map<string, number> = new Map<string, number>();
    private readonly idToToken: string[] = [];

    public constructor(tokens: readonly string[] = []) {
        for (let index: number = 0; index < tokens.length; index += 1) {
            const token: string | undefined = tokens[index];
            if (token === undefined) {
                throw new VocabularyError(`Vocabulary token at index ${index} does not exist.`);
            }
            this.add(token);
        }
    }

    public get size(): number {
        return this.idToToken.length;
    }

    public add(token: string): number {
        if (token.length === 0) {
            throw new VocabularyError('Vocabulary token cannot be empty.');
        }
        const existing: number | undefined = this.tokenToId.get(token);
        if (existing !== undefined) {
            return existing;
        }
        const id: number = this.idToToken.length;
        this.idToToken.push(token);
        this.tokenToId.set(token, id);
        return id;
    }

    public getId(token: string): number | undefined {
        return this.tokenToId.get(token);
    }

    public getToken(id: number): string {
        if (!Number.isInteger(id) || id < 0 || id >= this.idToToken.length) {
            throw new VocabularyError(`Vocabulary id ${id} is outside the valid range.`);
        }
        const token: string | undefined = this.idToToken[id];
        if (token === undefined) {
            throw new VocabularyError(`Vocabulary token for id ${id} does not exist.`);
        }
        return token;
    }

    public toState(): VocabularyState {
        return { tokens: [...this.idToToken] };
    }

    public static fromState(state: VocabularyState): Vocabulary {
        return new Vocabulary(state.tokens);
    }
}
