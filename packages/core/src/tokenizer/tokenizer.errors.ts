export class TokenizerError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'TokenizerError';
    }
}

export class VocabularyError extends TokenizerError {
    public constructor(message: string) {
        super(message);
        this.name = 'VocabularyError';
    }
}
