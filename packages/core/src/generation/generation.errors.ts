export class GenerationError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = 'GenerationError';
    }
}
