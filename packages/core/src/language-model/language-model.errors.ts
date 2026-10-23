/** Base language-model error. */
export class LanguageModelError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = new.target.name;
    }
}
