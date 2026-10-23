import type { Dataset } from '@core/data/dataset';
import type { Tokenizer } from '@core/tokenizer/tokenizer.interface';

/** Converts a text dataset into token-id samples lazily. */
export class TokenizedTextDataset implements Dataset<readonly number[]> {
    public constructor(
        private readonly source: Dataset<string>,
        private readonly tokenizer: Tokenizer,
        private readonly addSpecialTokens: boolean = false,
    ) {}

    public get length(): number {
        return this.source.length;
    }

    public get(index: number): readonly number[] {
        return this.tokenizer.encode(this.source.get(index), this.addSpecialTokens);
    }
}
