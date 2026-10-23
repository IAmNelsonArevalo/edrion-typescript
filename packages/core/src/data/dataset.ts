import { DatasetIndexError } from '@core/data/data.errors';

/** Read-only random-access dataset abstraction. */
export interface Dataset<TSample> {
    /** Number of samples available in this dataset. */
    readonly length: number;

    /** Returns one sample by zero-based index. */
    get(index: number): TSample;
}

/** Immutable array-backed dataset. */
export class ArrayDataset<TSample> implements Dataset<TSample> {
    private readonly samples: readonly TSample[];

    public constructor(samples: readonly TSample[]) {
        this.samples = [...samples];
    }

    public get length(): number {
        return this.samples.length;
    }

    public get(index: number): TSample {
        assertDatasetIndex(index, this.samples.length);

        const sample: TSample | undefined = this.samples[index];

        if (sample === undefined) {
            throw new DatasetIndexError(`Dataset sample at index ${index} does not exist.`);
        }

        return sample;
    }
}

/** Lazily applies a transform whenever a sample is read. */
export class MappedDataset<TInput, TOutput> implements Dataset<TOutput> {
    public constructor(
        private readonly source: Dataset<TInput>,
        private readonly transform: (sample: TInput, index: number) => TOutput,
    ) {}

    public get length(): number {
        return this.source.length;
    }

    public get(index: number): TOutput {
        assertDatasetIndex(index, this.source.length);
        return this.transform(this.source.get(index), index);
    }
}

/** Validates a zero-based dataset index. */
export function assertDatasetIndex(index: number, length: number): void {
    if (!Number.isInteger(index)) {
        throw new DatasetIndexError(`Dataset index must be an integer. Received ${index}.`);
    }

    if (index < 0 || index >= length) {
        throw new DatasetIndexError(
            `Dataset index ${index} is outside the valid range [0, ${Math.max(0, length - 1)}].`,
        );
    }
}
