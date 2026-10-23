import { DataLoaderConfigError } from '@core/data/data.errors';
import type { Dataset } from '@core/data/dataset';
import { createSeededRandom, type SeededRandom } from '@core/data/random';

/** Public DataLoader configuration. */
export interface DataLoaderConfig {
    readonly batchSize: number;
    readonly shuffle?: boolean;
    readonly dropLast?: boolean;
    readonly seed?: number;
}

/** One logical batch emitted by a DataLoader. */
export interface DataBatch<TSample> {
    readonly batchIndex: number;
    readonly indices: readonly number[];
    readonly samples: readonly TSample[];
}

/** Normalized DataLoader configuration. */
export interface NormalizedDataLoaderConfig {
    readonly batchSize: number;
    readonly shuffle: boolean;
    readonly dropLast: boolean;
    readonly seed: number;
}

/**
 * Framework-independent synchronous data loader.
 *
 * FEATURE-012 intentionally keeps I/O outside @edrion/core. The loader owns
 * deterministic ordering and batching while adapters can provide datasets
 * backed by files, HTTP, IndexedDB or any other storage mechanism.
 */
export class DataLoader<TSample> implements Iterable<DataBatch<TSample>> {
    public readonly config: NormalizedDataLoaderConfig;

    public constructor(
        public readonly dataset: Dataset<TSample>,
        config: DataLoaderConfig,
    ) {
        this.config = normalizeDataLoaderConfig(config);
    }

    /** Number of batches emitted for one epoch. */
    public get batchCount(): number {
        if (this.config.dropLast) {
            return Math.floor(this.dataset.length / this.config.batchSize);
        }

        return Math.ceil(this.dataset.length / this.config.batchSize);
    }

    /** Returns batches for a specific epoch, allowing deterministic reshuffling. */
    public *batches(epoch: number = 0): IterableIterator<DataBatch<TSample>> {
        if (!Number.isInteger(epoch) || epoch < 0) {
            throw new DataLoaderConfigError(`DataLoader epoch must be a non-negative integer. Received ${epoch}.`);
        }

        const indices: number[] = createSequentialIndices(this.dataset.length);

        if (this.config.shuffle) {
            shuffleIndices(indices, this.config.seed + epoch);
        }

        let batchIndex: number = 0;

        for (let start: number = 0; start < indices.length; start += this.config.batchSize) {
            const end: number = Math.min(start + this.config.batchSize, indices.length);
            const batchIndices: number[] = indices.slice(start, end);

            if (this.config.dropLast && batchIndices.length < this.config.batchSize) {
                break;
            }

            const samples: TSample[] = new Array<TSample>(batchIndices.length);

            for (let offset: number = 0; offset < batchIndices.length; offset += 1) {
                const sampleIndex: number | undefined = batchIndices[offset];

                if (sampleIndex === undefined) {
                    throw new DataLoaderConfigError(`Unable to resolve dataset index for batch ${batchIndex}.`);
                }

                samples[offset] = this.dataset.get(sampleIndex);
            }

            yield {
                batchIndex,
                indices: batchIndices,
                samples,
            };

            batchIndex += 1;
        }
    }

    /** Default iteration uses epoch zero. */
    public [Symbol.iterator](): IterableIterator<DataBatch<TSample>> {
        return this.batches(0);
    }
}

/** Validates and normalizes a DataLoader configuration. */
export function normalizeDataLoaderConfig(config: DataLoaderConfig): NormalizedDataLoaderConfig {
    if (!Number.isInteger(config.batchSize) || config.batchSize <= 0) {
        throw new DataLoaderConfigError(`DataLoader batchSize must be a positive integer. Received ${config.batchSize}.`);
    }

    const seed: number = config.seed ?? 1337;

    if (!Number.isInteger(seed) || !Number.isFinite(seed)) {
        throw new DataLoaderConfigError(`DataLoader seed must be a finite integer. Received ${seed}.`);
    }

    return {
        batchSize: config.batchSize,
        shuffle: config.shuffle ?? false,
        dropLast: config.dropLast ?? false,
        seed,
    };
}

function createSequentialIndices(length: number): number[] {
    const indices: number[] = new Array<number>(length);

    for (let index: number = 0; index < length; index += 1) {
        indices[index] = index;
    }

    return indices;
}

function shuffleIndices(indices: number[], seed: number): void {
    const random: SeededRandom = createSeededRandom(seed);

    for (let index: number = indices.length - 1; index > 0; index -= 1) {
        const swapIndex: number = Math.floor(random.next() * (index + 1));
        const current: number | undefined = indices[index];
        const replacement: number | undefined = indices[swapIndex];

        if (current === undefined || replacement === undefined) {
            throw new DataLoaderConfigError('Unable to resolve shuffle indices.');
        }

        indices[index] = replacement;
        indices[swapIndex] = current;
    }
}
