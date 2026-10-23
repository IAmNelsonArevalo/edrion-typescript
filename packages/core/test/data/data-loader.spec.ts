import { describe, expect, it } from 'vitest';

import { ArrayDataset, DataLoader, DataLoaderConfigError, type DataBatch } from '@core';

describe('DataLoader', (): void => {
    it('should create sequential batches and keep the final partial batch', (): void => {
        const loader: DataLoader<number> = new DataLoader(new ArrayDataset([0, 1, 2, 3, 4]), {
            batchSize: 2,
        });
        const batches: DataBatch<number>[] = Array.from(loader);

        expect(loader.batchCount).toBe(3);
        expect(batches.map((batch: DataBatch<number>): readonly number[] => batch.samples)).toEqual([
            [0, 1],
            [2, 3],
            [4],
        ]);
    });

    it('should drop the final incomplete batch when requested', (): void => {
        const loader: DataLoader<number> = new DataLoader(new ArrayDataset([0, 1, 2, 3, 4]), {
            batchSize: 2,
            dropLast: true,
        });

        expect(Array.from(loader)).toHaveLength(2);
        expect(loader.batchCount).toBe(2);
    });

    it('should shuffle deterministically for the same seed and epoch', (): void => {
        const dataset: ArrayDataset<number> = new ArrayDataset([0, 1, 2, 3, 4, 5, 6, 7]);
        const first: DataLoader<number> = new DataLoader(dataset, {
            batchSize: 2,
            shuffle: true,
            seed: 42,
        });
        const second: DataLoader<number> = new DataLoader(dataset, {
            batchSize: 2,
            shuffle: true,
            seed: 42,
        });

        const firstEpoch: readonly number[] = Array.from(first.batches(3)).flatMap(
            (batch: DataBatch<number>): readonly number[] => batch.indices,
        );
        const secondEpoch: readonly number[] = Array.from(second.batches(3)).flatMap(
            (batch: DataBatch<number>): readonly number[] => batch.indices,
        );
        const nextEpoch: readonly number[] = Array.from(second.batches(4)).flatMap(
            (batch: DataBatch<number>): readonly number[] => batch.indices,
        );

        expect(firstEpoch).toEqual(secondEpoch);
        expect(nextEpoch).not.toEqual(firstEpoch);
    });

    it('should validate loader configuration', (): void => {
        expect(
            (): DataLoader<number> =>
                new DataLoader(new ArrayDataset([1]), {
                    batchSize: 0,
                }),
        ).toThrow(DataLoaderConfigError);
    });
});
