import { describe, expect, it } from 'vitest';

import { ArrayDataset, DatasetIndexError, MappedDataset } from '@core';

describe('Dataset', (): void => {
    it('should expose immutable array-backed samples', (): void => {
        const source: number[] = [1, 2, 3];
        const dataset: ArrayDataset<number> = new ArrayDataset(source);

        source[0] = 99;

        expect(dataset.length).toBe(3);
        expect(dataset.get(0)).toBe(1);
    });

    it('should reject invalid indices', (): void => {
        const dataset: ArrayDataset<number> = new ArrayDataset([1]);

        expect((): number => dataset.get(-1)).toThrow(DatasetIndexError);
        expect((): number => dataset.get(1)).toThrow(DatasetIndexError);
        expect((): number => dataset.get(0.5)).toThrow(DatasetIndexError);
    });

    it('should lazily transform samples with MappedDataset', (): void => {
        const source: ArrayDataset<number> = new ArrayDataset([1, 2, 3]);
        const mapped: MappedDataset<number, string> = new MappedDataset(
            source,
            (sample: number, index: number): string => `${index}:${sample * 2}`,
        );

        expect(mapped.get(2)).toBe('2:6');
    });
});
