import { describe, expect, it } from 'vitest';
import { Tensor } from '@core';

describe('Tensor indexing', () => {
    it('should read matrix values', () => {
        const tensor = new Tensor([1, 2, 3, 4], [2, 2]);
        expect(tensor.get(0, 0)).toBe(1);
        expect(tensor.get(0, 1)).toBe(2);
        expect(tensor.get(1, 0)).toBe(3);
        expect(tensor.get(1, 1)).toBe(4);
    });

    it('should calculate rank 3 offsets correctly', () => {
        const tensor = new Tensor(
            Array.from(
                {
                    length: 24,
                },
                (_, index) => index,
            ),
            [2, 3, 4],
        );
        expect(tensor.get(1, 2, 3)).toBe(23);
    });

    it('should update tensor values', () => {
        const tensor = Tensor.zeros([2, 2]);
        tensor.set(25, 1, 0);
        expect(tensor.get(1, 0)).toBe(25);
    });

    it('should reject out of bounds indices', () => {
        const tensor = Tensor.zeros([2, 2]);
        expect(() => tensor.get(2, 0)).toThrow('Tensor index 2 is out of bounds for dimension 0 with size 2.');
    });

    it('should reject incorrect index count', () => {
        const tensor = Tensor.zeros([2, 2]);
        expect(() => tensor.get(0)).toThrow('Tensor rank is 2, but 1 indices were provided.');
    });

    it('should reject decimal indices', () => {
        const tensor = Tensor.zeros([2]);
        expect(() => tensor.get(0.5)).toThrow('Tensor index for dimension 0 must be an integer.');
    });
});
