import { describe, expect, it } from 'vitest';
import { Tensor } from '@core';

describe('Tensor', () => {
    it('should create a matrix tensor', () => {
        const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);

        expect(tensor.shape).toEqual([2, 2]);
        expect(tensor.strides).toEqual([2, 1]);
        expect(tensor.rank).toBe(2);
        expect(tensor.size).toBe(4);
        expect(tensor.dtype).toBe('float32');
        expect(tensor.backend).toBe('cpu');
    });

    it('should copy input data', () => {
        const data = new Float32Array([1, 2]),
            tensor = new Tensor(data, [2]);
        data[0] = 100;
        expect(tensor.get(0)).toBe(1);
    });

    it('should reject data and shape size mismatch', () => {
        expect(() => new Tensor(new Float32Array([1, 2, 3]), [2, 2])).toThrow('Tensor data length 3 does not match shape size 4.');
    });

    it('should expose nested arrays', () => {
        const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);

        expect(tensor.toArray()).toEqual([
            [1, 2],
            [3, 4],
        ]);
    });

    it('should expose flat arrays', () => {
        const tensor = new Tensor([1, 2, 3, 4], [2, 2]);
        expect(tensor.toFlatArray()).toEqual([1, 2, 3, 4]);
    });

    it('should clone tensors independently', () => {
        const tensor = Tensor.ones([2, 2]),
            clone = tensor.clone();
        clone.set(10, 0, 0);
        expect(tensor.get(0, 0)).toBe(1);
        expect(clone.get(0, 0)).toBe(10);
    });
});
