import { describe, expect, it } from 'vitest';
import { Tensor, TensorDataError, TensorIndexError, TensorReshapeError } from '@core';

describe('Tensor', (): void => {
    it('should create a vector', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3], [3]);
        expect(tensor.shape).toEqual([3]);
        expect(tensor.rank).toBe(1);
        expect(tensor.size).toBe(3);
        expect(tensor.dtype).toBe('float32');
        expect(tensor.backend).toBe('cpu');
    });

    it('should create a matrix', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
        expect(tensor.shape).toEqual([2, 3]);
        expect(tensor.strides).toEqual([3, 1]);
        expect(tensor.rank).toBe(2);
        expect(tensor.size).toBe(6);
    });

    it('should read tensor coordinates', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
        expect(tensor.get(0, 0)).toBe(1);
        expect(tensor.get(1, 2)).toBe(6);
    });

    it('should update tensor coordinates', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4], [2, 2]);
        tensor.set([1, 0], 50);
        expect(tensor.get(1, 0)).toBe(50);
    });

    it('should reject invalid coordinates', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4], [2, 2]);
        expect((): void => {
            tensor.get(4, 0);
        }).toThrow(TensorIndexError);
    });

    it('should reject invalid data length', (): void => {
        expect((): void => {
            new Tensor([1, 2], [2, 2]);
        }).toThrow(TensorDataError);
    });

    it('should reshape tensor', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
        const reshaped: Tensor = tensor.reshape([3, 2]);
        expect(reshaped.shape).toEqual([3, 2]);
        expect(reshaped.toFlatArray()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('should reject incompatible reshape', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4], [2, 2]);
        expect((): void => {
            tensor.reshape([3, 2]);
        }).toThrow(TensorReshapeError);
    });

    it('should clone tensor independently', (): void => {
        const original: Tensor = new Tensor([1, 2], [2]);
        const clone: Tensor = original.clone();
        clone.set([0], 100);
        expect(original.get(0)).toBe(1);
        expect(clone.get(0)).toBe(100);
    });

    it('should return nested array', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4], [2, 2]);
        expect(tensor.toArray()).toEqual([
            [1, 2],
            [3, 4],
        ]);
    });
});
