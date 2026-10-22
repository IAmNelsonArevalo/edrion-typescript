import { describe, expect, it } from 'vitest';
import { TensorDataError, TensorIndexError, TensorStorage } from '@core';

describe('TensorStorage', (): void => {
    it('should create Float32 storage', (): void => {
        const storage: TensorStorage = new TensorStorage([1, 2, 3], 3);
        expect(storage.length).toBe(3);
        expect(storage.toFloat32Array()).toBeInstanceOf(Float32Array);
    });

    it('should create independent storage', (): void => {
        const source: Float32Array = new Float32Array([1, 2]);
        const storage: TensorStorage = new TensorStorage(source, 2);
        source[0] = 999;
        expect(storage.get(0)).toBe(1);
    });

    it('should reject invalid data length', (): void => {
        expect((): void => {
            new TensorStorage([1], 2);
        }).toThrow(TensorDataError);
    });

    it('should update values', (): void => {
        const storage: TensorStorage = new TensorStorage([1, 2], 2);
        storage.set(1, 10);
        expect(storage.get(1)).toBe(10);
    });

    it('should reject invalid offset', (): void => {
        const storage: TensorStorage = new TensorStorage([1], 1);
        expect((): void => {
            storage.get(5);
        }).toThrow(TensorIndexError);
    });
});
