import { describe, expect, it } from 'vitest';
import { DEFAULT_DTYPE, assertSupportedDType, isSupportedDType } from '@core';

describe('Tensor DType', () => {
    it('should use float32 as the default dtype', () => {
        expect(DEFAULT_DTYPE).toBe('float32');
    });

    it('should recognize float32', () => {
        expect(isSupportedDType('float32')).toBe(true);
    });

    it('should reject unsupported dtypes', () => {
        expect(() => assertSupportedDType('float64')).toThrow('Unsupported tensor dtype "float64".');
    });
});
