import { describe, expect, it } from 'vitest';
import { assertSupportedDType, DEFAULT_DTYPE, isSupportedDType, TensorDTypeError } from '@core';

describe('Tensor dtype', (): void => {
    it('should use float32 as default dtype', (): void => {
        expect(DEFAULT_DTYPE).toBe('float32');
    });

    it('should recognize float32', (): void => {
        expect(isSupportedDType('float32')).toBe(true);
    });

    it('should reject unsupported dtype', (): void => {
        expect((): void => {
            assertSupportedDType('float64');
        }).toThrow(TensorDTypeError);
    });
});
