import { describe, expect, it } from 'vitest';
import { CAUSAL_MASK_VALUE, createCausalMask } from '@core';

describe('createCausalMask', (): void => {
    it('should hide future positions', (): void => {
        const mask = createCausalMask(3);
        expect(mask.shape).toEqual([3, 3]);
        expect(mask.get(0, 0)).toBe(0);
        expect(mask.get(0, 1)).toBe(CAUSAL_MASK_VALUE);
        expect(mask.get(1, 0)).toBe(0);
        expect(mask.get(1, 2)).toBe(CAUSAL_MASK_VALUE);
        expect(mask.get(2, 2)).toBe(0);
    });
});
