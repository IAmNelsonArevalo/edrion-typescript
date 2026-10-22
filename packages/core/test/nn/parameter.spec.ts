import { describe, expect, it } from 'vitest';

import { Parameter } from '@core';

describe('Parameter', (): void => {
    it('should always require gradients', (): void => {
        const parameter: Parameter = new Parameter([1, 2, 3], [3]);

        expect(parameter.requiresGrad).toBe(true);
    });
});
