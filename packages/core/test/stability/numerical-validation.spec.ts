import { describe, expect, it } from 'vitest';

import { NonFiniteTensorError, Tensor, assertFiniteTensor } from '@core';

describe('Numerical validation', (): void => {
    it('accepts finite tensors', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3], [3]);
        expect((): void => assertFiniteTensor(tensor)).not.toThrow();
    });

    it('rejects NaN tensor values', (): void => {
        const tensor: Tensor = new Tensor([1, Number.NaN], [2]);
        expect((): void => assertFiniteTensor(tensor)).toThrow(NonFiniteTensorError);
    });

    it('rejects infinite tensor values', (): void => {
        const tensor: Tensor = new Tensor([1, Number.POSITIVE_INFINITY], [2]);
        expect((): void => assertFiniteTensor(tensor)).toThrow(NonFiniteTensorError);
    });
});
