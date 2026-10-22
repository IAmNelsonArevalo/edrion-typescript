import { describe, expect, it } from 'vitest';

import { SerializationError, Tensor, TensorSerializer } from '@core';

describe('Serialization numerical safety', (): void => {
    it('refuses to serialize non-finite tensors', (): void => {
        const tensor: Tensor = new Tensor([1, Number.NaN], [2]);
        expect((): void => {
            TensorSerializer.serialize(tensor);
        }).toThrow();
    });

    it('refuses to deserialize non-finite tensor data', (): void => {
        expect((): void => {
            TensorSerializer.deserialize({
                shape: [2],
                dtype: 'float32',
                backend: 'cpu',
                data: [1, Number.POSITIVE_INFINITY],
            });
        }).toThrow(SerializationError);
    });
});
