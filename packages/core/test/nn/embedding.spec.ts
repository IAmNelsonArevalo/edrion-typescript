import { describe, expect, it } from 'vitest';

import { Embedding, EmbeddingIndexError, Tensor } from '@core';

describe('Embedding', (): void => {
    it('should map token indices to vectors', (): void => {
        const embedding: Embedding = new Embedding(3, 2);

        embedding.weight.set([0, 0], 1);
        embedding.weight.set([0, 1], 2);
        embedding.weight.set([1, 0], 3);
        embedding.weight.set([1, 1], 4);
        embedding.weight.set([2, 0], 5);
        embedding.weight.set([2, 1], 6);

        const indices: Tensor = new Tensor([2, 0, 1], [3]);

        expect(embedding.forward(indices).toArray()).toEqual([
            [5, 6],
            [1, 2],
            [3, 4],
        ]);
    });

    it('should preserve multidimensional index shape', (): void => {
        const embedding: Embedding = new Embedding(4, 3);
        const indices: Tensor = new Tensor([0, 1, 2, 3], [2, 2]);

        expect(embedding.forward(indices).shape).toEqual([2, 2, 3]);
    });

    it('should accumulate gradients into embedding weights', (): void => {
        const embedding: Embedding = new Embedding(3, 2);
        const indices: Tensor = new Tensor([1, 1], [2]);

        embedding.forward(indices).sum().backward();

        expect(embedding.weight.grad?.toArray()).toEqual([
            [0, 0],
            [2, 2],
            [0, 0],
        ]);
    });

    it('should reject invalid indices', (): void => {
        const embedding: Embedding = new Embedding(3, 2);
        const indices: Tensor = new Tensor([3], [1]);

        expect((): Tensor => embedding.forward(indices)).toThrow(EmbeddingIndexError);
    });
});
