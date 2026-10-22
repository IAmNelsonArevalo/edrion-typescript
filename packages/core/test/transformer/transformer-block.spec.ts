import { describe, expect, it } from 'vitest';
import { Tensor, TransformerBlock } from '@core';

describe('TransformerBlock', (): void => {
    it('should preserve hidden state shape', (): void => {
        const block: TransformerBlock = new TransformerBlock(4, 2, 8);
        const input: Tensor = new Tensor([1, 0, 0, 1, 0, 1, 1, 0], [2, 4], { requiresGrad: true });
        const output: Tensor = block.forward(input);
        expect(output.shape).toEqual([2, 4]);
        expect(block.parameters().length).toBeGreaterThan(0);
    });
});
