import { describe, expect, it } from 'vitest';
import { Parameter } from '@core/nn';
import { AdamW } from '@core/optim';

describe('AdamW', () => {
    it('applies decoupled weight decay', () => {
        const parameter: Parameter = new Parameter([1], [1]);
        const optimizer: AdamW = new AdamW([parameter], { learningRate: 0.1, weightDecay: 0.1 });
        parameter.mul(0).sum().backward();
        optimizer.step();
        expect(parameter.get(0)).toBeCloseTo(0.99, 5);
    });
});
