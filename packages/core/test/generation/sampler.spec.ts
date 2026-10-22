import { describe, expect, it } from 'vitest';
import { GreedySampler, TemperatureSampler, TopKSampler } from '@core';

describe('Generation samplers', (): void => {
    it('should choose the highest logit greedily', (): void => {
        const sampler: GreedySampler = new GreedySampler();
        expect(sampler.sample([1, 5, 2])).toBe(1);
    });

    it('should sample deterministically with an injected random source', (): void => {
        const sampler: TemperatureSampler = new TemperatureSampler(1, (): number => 0);
        expect(sampler.sample([10, 0])).toBe(0);
    });

    it('should only choose from top-k candidates', (): void => {
        const sampler: TopKSampler = new TopKSampler(1, 1, (): number => 0.75);
        expect(sampler.sample([1, 9, 3])).toBe(1);
    });
});
