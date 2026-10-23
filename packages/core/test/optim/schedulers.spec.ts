import { describe, expect, it } from 'vitest';
import { Parameter } from '@core/nn';
import { SGD, WarmupCosineScheduler } from '@core/optim';

describe('learning-rate schedulers', () => {
    it('warms up and then decays', () => {
        const optimizer: SGD = new SGD([new Parameter([1], [1])], { learningRate: 1 });
        const scheduler: WarmupCosineScheduler = new WarmupCosineScheduler(optimizer, {
            warmupSteps: 2,
            totalSteps: 6,
            startFactor: 0.5,
            minimumLearningRate: 0.1,
        });
        const first: number = scheduler.step(1);
        const warm: number = scheduler.step(2);
        const final: number = scheduler.step(6);
        expect(first).toBeGreaterThan(0.5);
        expect(warm).toBeCloseTo(1);
        expect(final).toBeCloseTo(0.1);
    });
});
