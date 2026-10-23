import { describe, expect, it } from 'vitest';
import { Linear, createShardedModelState, restoreShardedModelState, profileModel, planLargeModelMemory } from '@core/index';

describe('large-model infrastructure', () => {
    it('profiles model memory', () => {
        const model = new Linear(4, 3);
        const profile = profileModel(model);
        expect(profile.parameterCount).toBeGreaterThan(0);
        expect(profile.totalTrainingStateBytesAdamW).toBeGreaterThan(profile.parameterBytes);
    });

    it('round-trips sharded state', () => {
        const model = new Linear(4, 3);
        const state = createShardedModelState(model, 16);
        const restored = new Linear(4, 3);
        restoreShardedModelState(restored, state);
        expect(restored.namedParameters()[0]?.parameter.toFlatArray()).toEqual(model.namedParameters()[0]?.parameter.toFlatArray());
    });

    it('plans memory conservatively', () => {
        const plan = planLargeModelMemory({ parameterBytes: 100, gradientBytes: 100, optimizerStateBytes: 200, activationBytesPerSample: 100, availableBytes: 1000 });
        expect(plan.fitsStaticState).toBe(true);
        expect(plan.recommendedBatchSize).toBeGreaterThan(0);
    });
});
