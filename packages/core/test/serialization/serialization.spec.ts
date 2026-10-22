import { describe, expect, it } from 'vitest';

import {
    Linear,
    Serializer,
    Tensor,
    TensorSerializer,
    captureModelState,
    createCheckpoint,
    restoreModelState,
} from '@core';

describe('Serialization', (): void => {
    it('should serialize and deserialize tensor values', (): void => {
        const tensor: Tensor = new Tensor([1, 2, 3, 4], [2, 2]);

        const serialized = TensorSerializer.serialize(tensor);
        const restored: Tensor = TensorSerializer.deserialize(serialized);

        expect(restored.shape).toEqual([2, 2]);
        expect(restored.toFlatArray()).toEqual([1, 2, 3, 4]);
    });

    it('should capture and restore model state', (): void => {
        const model: Linear = new Linear(2, 1);
        const originalState = captureModelState(model);
        const weightBefore: number[] = model.weight.toFlatArray();

        model.weight.assign(new Float32Array(model.weight.size).fill(99));
        expect(model.weight.toFlatArray()).not.toEqual(weightBefore);

        restoreModelState(model, originalState);
        expect(model.weight.toFlatArray()).toEqual(weightBefore);
    });

    it('should round-trip a checkpoint as JSON', (): void => {
        const model: Linear = new Linear(1, 1);
        const checkpoint = createCheckpoint({
            frameworkVersion: '0.9.0',
            model: captureModelState(model),
            training: {
                epoch: 3,
                globalStep: 12,
                completed: false,
            },
            metadata: {
                experiment: 'feature-008',
            },
        });

        const json: string = Serializer.stringify(checkpoint);
        const restored = Serializer.parse(json);

        expect(restored.formatVersion).toBe(1);
        expect(restored.frameworkVersion).toBe('0.9.0');
        expect(restored.training.epoch).toBe(3);
        expect(restored.metadata?.['experiment']).toBe('feature-008');
    });
});
