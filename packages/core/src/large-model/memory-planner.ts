import { MemoryPlanningError } from '@core/large-model/large-model.errors';

export interface LargeModelMemoryPlanInput {
    readonly parameterBytes: number;
    readonly gradientBytes: number;
    readonly optimizerStateBytes: number;
    readonly activationBytesPerSample: number;
    readonly availableBytes: number;
    readonly safetyFraction?: number;
}

export interface LargeModelMemoryPlan {
    readonly staticBytes: number;
    readonly usableBytes: number;
    readonly activationBudgetBytes: number;
    readonly recommendedBatchSize: number;
    readonly fitsStaticState: boolean;
}

/** Produces a conservative batch-size plan from a byte budget. */
export function planLargeModelMemory(input: LargeModelMemoryPlanInput): LargeModelMemoryPlan {
    const safetyFraction: number = input.safetyFraction ?? 0.9;
    const values: readonly number[] = [
        input.parameterBytes,
        input.gradientBytes,
        input.optimizerStateBytes,
        input.activationBytesPerSample,
        input.availableBytes,
    ];
    for (let index: number = 0; index < values.length; index += 1) {
        const value: number | undefined = values[index];
        if (value === undefined || !Number.isFinite(value) || value < 0) {
            throw new MemoryPlanningError(`Memory planning value at index ${index} must be finite and non-negative.`);
        }
    }
    if (!(safetyFraction > 0 && safetyFraction <= 1)) {
        throw new MemoryPlanningError(`safetyFraction must be in (0, 1]. Received ${safetyFraction}.`);
    }

    const usableBytes: number = Math.floor(input.availableBytes * safetyFraction);
    const staticBytes: number = input.parameterBytes + input.gradientBytes + input.optimizerStateBytes;
    const activationBudgetBytes: number = Math.max(0, usableBytes - staticBytes);
    const fitsStaticState: boolean = staticBytes <= usableBytes;
    const recommendedBatchSize: number =
        fitsStaticState && input.activationBytesPerSample > 0
            ? Math.max(1, Math.floor(activationBudgetBytes / input.activationBytesPerSample))
            : fitsStaticState
              ? 1
              : 0;

    return { staticBytes, usableBytes, activationBudgetBytes, recommendedBatchSize, fitsStaticState };
}
