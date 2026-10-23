import type { Module, NamedParameter } from '@core/nn/module';
import type { DType } from '@core/tensor/dtype';

/** Approximate bytes required by one logical tensor element. */
export function bytesPerDType(dtype: DType): number {
    switch (dtype) {
        case 'float16':
        case 'bfloat16':
            return 2;
        case 'float32':
            return 4;
    }
}

export interface ModelProfile {
    readonly parameterCount: number;
    readonly parameterBytes: number;
    readonly gradientBytes: number;
    readonly optimizerStateBytesAdamW: number;
    readonly totalTrainingStateBytesAdamW: number;
}

/** Estimates model/training-state memory from registered parameters. */
export function profileModel(model: Module): ModelProfile {
    const named: readonly NamedParameter[] = model.namedParameters();
    let parameterCount: number = 0;
    let parameterBytes: number = 0;

    for (let index: number = 0; index < named.length; index += 1) {
        const entry: NamedParameter | undefined = named[index];
        if (entry === undefined) {
            continue;
        }
        const count: number = entry.parameter.size;
        const bytes: number = count * bytesPerDType(entry.parameter.dtype);
        parameterCount += count;
        parameterBytes += bytes;
    }

    const gradientBytes: number = parameterBytes;
    const optimizerStateBytesAdamW: number = parameterBytes * 2;
    return {
        parameterCount,
        parameterBytes,
        gradientBytes,
        optimizerStateBytesAdamW,
        totalTrainingStateBytesAdamW: parameterBytes + gradientBytes + optimizerStateBytesAdamW,
    };
}
