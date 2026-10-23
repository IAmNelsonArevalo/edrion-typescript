import type { DType } from '@core/tensor/dtype';
import { Tensor } from '@core/tensor/tensor';

/** Precision policy used by training/inference code. */
export interface PrecisionPolicyOptions {
    readonly computeDType?: DType;
    readonly parameterDType?: DType;
    readonly outputDType?: DType;
}

/** Explicit mixed-precision policy. Parameters remain float32 by default. */
export class PrecisionPolicy {
    public readonly computeDType: DType;
    public readonly parameterDType: DType;
    public readonly outputDType: DType;

    public constructor(options: PrecisionPolicyOptions = {}) {
        this.computeDType = options.computeDType ?? 'float16';
        this.parameterDType = options.parameterDType ?? 'float32';
        this.outputDType = options.outputDType ?? 'float32';
    }

    public castInput(tensor: Tensor): Tensor {
        if (tensor.dtype === this.computeDType) {
            return tensor;
        }
        // CPU Tensor operators currently require equal logical dtypes. Quantize through
        // the requested compute dtype and cast back so v1.8.0 can emulate reduced
        // activation precision without violating existing operator contracts.
        return tensor.cast(this.computeDType).cast(tensor.dtype);
    }

    public castOutput(tensor: Tensor): Tensor {
        return tensor.cast(this.outputDType);
    }
}
