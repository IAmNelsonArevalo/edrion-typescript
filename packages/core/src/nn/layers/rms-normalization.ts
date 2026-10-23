import { LayerShapeError, NeuralNetworkError } from '@core/nn/nn.errors';
import { Module } from '@core/nn/module';
import { Parameter } from '@core/nn/parameter';
import { Tensor } from '@core/tensor/tensor';

export interface RMSNormalizationOptions {
    readonly epsilon?: number;
    readonly backend?: string;
}

/** Root-mean-square normalization used by modern decoder-only LLMs. */
export class RMSNormalization extends Module {
    public readonly featureCount: number;
    public readonly epsilon: number;
    public readonly weight: Parameter;

    public constructor(featureCount: number, options: RMSNormalizationOptions = {}) {
        super();
        if (!Number.isInteger(featureCount) || featureCount <= 0) {
            throw new NeuralNetworkError(`featureCount must be a positive integer. Received ${featureCount}.`);
        }
        const epsilon: number = options.epsilon ?? 1e-6;
        if (!Number.isFinite(epsilon) || epsilon <= 0) {
            throw new NeuralNetworkError(`RMSNormalization epsilon must be finite and greater than zero. Received ${epsilon}.`);
        }
        this.featureCount = featureCount;
        this.epsilon = epsilon;
        this.weight = this.registerParameter(
            'weight',
            new Parameter(new Float32Array(featureCount).fill(1), [featureCount], options.backend === undefined ? {} : { backend: options.backend }),
        );
    }

    public override forward(input: Tensor): Tensor {
        if (input.rank < 1) {
            throw new LayerShapeError(`RMSNormalization expects rank >= 1 input. Received rank ${input.rank}.`);
        }
        const features: number | undefined = input.shape[input.rank - 1];
        if (features !== this.featureCount) {
            throw new LayerShapeError(`RMSNormalization expected ${this.featureCount} features but received ${String(features)}.`);
        }
        const rms: Tensor = input.pow(2).mean(-1, true).add(this.epsilon).sqrt();
        return input.div(rms).mul(this.weight);
    }
}
