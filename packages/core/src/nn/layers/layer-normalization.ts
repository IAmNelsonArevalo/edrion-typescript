import { LayerShapeError, NeuralNetworkError } from '@core/nn/nn.errors';
import { Module } from '@core/nn/module';
import { Parameter } from '@core/nn/parameter';
import { Tensor } from '@core/tensor/tensor';

/**
 * LayerNorm construction options.
 */
export interface LayerNormalizationOptions {
    readonly epsilon?: number;
    readonly affine?: boolean;
    readonly backend?: string;
}

/**
 * Normalizes the final tensor dimension.
 */
export class LayerNormalization extends Module {
    public readonly normalizedDimension: number;
    public readonly epsilon: number;
    public readonly weight: Parameter | undefined;
    public readonly bias: Parameter | undefined;

    public constructor(normalizedDimension: number, options: LayerNormalizationOptions = {}) {
        super();

        if (!Number.isInteger(normalizedDimension) || normalizedDimension <= 0) {
            throw new NeuralNetworkError(
                `normalizedDimension must be a positive integer. Received ${normalizedDimension}.`,
            );
        }

        const epsilon: number = options.epsilon ?? 1e-5;

        if (!Number.isFinite(epsilon) || epsilon <= 0) {
            throw new NeuralNetworkError(
                `LayerNormalization epsilon must be finite and greater than zero. Received ${epsilon}.`,
            );
        }

        this.normalizedDimension = normalizedDimension;
        this.epsilon = epsilon;

        const affine: boolean = options.affine ?? true;

        this.weight = affine
            ? this.registerParameter(
                  'weight',
                  new Parameter(
                      new Float32Array(normalizedDimension).fill(1),
                      [normalizedDimension],
                      options.backend === undefined ? {} : { backend: options.backend },
                  ),
              )
            : undefined;

        this.bias = affine
            ? this.registerParameter(
                  'bias',
                  new Parameter(
                      new Float32Array(normalizedDimension),
                      [normalizedDimension],
                      options.backend === undefined ? {} : { backend: options.backend },
                  ),
              )
            : undefined;
    }

    public override forward(input: Tensor): Tensor {
        if (input.rank === 0) {
            throw new LayerShapeError('LayerNormalization requires an input with rank greater than zero.');
        }

        const inputShape: readonly number[] = input.shape;
        const lastDimension: number | undefined = inputShape[inputShape.length - 1];

        if (lastDimension === undefined) {
            throw new LayerShapeError('LayerNormalization could not resolve the final input dimension.');
        }

        if (lastDimension !== this.normalizedDimension) {
            throw new LayerShapeError(
                `LayerNormalization expected final dimension ${this.normalizedDimension} but received ${lastDimension}.`,
            );
        }

        if (this.weight !== undefined && input.backend !== this.weight.backend) {
            throw new LayerShapeError(
                `LayerNormalization backend mismatch. Input uses "${input.backend}" while parameters use "${this.weight.backend}".`,
            );
        }

        const mean: Tensor = input.mean(-1, true);
        const centered: Tensor = input.sub(mean);
        const variance: Tensor = centered.pow(2).mean(-1, true);
        const denominator: Tensor = variance.add(this.epsilon).sqrt();

        let output: Tensor = centered.div(denominator);

        if (this.weight !== undefined) {
            output = output.mul(this.weight);
        }

        if (this.bias !== undefined) {
            output = output.add(this.bias);
        }

        return output;
    }
}
