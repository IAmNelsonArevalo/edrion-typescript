import { LayerShapeError, NeuralNetworkError } from '@core/nn/nn.errors';
import { xavierUniform } from '@core/nn/initializers';
import { Module } from '@core/nn/module';
import { Parameter } from '@core/nn/parameter';
import { Tensor } from '@core/tensor/tensor';

/**
 * Linear layer construction options.
 */
export interface LinearOptions {
    readonly bias?: boolean;
    readonly backend?: string;
}

/**
 * Fully connected affine transformation.
 *
 * Computes: y = xW + b
 */
export class Linear extends Module {
    public readonly inFeatures: number;
    public readonly outFeatures: number;
    public readonly weight: Parameter;
    public readonly bias: Parameter | undefined;

    public constructor(inFeatures: number, outFeatures: number, options: LinearOptions = {}) {
        super();

        assertPositiveFeatureCount(inFeatures, 'inFeatures');
        assertPositiveFeatureCount(outFeatures, 'outFeatures');

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;

        this.weight = this.registerParameter(
            'weight',
            new Parameter(
                xavierUniform(inFeatures, outFeatures),
                [inFeatures, outFeatures],
                options.backend === undefined ? {} : { backend: options.backend },
            ),
        );

        const useBias: boolean = options.bias ?? true;

        this.bias = useBias
            ? this.registerParameter(
                  'bias',
                  new Parameter(
                      new Float32Array(outFeatures),
                      [outFeatures],
                      options.backend === undefined ? {} : { backend: options.backend },
                  ),
              )
            : undefined;
    }

    public override forward(input: Tensor): Tensor {
        if (input.rank !== 1 && input.rank !== 2) {
            throw new LayerShapeError(`Linear.forward() expects rank 1 or 2 input. Received rank ${input.rank}.`);
        }

        const shape: readonly number[] = input.shape;
        const featureAxis: number = shape.length - 1;
        const inputFeatures: number | undefined = shape[featureAxis];

        if (inputFeatures === undefined) {
            throw new LayerShapeError('Linear.forward() could not resolve the input feature dimension.');
        }

        if (inputFeatures !== this.inFeatures) {
            throw new LayerShapeError(
                `Linear expected ${this.inFeatures} input features but received ${inputFeatures}.`,
            );
        }

        if (input.backend !== this.weight.backend) {
            throw new LayerShapeError(
                `Linear backend mismatch. Input uses "${input.backend}" while parameters use "${this.weight.backend}".`,
            );
        }

        const matrixInput: Tensor = input.rank === 1 ? input.reshape([1, this.inFeatures]) : input;

        let output: Tensor = matrixInput.matmul(this.weight);

        if (this.bias !== undefined) {
            output = output.add(this.bias);
        }

        if (input.rank === 1) {
            return output.reshape([this.outFeatures]);
        }

        return output;
    }
}

function assertPositiveFeatureCount(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new NeuralNetworkError(`${name} must be a positive integer. Received ${value}.`);
    }
}
