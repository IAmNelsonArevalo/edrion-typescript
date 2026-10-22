import { EmbeddingIndexError, LayerShapeError, NeuralNetworkError } from '@core/nn/nn.errors';
import { embeddingUniform } from '@core/nn/initializers';
import { Module } from '@core/nn/module';
import { Parameter } from '@core/nn/parameter';
import { calculateTensorSize } from '@core/tensor/shape';
import { Tensor } from '@core/tensor/tensor';

/**
 * Embedding layer construction options.
 */
export interface EmbeddingOptions {
    readonly backend?: string;
}

/**
 * Converts integer token indices into trainable dense vectors.
 */
export class Embedding extends Module {
    public readonly numEmbeddings: number;
    public readonly embeddingDimension: number;
    public readonly weight: Parameter;

    public constructor(numEmbeddings: number, embeddingDimension: number, options: EmbeddingOptions = {}) {
        super();

        assertPositiveInteger(numEmbeddings, 'numEmbeddings');
        assertPositiveInteger(embeddingDimension, 'embeddingDimension');

        this.numEmbeddings = numEmbeddings;
        this.embeddingDimension = embeddingDimension;

        this.weight = this.registerParameter(
            'weight',
            new Parameter(
                embeddingUniform(numEmbeddings, embeddingDimension),
                [numEmbeddings, embeddingDimension],
                options.backend === undefined ? {} : { backend: options.backend },
            ),
        );
    }

    public override forward(input: Tensor): Tensor {
        if (input.requiresGrad) {
            throw new LayerShapeError('Embedding indices must not require gradients.');
        }

        if (input.backend !== this.weight.backend) {
            throw new LayerShapeError(
                `Embedding backend mismatch. Input uses "${input.backend}" while parameters use "${this.weight.backend}".`,
            );
        }

        const indices: Float32Array = input.toFloat32Array();
        const indexCount: number = calculateTensorSize(input.shape);
        const oneHot: Float32Array = new Float32Array(indexCount * this.numEmbeddings);

        for (let index: number = 0; index < indexCount; index += 1) {
            const tokenIndex: number | undefined = indices[index];

            if (tokenIndex === undefined) {
                throw new EmbeddingIndexError(`Missing embedding index at flat position ${index}.`);
            }

            if (!Number.isInteger(tokenIndex)) {
                throw new EmbeddingIndexError(
                    `Embedding index at flat position ${index} must be an integer. Received ${tokenIndex}.`,
                );
            }

            if (tokenIndex < 0 || tokenIndex >= this.numEmbeddings) {
                throw new EmbeddingIndexError(
                    `Embedding index ${tokenIndex} is outside the valid range 0-${this.numEmbeddings - 1}.`,
                );
            }

            oneHot[index * this.numEmbeddings + tokenIndex] = 1;
        }

        const oneHotTensor: Tensor = new Tensor(oneHot, [indexCount, this.numEmbeddings], {
            backend: input.backend,
        });

        const embedded: Tensor = oneHotTensor.matmul(this.weight);
        const outputShape: number[] = [...input.shape, this.embeddingDimension];

        return embedded.reshape(outputShape);
    }
}

function assertPositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new NeuralNetworkError(`${name} must be a positive integer. Received ${value}.`);
    }
}
