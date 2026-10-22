import { NeuralNetworkError } from '@core/nn/nn.errors';

/**
 * Creates Xavier/Glorot uniform values for a weight matrix.
 */
export function xavierUniform(fanIn: number, fanOut: number): Float32Array {
    assertPositiveInteger(fanIn, 'fanIn');
    assertPositiveInteger(fanOut, 'fanOut');

    const size: number = fanIn * fanOut;
    const limit: number = Math.sqrt(6 / (fanIn + fanOut));
    const values: Float32Array = new Float32Array(size);

    for (let index: number = 0; index < size; index += 1) {
        values[index] = (Math.random() * 2 - 1) * limit;
    }

    return values;
}

/**
 * Creates uniformly distributed embedding values.
 */
export function embeddingUniform(vocabularySize: number, embeddingDimension: number): Float32Array {
    assertPositiveInteger(vocabularySize, 'vocabularySize');
    assertPositiveInteger(embeddingDimension, 'embeddingDimension');

    const size: number = vocabularySize * embeddingDimension;
    const limit: number = 1 / Math.sqrt(embeddingDimension);
    const values: Float32Array = new Float32Array(size);

    for (let index: number = 0; index < size; index += 1) {
        values[index] = (Math.random() * 2 - 1) * limit;
    }

    return values;
}

function assertPositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new NeuralNetworkError(`${name} must be a positive integer. Received ${value}.`);
    }
}
