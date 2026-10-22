import { TensorIndexError, TensorShapeError } from '@core/tensor/tensor.errors';

/**
 * Transposes a rank-2 tensor.
 */
export function transposeMatrix(data: Float32Array, rows: number, columns: number): Float32Array {
    const result: Float32Array = new Float32Array(data.length);

    for (let row: number = 0; row < rows; row += 1) {
        for (let column: number = 0; column < columns; column += 1) {
            const sourceOffset: number = row * columns + column;
            const targetOffset: number = column * rows + row;
            const value: number | undefined = data[sourceOffset];

            if (value === undefined) {
                throw new TensorIndexError(`Matrix value at offset ${sourceOffset} does not exist.`);
            }

            result[targetOffset] = value;
        }
    }

    return result;
}

/**
 * Multiplies two rank-2 matrices.
 */
export function matmulOperation(
    leftData: Float32Array,
    leftRows: number,
    leftColumns: number,
    rightData: Float32Array,
    rightRows: number,
    rightColumns: number,
): Float32Array {
    if (leftColumns !== rightRows) {
        throw new TensorShapeError(
            `Cannot multiply matrices with shapes [${leftRows}, ${leftColumns}] and [${rightRows}, ${rightColumns}].`,
        );
    }

    const result: Float32Array = new Float32Array(leftRows * rightColumns);

    for (let row: number = 0; row < leftRows; row += 1) {
        for (let column: number = 0; column < rightColumns; column += 1) {
            let total: number = 0;

            for (let inner: number = 0; inner < leftColumns; inner += 1) {
                const leftOffset: number = row * leftColumns + inner;
                const rightOffset: number = inner * rightColumns + column;
                const leftValue: number = readValue(leftData, leftOffset);
                const rightValue: number = readValue(rightData, rightOffset);

                total += leftValue * rightValue;
            }

            result[row * rightColumns + column] = total;
        }
    }

    return result;
}

function readValue(data: Float32Array, offset: number): number {
    const value: number | undefined = data[offset];

    if (value === undefined) {
        throw new TensorIndexError(`Matrix value at offset ${offset} does not exist.`);
    }

    return value;
}
