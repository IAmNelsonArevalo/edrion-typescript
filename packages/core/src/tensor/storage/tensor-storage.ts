import { quantizeToDType, type DType } from '@core/tensor/dtype';
import { TensorDataError, TensorIndexError } from '@core/tensor/tensor.errors';

/** Accepted input data for tensor storage. */
export type TensorData = readonly number[] | Float32Array;

/** Owns the numeric storage used by a Tensor. */
export class TensorStorage {
    private readonly values: Float32Array;

    /** Creates independent tensor storage. */
    public constructor(
        data: TensorData,
        expectedSize: number,
        private readonly dtype: DType = 'float32',
    ) {
        if (data.length !== expectedSize) {
            throw new TensorDataError(
                `Tensor data length ${data.length} does not match expected tensor size ${expectedSize}.`,
            );
        }

        this.values = new Float32Array(expectedSize);
        for (let index: number = 0; index < expectedSize; index += 1) {
            const value: number | undefined = data[index];
            if (value === undefined) {
                throw new TensorDataError(`Missing tensor storage value at offset ${index}.`);
            }
            this.values[index] = quantizeToDType(value, this.dtype);
        }
    }

    /** Number of scalar values stored. */
    public get length(): number {
        return this.values.length;
    }

    /** Returns a scalar value at a flat offset. */
    public get(offset: number): number {
        this.assertValidOffset(offset);

        const value: number | undefined = this.values[offset];

        if (value === undefined) {
            throw new TensorIndexError(`Tensor storage value at offset ${offset} does not exist.`);
        }

        return value;
    }

    /** Updates a scalar value at a flat offset. */
    public set(offset: number, value: number): void {
        this.assertValidOffset(offset);
        this.values[offset] = quantizeToDType(value, this.dtype);
    }

    /** Returns an independent Float32Array copy. */
    public toFloat32Array(): Float32Array {
        return new Float32Array(this.values);
    }

    /** Returns an independent JavaScript array. */
    public toArray(): number[] {
        return Array.from(this.values);
    }

    /** Creates an independent clone. */
    public clone(): TensorStorage {
        return new TensorStorage(this.values, this.values.length, this.dtype);
    }

    private assertValidOffset(offset: number): void {
        if (!Number.isInteger(offset)) {
            throw new TensorIndexError(`Tensor storage offset must be an integer. Received ${offset}.`);
        }

        if (offset < 0 || offset >= this.values.length) {
            throw new TensorIndexError(
                `Tensor storage offset ${offset} is outside the valid range 0-${this.values.length - 1}.`,
            );
        }
    }
}
