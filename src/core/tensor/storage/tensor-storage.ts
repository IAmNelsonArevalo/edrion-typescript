/**
 * Data accepted when constructing a tensor.
 */
export type TensorData = Float32Array | readonly number[];

/**
 * Internal Float32 tensor storage.
 *
 * TensorStorage owns its underlying memory.
 */
export class TensorStorage {
    private readonly values: Float32Array;

    /**
     * Creates Float32 tensor storage.
     *
     * Input data is copied to prevent external
     * mutation of tensor memory.
     */
    constructor(data: TensorData) {
        if (data instanceof Float32Array) {
            this.values = new Float32Array(data);
            return;
        }

        this.values = Float32Array.from(data);
    }

    /**
     * Number of elements in storage.
     */
    get length(): number {
        return this.values.length;
    }

    /**
     * Returns a value using a linear index.
     */
    get(index: number): number {
        const value = this.values[index];

        if (value === undefined) throw new RangeError(`Tensor storage index ${index} is out of bounds.`);

        return value;
    }

    /**
     * Updates a value using a linear index.
     */
    set(index: number, value: number): void {
        if (index < 0 || index >= this.values.length) throw new RangeError(`Tensor storage index ${index} is out of bounds.`);

        this.values[index] = value;
    }

    /**
     * Creates an independent copy of storage.
     */
    clone(): TensorStorage {
        return new TensorStorage(this.values);
    }

    /**
     * Returns a copy of the underlying Float32Array.
     */
    toFloat32Array(): Float32Array {
        return new Float32Array(this.values);
    }

    /**
     * Returns tensor values as regular numbers.
     */
    toArray(): number[] {
        return Array.from(this.values);
    }
}
