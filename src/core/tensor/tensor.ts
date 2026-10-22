import { DEFAULT_DTYPE, assertSupportedDType, type DType } from '@core/tensor/dtype';
import { calculateTensorSize, calculateTensorStrides, normalizeTensorShape, type TensorShape } from '@core/tensor/shape';
import { TensorStorage, type TensorData } from '@core/tensor/storage';
import { TensorDataError, TensorReshapeError } from '@core/tensor/tensor.errors';
import { buildNestedTensorArray, calculateTensorOffset, type TensorArray } from '@core/tensor/tensor.utils';

/**
 * Tensor construction options.
 */
export interface TensorOptions {
    /**
     * Tensor data type.
     */
    dtype?: DType;

    /**
     * Backend identifier associated with this tensor.
     */
    backend?: string;
}

/**
 * Core multidimensional Edrion Tensor.
 *
 * Feature 002 provides tensor structure,
 * storage and indexing.
 *
 * Mathematical operations and automatic
 * differentiation are introduced later.
 */
export class Tensor {
    private readonly storage: TensorStorage;
    private readonly internalShape: number[];
    private readonly internalStrides: number[];
    private readonly internalDType: DType;
    private readonly internalBackend: string;

    /**
     * Creates a Tensor.
     */
    constructor(data: TensorData, shape: TensorShape, options: TensorOptions = {}) {
        const normalizedShape = normalizeTensorShape(shape),
            size = calculateTensorSize(normalizedShape),
            storage = new TensorStorage(data);

        if (storage.length !== size) throw new TensorDataError(`Tensor data length ${storage.length} does not match shape size ${size}.`);

        const dtype = options.dtype ?? DEFAULT_DTYPE;
        assertSupportedDType(dtype);

        const backend = options.backend?.trim().toLowerCase() ?? 'cpu';
        if (backend.length === 0) throw new TensorDataError('Tensor backend cannot be empty.');

        this.storage = storage;
        this.internalShape = normalizedShape;
        this.internalStrides = calculateTensorStrides(normalizedShape);
        this.internalDType = dtype;
        this.internalBackend = backend;
    }

    /**
     * Tensor shape.
     */
    get shape(): readonly number[] {
        return [...this.internalShape];
    }

    /**
     * Tensor strides.
     */
    get strides(): readonly number[] {
        return [...this.internalStrides];
    }

    /**
     * Number of tensor dimensions.
     */
    get rank(): number {
        return this.internalShape.length;
    }

    /**
     * Number of tensor elements.
     */
    get size(): number {
        return this.storage.length;
    }

    /**
     * Tensor data type.
     */
    get dtype(): DType {
        return this.internalDType;
    }

    /**
     * Backend identifier associated with this tensor.
     */
    get backend(): string {
        return this.internalBackend;
    }

    /**
     * Indicates whether this tensor represents a scalar.
     */
    get isScalar(): boolean {
        return this.rank === 0;
    }

    /**
     * Reads a tensor value.
     */
    get(...indices: number[]): number {
        const offset = calculateTensorOffset(this.internalShape, this.internalStrides, indices);
        return this.storage.get(offset);
    }

    /**
     * Updates a tensor value.
     */
    set(value: number, ...indices: number[]): void {
        const offset = calculateTensorOffset(this.internalShape, this.internalStrides, indices);
        this.storage.set(offset, value);
    }

    /**
     * Creates a Tensor with the same values
     * and a different shape.
     *
     * Feature 002 creates independent storage for
     * the reshaped Tensor. Tensor views can be
     * introduced later without changing the public API.
     */
    reshape(shape: TensorShape): Tensor {
        const normalizedShape = normalizeTensorShape(shape),
            newSize = calculateTensorSize(normalizedShape);

        if (newSize !== this.size)
            throw new TensorReshapeError(
                `Cannot reshape tensor of size ${this.size} into shape [${normalizedShape.join(', ')}] with size ${newSize}.`,
            );

        return new Tensor(this.storage.toFloat32Array(), normalizedShape, {
            dtype: this.dtype,
            backend: this.backend,
        });
    }

    /**
     * Creates an independent Tensor copy.
     */
    clone(): Tensor {
        return new Tensor(this.storage.toFloat32Array(), this.internalShape, {
            dtype: this.dtype,
            backend: this.backend,
        });
    }

    /**
     * Returns tensor values as a nested JavaScript array.
     */
    toArray(): TensorArray {
        return buildNestedTensorArray(this.storage.toArray(), this.internalShape);
    }

    /**
     * Returns tensor values as a flat number array.
     */
    toFlatArray(): number[] {
        return this.storage.toArray();
    }

    /**
     * Returns an independent Float32Array copy.
     */
    toFloat32Array(): Float32Array {
        return this.storage.toFloat32Array();
    }

    /**
     * Creates a tensor filled with zeros.
     */
    static zeros(shape: TensorShape, options: TensorOptions = {}): Tensor {
        const size = calculateTensorSize(shape);
        return new Tensor(new Float32Array(size), shape, options);
    }

    /**
     * Creates a tensor filled with ones.
     */
    static ones(shape: TensorShape, options: TensorOptions = {}): Tensor {
        return Tensor.full(shape, 1, options);
    }

    /**
     * Creates a tensor filled with a specific value.
     */
    static full(shape: TensorShape, value: number, options: TensorOptions = {}): Tensor {
        const size = calculateTensorSize(shape),
            data = new Float32Array(size);
        data.fill(value);
        return new Tensor(data, shape, options);
    }

    /**
     * Creates a scalar Tensor.
     */
    static scalar(value: number, options: TensorOptions = {}): Tensor {
        return new Tensor(new Float32Array([value]), [], options);
    }
}
