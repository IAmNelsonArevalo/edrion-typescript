import { AutogradError, InvalidGradientError } from '@core/autograd';
import { DEFAULT_DTYPE, assertSupportedDType, type DType } from '@core/tensor/dtype';
import {
    binaryOperation,
    matmulOperation,
    meanOperation,
    powerOperation,
    reduceBroadcastGradient,
    sumOperation,
    transposeMatrix,
    unaryOperation,
    unravelTensorIndex,
    type ReductionResult,
} from '@core/tensor/operations';
import {
    calculateTensorSize,
    calculateTensorStrides,
    normalizeTensorShape,
    tensorShapesEqual,
    type TensorShape,
    type TensorStrides,
} from '@core/tensor/shape';
import { TensorStorage, type TensorData } from '@core/tensor/storage';
import { TensorDataError, TensorIndexError, TensorReshapeError, TensorShapeError } from '@core/tensor/tensor.errors';
import { buildNestedTensorArray, calculateTensorOffset, type TensorArray } from '@core/tensor/tensor.utils';

/**
 * Tensor construction options.
 */
export interface TensorOptions {
    readonly dtype?: DType;
    readonly backend?: string;
    readonly requiresGrad?: boolean;
}

/**
 * Value accepted by element-wise binary Tensor operations.
 */
export type TensorOperand = Tensor | number;

/**
 * Internal gradient propagation callback.
 */
type BackwardFunction = (gradient: Float32Array) => void;

/**
 * Edrion multidimensional numeric tensor.
 *
 * FEATURE-003 extends Tensor with mathematical operations
 * and reverse-mode automatic differentiation.
 */
export class Tensor {
    private readonly tensorShape: TensorShape;
    private readonly tensorStrides: TensorStrides;
    private readonly storage: TensorStorage;
    private readonly tensorDType: DType;
    private readonly tensorBackend: string;
    private readonly tensorSize: number;
    private readonly tensorRequiresGrad: boolean;
    private readonly parents: readonly Tensor[];
    private readonly backwardFunction: BackwardFunction | undefined;

    private gradientStorage: Float32Array | undefined = undefined;

    /**
     * Creates a Tensor.
     */
    public constructor(
        data: TensorData,
        shape: TensorShape,
        options: TensorOptions = {},
        parents: readonly Tensor[] = [],
        backwardFunction?: BackwardFunction,
    ) {
        const normalizedShape: TensorShape = normalizeTensorShape(shape);
        const size: number = calculateTensorSize(normalizedShape);
        const dtype: DType = options.dtype ?? DEFAULT_DTYPE;

        assertSupportedDType(dtype);

        const backend: string = normalizeBackendName(options.backend);

        if (data.length !== size) {
            throw new TensorDataError(
                `Tensor data length ${data.length} does not match shape [${normalizedShape.join(', ')}] with size ${size}.`,
            );
        }

        const requiresGrad: boolean = options.requiresGrad ?? false;

        if (!requiresGrad && parents.length > 0) {
            throw new AutogradError('A tensor that does not require gradients cannot own autograd parents.');
        }

        this.tensorShape = normalizedShape;
        this.tensorStrides = calculateTensorStrides(normalizedShape);
        this.tensorSize = size;
        this.tensorDType = dtype;
        this.tensorBackend = backend;
        this.tensorRequiresGrad = requiresGrad;
        this.parents = [...parents];
        this.backwardFunction = backwardFunction;
        this.storage = new TensorStorage(data, size);
    }

    public get shape(): TensorShape {
        return [...this.tensorShape];
    }

    public get strides(): TensorStrides {
        return [...this.tensorStrides];
    }

    public get rank(): number {
        return this.tensorShape.length;
    }

    public get size(): number {
        return this.tensorSize;
    }

    public get dtype(): DType {
        return this.tensorDType;
    }

    public get backend(): string {
        return this.tensorBackend;
    }

    public get isScalar(): boolean {
        return this.rank === 0;
    }

    /**
     * Indicates whether this tensor participates in autograd.
     */
    public get requiresGrad(): boolean {
        return this.tensorRequiresGrad;
    }

    /**
     * Returns the accumulated gradient.
     */
    public get grad(): Tensor | undefined {
        const gradient: Float32Array | undefined = this.gradientStorage;

        if (gradient === undefined) {
            return undefined;
        }

        return new Tensor(gradient, this.tensorShape, {
            dtype: this.tensorDType,
            backend: this.tensorBackend,
        });
    }

    /**
     * Reads a tensor value using multidimensional coordinates.
     */
    public get(...indices: number[]): number {
        const offset: number = calculateTensorOffset(indices, this.tensorShape, this.tensorStrides);

        return this.storage.get(offset);
    }

    /**
     * Updates a tensor value using multidimensional coordinates.
     */
    public set(indices: readonly number[], value: number): void {
        const offset: number = calculateTensorOffset(indices, this.tensorShape, this.tensorStrides);

        this.storage.set(offset, value);
    }

    /**
     * Element-wise addition with NumPy-style broadcasting.
     */
    public add(operand: TensorOperand): Tensor {
        const right: Tensor = this.resolveOperand(operand);
        this.assertCompatibleTensor(right);

        const operation = binaryOperation(
            this.toFloat32Array(),
            this.tensorShape,
            right.toFloat32Array(),
            right.tensorShape,
            'add',
        );

        const requiresGrad: boolean = this.tensorRequiresGrad || right.tensorRequiresGrad;

        const resultShape: TensorShape = operation.shape;

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (this.tensorRequiresGrad) {
                this.accumulateGradient(reduceBroadcastGradient(gradient, resultShape, this.tensorShape));
            }

            if (right.tensorRequiresGrad) {
                right.accumulateGradient(reduceBroadcastGradient(gradient, resultShape, right.tensorShape));
            }
        };

        const output: Tensor = this.createOperationTensor(
            operation.data,
            operation.shape,
            requiresGrad,
            [this, right],
            backward,
        );

        return output;
    }

    /**
     * Element-wise subtraction with broadcasting.
     */
    public sub(operand: TensorOperand): Tensor {
        const right: Tensor = this.resolveOperand(operand);
        this.assertCompatibleTensor(right);

        const operation = binaryOperation(
            this.toFloat32Array(),
            this.tensorShape,
            right.toFloat32Array(),
            right.tensorShape,
            'sub',
        );

        const requiresGrad: boolean = this.tensorRequiresGrad || right.tensorRequiresGrad;

        const resultShape: TensorShape = operation.shape;

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (this.tensorRequiresGrad) {
                this.accumulateGradient(reduceBroadcastGradient(gradient, resultShape, this.tensorShape));
            }

            if (right.tensorRequiresGrad) {
                const rightGradient: Float32Array = new Float32Array(gradient.length);

                for (let index: number = 0; index < gradient.length; index += 1) {
                    const value: number = readArrayValue(gradient, index);
                    rightGradient[index] = -value;
                }

                right.accumulateGradient(reduceBroadcastGradient(rightGradient, resultShape, right.tensorShape));
            }
        };

        const output: Tensor = this.createOperationTensor(
            operation.data,
            operation.shape,
            requiresGrad,
            [this, right],
            backward,
        );

        return output;
    }

    /**
     * Element-wise multiplication with broadcasting.
     */
    public mul(operand: TensorOperand): Tensor {
        const right: Tensor = this.resolveOperand(operand);
        this.assertCompatibleTensor(right);

        const operation = binaryOperation(
            this.toFloat32Array(),
            this.tensorShape,
            right.toFloat32Array(),
            right.tensorShape,
            'mul',
        );

        const requiresGrad: boolean = this.tensorRequiresGrad || right.tensorRequiresGrad;

        const leftData: Float32Array = this.toFloat32Array();
        const rightData: Float32Array = right.toFloat32Array();

        const resultShape: TensorShape = operation.shape;

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            const resultSize: number = calculateTensorSize(resultShape);
            const leftExpanded: Float32Array = expandBroadcastData(leftData, this.tensorShape, resultShape);
            const rightExpanded: Float32Array = expandBroadcastData(rightData, right.tensorShape, resultShape);

            if (this.tensorRequiresGrad) {
                const leftGradient: Float32Array = new Float32Array(resultSize);

                for (let index: number = 0; index < resultSize; index += 1) {
                    leftGradient[index] = readArrayValue(gradient, index) * readArrayValue(rightExpanded, index);
                }

                this.accumulateGradient(reduceBroadcastGradient(leftGradient, resultShape, this.tensorShape));
            }

            if (right.tensorRequiresGrad) {
                const rightGradient: Float32Array = new Float32Array(resultSize);

                for (let index: number = 0; index < resultSize; index += 1) {
                    rightGradient[index] = readArrayValue(gradient, index) * readArrayValue(leftExpanded, index);
                }

                right.accumulateGradient(reduceBroadcastGradient(rightGradient, resultShape, right.tensorShape));
            }
        };

        const output: Tensor = this.createOperationTensor(
            operation.data,
            operation.shape,
            requiresGrad,
            [this, right],
            backward,
        );

        return output;
    }

    /**
     * Element-wise division with broadcasting.
     */
    public div(operand: TensorOperand): Tensor {
        const right: Tensor = this.resolveOperand(operand);
        this.assertCompatibleTensor(right);

        const operation = binaryOperation(
            this.toFloat32Array(),
            this.tensorShape,
            right.toFloat32Array(),
            right.tensorShape,
            'div',
        );

        const requiresGrad: boolean = this.tensorRequiresGrad || right.tensorRequiresGrad;

        const leftData: Float32Array = this.toFloat32Array();
        const rightData: Float32Array = right.toFloat32Array();

        const resultShape: TensorShape = operation.shape;

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            const resultSize: number = calculateTensorSize(resultShape);
            const leftExpanded: Float32Array = expandBroadcastData(leftData, this.tensorShape, resultShape);
            const rightExpanded: Float32Array = expandBroadcastData(rightData, right.tensorShape, resultShape);

            if (this.tensorRequiresGrad) {
                const leftGradient: Float32Array = new Float32Array(resultSize);

                for (let index: number = 0; index < resultSize; index += 1) {
                    leftGradient[index] = readArrayValue(gradient, index) / readArrayValue(rightExpanded, index);
                }

                this.accumulateGradient(reduceBroadcastGradient(leftGradient, resultShape, this.tensorShape));
            }

            if (right.tensorRequiresGrad) {
                const rightGradient: Float32Array = new Float32Array(resultSize);

                for (let index: number = 0; index < resultSize; index += 1) {
                    const upstream: number = readArrayValue(gradient, index);
                    const leftValue: number = readArrayValue(leftExpanded, index);
                    const rightValue: number = readArrayValue(rightExpanded, index);

                    rightGradient[index] = (-upstream * leftValue) / (rightValue * rightValue);
                }

                right.accumulateGradient(reduceBroadcastGradient(rightGradient, resultShape, right.tensorShape));
            }
        };

        const output: Tensor = this.createOperationTensor(
            operation.data,
            operation.shape,
            requiresGrad,
            [this, right],
            backward,
        );

        return output;
    }

    /**
     * Raises every tensor element to a scalar exponent.
     */
    public pow(exponent: number): Tensor {
        const inputData: Float32Array = this.toFloat32Array();
        const resultData: Float32Array = powerOperation(inputData, exponent);
        const requiresGrad: boolean = this.tensorRequiresGrad;

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (!this.tensorRequiresGrad) {
                return;
            }

            const inputGradient: Float32Array = new Float32Array(this.tensorSize);

            for (let index: number = 0; index < this.tensorSize; index += 1) {
                const inputValue: number = readArrayValue(inputData, index);
                const upstream: number = readArrayValue(gradient, index);

                inputGradient[index] = upstream * exponent * Math.pow(inputValue, exponent - 1);
            }

            this.accumulateGradient(inputGradient);
        };

        const output: Tensor = this.createOperationTensor(resultData, this.tensorShape, requiresGrad, [this], backward);

        return output;
    }

    /**
     * Negates every tensor element.
     */
    public neg(): Tensor {
        return this.unaryWithDerivative('neg', (): number => -1);
    }

    /**
     * Applies the exponential function.
     */
    public exp(): Tensor {
        return this.unaryWithDerivative('exp', (_input: number, output: number): number => output);
    }

    /**
     * Applies the natural logarithm.
     */
    public log(): Tensor {
        return this.unaryWithDerivative('log', (input: number): number => 1 / input);
    }

    /**
     * Applies the square root.
     */
    public sqrt(): Tensor {
        return this.unaryWithDerivative('sqrt', (_input: number, output: number): number => 0.5 / output);
    }

    /**
     * Applies the Rectified Linear Unit activation element-wise.
     */
    public relu(): Tensor {
        const inputData: Float32Array = this.toFloat32Array();
        const resultData: Float32Array = new Float32Array(this.tensorSize);

        for (let index: number = 0; index < this.tensorSize; index += 1) {
            const value: number = readArrayValue(inputData, index);
            resultData[index] = value > 0 ? value : 0;
        }

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (!this.tensorRequiresGrad) {
                return;
            }

            const inputGradient: Float32Array = new Float32Array(this.tensorSize);

            for (let index: number = 0; index < this.tensorSize; index += 1) {
                const value: number = readArrayValue(inputData, index);
                const upstream: number = readArrayValue(gradient, index);

                inputGradient[index] = value > 0 ? upstream : 0;
            }

            this.accumulateGradient(inputGradient);
        };

        return this.createOperationTensor(resultData, this.tensorShape, this.tensorRequiresGrad, [this], backward);
    }

    /**
     * Reduces tensor values using sum.
     */
    public sum(axis?: number, keepDims: boolean = false): Tensor {
        return this.reduction('sum', axis, keepDims);
    }

    /**
     * Reduces tensor values using arithmetic mean.
     */
    public mean(axis?: number, keepDims: boolean = false): Tensor {
        return this.reduction('mean', axis, keepDims);
    }

    /**
     * Transposes a rank-2 matrix.
     */
    public transpose(): Tensor {
        if (this.rank !== 2) {
            throw new TensorShapeError(`transpose() currently requires a rank-2 tensor. Received rank ${this.rank}.`);
        }

        const rows: number = readShapeDimension(this.tensorShape, 0);
        const columns: number = readShapeDimension(this.tensorShape, 1);
        const resultData: Float32Array = transposeMatrix(this.toFloat32Array(), rows, columns);
        const resultShape: TensorShape = [columns, rows];

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (!this.tensorRequiresGrad) {
                return;
            }

            this.accumulateGradient(transposeMatrix(gradient, columns, rows));
        };

        return this.createOperationTensor(resultData, resultShape, this.tensorRequiresGrad, [this], backward);
    }

    /**
     * Matrix multiplication for rank-2 tensors.
     */
    public matmul(right: Tensor): Tensor {
        this.assertCompatibleTensor(right);

        if (this.rank !== 2 || right.rank !== 2) {
            throw new TensorShapeError(
                `matmul() currently requires two rank-2 tensors. Received ranks ${this.rank} and ${right.rank}.`,
            );
        }

        const leftRows: number = readShapeDimension(this.tensorShape, 0);
        const leftColumns: number = readShapeDimension(this.tensorShape, 1);
        const rightRows: number = readShapeDimension(right.tensorShape, 0);
        const rightColumns: number = readShapeDimension(right.tensorShape, 1);

        const leftData: Float32Array = this.toFloat32Array();
        const rightData: Float32Array = right.toFloat32Array();
        const resultData: Float32Array = matmulOperation(
            leftData,
            leftRows,
            leftColumns,
            rightData,
            rightRows,
            rightColumns,
        );

        const requiresGrad: boolean = this.tensorRequiresGrad || right.tensorRequiresGrad;

        const resultShape: TensorShape = [leftRows, rightColumns];

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (this.tensorRequiresGrad) {
                const rightTranspose: Float32Array = transposeMatrix(rightData, rightRows, rightColumns);

                const leftGradient: Float32Array = matmulOperation(
                    gradient,
                    leftRows,
                    rightColumns,
                    rightTranspose,
                    rightColumns,
                    rightRows,
                );

                this.accumulateGradient(leftGradient);
            }

            if (right.tensorRequiresGrad) {
                const leftTranspose: Float32Array = transposeMatrix(leftData, leftRows, leftColumns);

                const rightGradient: Float32Array = matmulOperation(
                    leftTranspose,
                    leftColumns,
                    leftRows,
                    gradient,
                    leftRows,
                    rightColumns,
                );

                right.accumulateGradient(rightGradient);
            }
        };

        return this.createOperationTensor(resultData, resultShape, requiresGrad, [this, right], backward);
    }

    /**
     * Creates an independent tensor with a compatible shape.
     */
    public reshape(newShape: TensorShape): Tensor {
        const normalizedShape: TensorShape = normalizeTensorShape(newShape);
        const newSize: number = calculateTensorSize(normalizedShape);

        if (newSize !== this.tensorSize) {
            throw new TensorReshapeError(
                `Cannot reshape tensor with size ${this.tensorSize} into shape [${normalizedShape.join(', ')}] with size ${newSize}.`,
            );
        }

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (this.tensorRequiresGrad) {
                this.accumulateGradient(new Float32Array(gradient));
            }
        };

        return this.createOperationTensor(
            this.storage.toFloat32Array(),
            normalizedShape,
            this.tensorRequiresGrad,
            [this],
            backward,
        );
    }

    /**
     * Creates an independent clone while preserving autograd connectivity.
     */
    public clone(): Tensor {
        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (this.tensorRequiresGrad) {
                this.accumulateGradient(new Float32Array(gradient));
            }
        };

        return this.createOperationTensor(
            this.storage.toFloat32Array(),
            this.tensorShape,
            this.tensorRequiresGrad,
            [this],
            backward,
        );
    }

    /**
     * Creates a tensor detached from the current autograd graph.
     */
    public detach(): Tensor {
        return new Tensor(this.storage.toFloat32Array(), this.tensorShape, {
            dtype: this.tensorDType,
            backend: this.tensorBackend,
            requiresGrad: false,
        });
    }

    /**
     * Clears the accumulated gradient for this tensor.
     */
    public zeroGrad(): void {
        this.gradientStorage = undefined;
    }

    /**
     * Executes reverse-mode automatic differentiation.
     *
     * A scalar output defaults to an upstream gradient of 1.
     * Non-scalar tensors require an explicit gradient.
     */
    public backward(gradient?: Tensor): void {
        if (!this.tensorRequiresGrad) {
            throw new AutogradError('Cannot call backward() on a tensor that does not require gradients.');
        }

        const initialGradient: Float32Array = this.resolveBackwardGradient(gradient);

        const topology: Tensor[] = [];
        const visited: Set<Tensor> = new Set<Tensor>();

        this.buildTopology(visited, topology);

        for (let index: number = 0; index < topology.length; index += 1) {
            const tensor: Tensor | undefined = topology[index];

            if (tensor === undefined) {
                throw new AutogradError(`Autograd topology is missing tensor at index ${index}.`);
            }

            if (tensor.parents.length > 0) {
                tensor.gradientStorage = undefined;
            }
        }

        this.accumulateGradient(initialGradient);

        for (let index: number = topology.length - 1; index >= 0; index -= 1) {
            const tensor: Tensor | undefined = topology[index];

            if (tensor === undefined) {
                throw new AutogradError(`Autograd topology is missing tensor at index ${index}.`);
            }

            const currentGradient: Float32Array | undefined = tensor.gradientStorage;

            if (currentGradient === undefined) {
                continue;
            }

            const backwardFunction: BackwardFunction | undefined = tensor.backwardFunction;

            if (backwardFunction !== undefined) {
                backwardFunction(new Float32Array(currentGradient));
            }
        }
    }

    /**
     * Returns the tensor as a nested JavaScript representation.
     */
    public toArray(): TensorArray {
        const values: number[] = this.storage.toArray();

        return buildNestedTensorArray(values, this.tensorShape);
    }

    /**
     * Returns tensor values as a flat JavaScript array.
     */
    public toFlatArray(): number[] {
        return this.storage.toArray();
    }

    /**
     * Returns tensor values as an independent Float32Array.
     */
    public toFloat32Array(): Float32Array {
        return this.storage.toFloat32Array();
    }

    private reduction(operation: 'sum' | 'mean', axis?: number, keepDims: boolean = false): Tensor {
        const result: ReductionResult =
            operation === 'sum'
                ? sumOperation(this.toFloat32Array(), this.tensorShape, axis, keepDims)
                : meanOperation(this.toFloat32Array(), this.tensorShape, axis, keepDims);

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (!this.tensorRequiresGrad) {
                return;
            }

            const expandedGradient: Float32Array = expandReductionGradient(
                gradient,
                result.shape,
                this.tensorShape,
                result.normalizedAxis,
                keepDims,
                result.reductionSize,
                operation === 'mean',
            );

            this.accumulateGradient(expandedGradient);
        };

        return this.createOperationTensor(result.data, result.shape, this.tensorRequiresGrad, [this], backward);
    }

    private unaryWithDerivative(
        operation: 'neg' | 'exp' | 'log' | 'sqrt',
        derivative: (input: number, output: number) => number,
    ): Tensor {
        const inputData: Float32Array = this.toFloat32Array();
        const resultData: Float32Array = unaryOperation(inputData, operation);

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (!this.tensorRequiresGrad) {
                return;
            }

            const inputGradient: Float32Array = new Float32Array(this.tensorSize);

            for (let index: number = 0; index < this.tensorSize; index += 1) {
                const inputValue: number = readArrayValue(inputData, index);
                const outputValue: number = readArrayValue(resultData, index);
                const upstream: number = readArrayValue(gradient, index);

                inputGradient[index] = upstream * derivative(inputValue, outputValue);
            }

            this.accumulateGradient(inputGradient);
        };

        return this.createOperationTensor(resultData, this.tensorShape, this.tensorRequiresGrad, [this], backward);
    }

    private createOperationTensor(
        data: TensorData,
        shape: TensorShape,
        requiresGrad: boolean,
        parents: readonly Tensor[],
        backwardFunction: BackwardFunction,
    ): Tensor {
        if (!requiresGrad) {
            return new Tensor(data, shape, {
                dtype: this.tensorDType,
                backend: this.tensorBackend,
            });
        }

        const autogradParents: Tensor[] = [];

        for (let index: number = 0; index < parents.length; index += 1) {
            const parent: Tensor | undefined = parents[index];

            if (parent === undefined) {
                throw new AutogradError(`Autograd parent at index ${index} does not exist.`);
            }

            if (parent.tensorRequiresGrad) {
                autogradParents.push(parent);
            }
        }

        return new Tensor(
            data,
            shape,
            {
                dtype: this.tensorDType,
                backend: this.tensorBackend,
                requiresGrad: true,
            },
            autogradParents,
            backwardFunction,
        );
    }

    private resolveOperand(operand: TensorOperand): Tensor {
        if (operand instanceof Tensor) {
            return operand;
        }

        return new Tensor(new Float32Array([operand]), [], {
            dtype: this.tensorDType,
            backend: this.tensorBackend,
        });
    }

    private assertCompatibleTensor(tensor: Tensor): void {
        if (tensor.tensorDType !== this.tensorDType) {
            throw new TensorDataError(
                `Tensor dtype mismatch: "${String(this.tensorDType)}" and "${String(tensor.tensorDType)}".`,
            );
        }

        if (tensor.tensorBackend !== this.tensorBackend) {
            throw new TensorDataError(
                `Tensor backend mismatch: "${this.tensorBackend}" and "${tensor.tensorBackend}".`,
            );
        }
    }

    private resolveBackwardGradient(gradient?: Tensor): Float32Array {
        if (gradient === undefined) {
            if (!this.isScalar) {
                throw new InvalidGradientError('backward() requires an explicit gradient for non-scalar tensors.');
            }

            return new Float32Array([1]);
        }

        if (!tensorShapesEqual(gradient.tensorShape, this.tensorShape)) {
            throw new InvalidGradientError(
                `Backward gradient shape [${gradient.tensorShape.join(', ')}] does not match tensor shape [${this.tensorShape.join(', ')}].`,
            );
        }

        if (gradient.tensorBackend !== this.tensorBackend) {
            throw new InvalidGradientError(
                `Backward gradient backend "${gradient.tensorBackend}" does not match tensor backend "${this.tensorBackend}".`,
            );
        }

        if (gradient.tensorDType !== this.tensorDType) {
            throw new InvalidGradientError(
                `Backward gradient dtype "${String(gradient.tensorDType)}" does not match tensor dtype "${String(this.tensorDType)}".`,
            );
        }

        return gradient.toFloat32Array();
    }

    private buildTopology(visited: Set<Tensor>, topology: Tensor[]): void {
        if (visited.has(this)) {
            return;
        }

        visited.add(this);

        for (let index: number = 0; index < this.parents.length; index += 1) {
            const parent: Tensor | undefined = this.parents[index];

            if (parent === undefined) {
                throw new AutogradError(`Autograd parent at index ${index} does not exist.`);
            }

            parent.buildTopology(visited, topology);
        }

        topology.push(this);
    }

    private accumulateGradient(gradient: Float32Array): void {
        if (!this.tensorRequiresGrad) {
            return;
        }

        if (gradient.length !== this.tensorSize) {
            throw new InvalidGradientError(
                `Gradient length ${gradient.length} does not match tensor size ${this.tensorSize}.`,
            );
        }

        const current: Float32Array | undefined = this.gradientStorage;

        if (current === undefined) {
            this.gradientStorage = new Float32Array(gradient);

            return;
        }

        for (let index: number = 0; index < current.length; index += 1) {
            const existingValue: number = readArrayValue(current, index);
            const incomingValue: number = readArrayValue(gradient, index);

            current[index] = existingValue + incomingValue;
        }
    }
}

function normalizeBackendName(backend: string | undefined): string {
    if (backend === undefined) {
        return 'cpu';
    }

    const normalizedBackend: string = backend.trim().toLowerCase();

    if (normalizedBackend.length === 0) {
        throw new TensorDataError('Tensor backend cannot be empty.');
    }

    return normalizedBackend;
}

function readArrayValue(data: Float32Array, index: number): number {
    const value: number | undefined = data[index];

    if (value === undefined) {
        throw new TensorIndexError(`Tensor value at flat offset ${index} does not exist.`);
    }

    return value;
}

function readShapeDimension(shape: TensorShape, axis: number): number {
    const dimension: number | undefined = shape[axis];

    if (dimension === undefined) {
        throw new TensorShapeError(`Tensor dimension at axis ${axis} does not exist.`);
    }

    return dimension;
}

function expandBroadcastData(data: Float32Array, sourceShape: TensorShape, resultShape: TensorShape): Float32Array {
    const sourceStrides: TensorStrides = calculateTensorStrides(sourceShape);
    const resultSize: number = calculateTensorSize(resultShape);
    const result: Float32Array = new Float32Array(resultSize);

    for (let offset: number = 0; offset < resultSize; offset += 1) {
        const resultIndices: number[] = unravelTensorIndex(offset, resultShape);

        const rankOffset: number = resultIndices.length - sourceShape.length;

        let sourceOffset: number = 0;

        for (let sourceAxis: number = 0; sourceAxis < sourceShape.length; sourceAxis += 1) {
            const dimension: number = readShapeDimension(sourceShape, sourceAxis);
            const stride: number | undefined = sourceStrides[sourceAxis];
            const resultIndex: number | undefined = resultIndices[rankOffset + sourceAxis];

            if (stride === undefined || resultIndex === undefined) {
                throw new TensorIndexError(`Unable to resolve broadcast value for source axis ${sourceAxis}.`);
            }

            const sourceIndex: number = dimension === 1 ? 0 : resultIndex;

            sourceOffset += sourceIndex * stride;
        }

        result[offset] = readArrayValue(data, sourceOffset);
    }

    return result;
}

function expandReductionGradient(
    gradient: Float32Array,
    gradientShape: TensorShape,
    inputShape: TensorShape,
    axis: number | undefined,
    keepDims: boolean,
    reductionSize: number,
    mean: boolean,
): Float32Array {
    const inputSize: number = calculateTensorSize(inputShape);
    const result: Float32Array = new Float32Array(inputSize);

    if (axis === undefined) {
        const scalarGradient: number = readArrayValue(gradient, 0);
        const value: number = mean ? scalarGradient / reductionSize : scalarGradient;

        result.fill(value);

        return result;
    }

    const gradientStrides: TensorStrides = calculateTensorStrides(gradientShape);

    for (let inputOffset: number = 0; inputOffset < inputSize; inputOffset += 1) {
        const inputIndices: number[] = unravelTensorIndex(inputOffset, inputShape);

        const gradientIndices: number[] = [];

        for (let currentAxis: number = 0; currentAxis < inputIndices.length; currentAxis += 1) {
            const index: number | undefined = inputIndices[currentAxis];

            if (index === undefined) {
                throw new TensorIndexError(`Missing input index at axis ${currentAxis}.`);
            }

            if (currentAxis === axis) {
                if (keepDims) {
                    gradientIndices.push(0);
                }

                continue;
            }

            gradientIndices.push(index);
        }

        let gradientOffset: number = 0;

        for (let gradientAxis: number = 0; gradientAxis < gradientIndices.length; gradientAxis += 1) {
            const index: number | undefined = gradientIndices[gradientAxis];
            const stride: number | undefined = gradientStrides[gradientAxis];

            if (index === undefined || stride === undefined) {
                throw new TensorIndexError(`Unable to resolve reduction gradient index at axis ${gradientAxis}.`);
            }

            gradientOffset += index * stride;
        }

        const upstream: number = readArrayValue(gradient, gradientOffset);

        result[inputOffset] = mean ? upstream / reductionSize : upstream;
    }

    return result;
}
