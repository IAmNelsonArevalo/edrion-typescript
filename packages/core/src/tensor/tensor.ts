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
     * Swaps the final two dimensions of a rank-2 or rank-3 tensor.
     *
     * FEATURE-011 extends transpose() to batched matrices so a tensor
     * shaped [batch, rows, columns] becomes [batch, columns, rows].
     */
    public transpose(): Tensor {
        if (this.rank !== 2 && this.rank !== 3) {
            throw new TensorShapeError(`transpose() requires a rank-2 or rank-3 tensor. Received rank ${this.rank}.`);
        }

        if (this.rank === 2) {
            const rows: number = readShapeDimension(this.tensorShape, 0);
            const columns: number = readShapeDimension(this.tensorShape, 1);
            const resultData: Float32Array = transposeMatrix(this.toFloat32Array(), rows, columns);
            const resultShape: TensorShape = [columns, rows];

            const backward: BackwardFunction = (gradient: Float32Array): void => {
                if (this.tensorRequiresGrad) {
                    this.accumulateGradient(transposeMatrix(gradient, columns, rows));
                }
            };

            return this.createOperationTensor(resultData, resultShape, this.tensorRequiresGrad, [this], backward);
        }

        const batchSize: number = readShapeDimension(this.tensorShape, 0);
        const rows: number = readShapeDimension(this.tensorShape, 1);
        const columns: number = readShapeDimension(this.tensorShape, 2);
        const source: Float32Array = this.toFloat32Array();
        const resultData: Float32Array = new Float32Array(source.length);
        const sourceMatrixSize: number = rows * columns;
        const targetMatrixSize: number = columns * rows;

        for (let batch: number = 0; batch < batchSize; batch += 1) {
            const sourceOffset: number = batch * sourceMatrixSize;
            const targetOffset: number = batch * targetMatrixSize;
            const matrix: Float32Array = source.slice(sourceOffset, sourceOffset + sourceMatrixSize);
            resultData.set(transposeMatrix(matrix, rows, columns), targetOffset);
        }

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (!this.tensorRequiresGrad) {
                return;
            }

            const inputGradient: Float32Array = new Float32Array(gradient.length);

            for (let batch: number = 0; batch < batchSize; batch += 1) {
                const sourceOffset: number = batch * targetMatrixSize;
                const targetOffset: number = batch * sourceMatrixSize;
                const matrix: Float32Array = gradient.slice(sourceOffset, sourceOffset + targetMatrixSize);
                inputGradient.set(transposeMatrix(matrix, columns, rows), targetOffset);
            }

            this.accumulateGradient(inputGradient);
        };

        return this.createOperationTensor(
            resultData,
            [batchSize, columns, rows],
            this.tensorRequiresGrad,
            [this],
            backward,
        );
    }

    /**
     * Matrix multiplication for rank-2 matrices and rank-3 batched matrices.
     *
     * Supported FEATURE-011 shapes:
     * - [M, K] @ [K, N] -> [M, N]
     * - [B, M, K] @ [B, K, N] -> [B, M, N]
     */
    public matmul(right: Tensor): Tensor {
        this.assertCompatibleTensor(right);

        if (this.rank === 2 && right.rank === 2) {
            return this.matmulRank2(right);
        }

        if (this.rank !== 3 || right.rank !== 3) {
            throw new TensorShapeError(
                `matmul() requires rank-2 matrices or matching rank-3 batched matrices. Received ranks ${this.rank} and ${right.rank}.`,
            );
        }

        const batchSize: number = readShapeDimension(this.tensorShape, 0);
        const leftRows: number = readShapeDimension(this.tensorShape, 1);
        const leftColumns: number = readShapeDimension(this.tensorShape, 2);
        const rightBatchSize: number = readShapeDimension(right.tensorShape, 0);
        const rightRows: number = readShapeDimension(right.tensorShape, 1);
        const rightColumns: number = readShapeDimension(right.tensorShape, 2);

        if (batchSize !== rightBatchSize) {
            throw new TensorShapeError(
                `Batched matmul batch mismatch: left batch ${batchSize} and right batch ${rightBatchSize}.`,
            );
        }

        if (leftColumns !== rightRows) {
            throw new TensorShapeError(
                `Cannot multiply batched matrices with shapes [${this.tensorShape.join(', ')}] and [${right.tensorShape.join(', ')}].`,
            );
        }

        const leftData: Float32Array = this.toFloat32Array();
        const rightData: Float32Array = right.toFloat32Array();
        const leftMatrixSize: number = leftRows * leftColumns;
        const rightMatrixSize: number = rightRows * rightColumns;
        const outputMatrixSize: number = leftRows * rightColumns;
        const resultData: Float32Array = new Float32Array(batchSize * outputMatrixSize);

        for (let batch: number = 0; batch < batchSize; batch += 1) {
            const leftOffset: number = batch * leftMatrixSize;
            const rightOffset: number = batch * rightMatrixSize;
            const outputOffset: number = batch * outputMatrixSize;
            const batchResult: Float32Array = matmulOperation(
                leftData.slice(leftOffset, leftOffset + leftMatrixSize),
                leftRows,
                leftColumns,
                rightData.slice(rightOffset, rightOffset + rightMatrixSize),
                rightRows,
                rightColumns,
            );
            resultData.set(batchResult, outputOffset);
        }

        const requiresGrad: boolean = this.tensorRequiresGrad || right.tensorRequiresGrad;
        const backward: BackwardFunction = (gradient: Float32Array): void => {
            const leftGradient: Float32Array = new Float32Array(leftData.length);
            const rightGradient: Float32Array = new Float32Array(rightData.length);

            for (let batch: number = 0; batch < batchSize; batch += 1) {
                const leftOffset: number = batch * leftMatrixSize;
                const rightOffset: number = batch * rightMatrixSize;
                const gradientOffset: number = batch * outputMatrixSize;
                const leftMatrix: Float32Array = leftData.slice(leftOffset, leftOffset + leftMatrixSize);
                const rightMatrix: Float32Array = rightData.slice(rightOffset, rightOffset + rightMatrixSize);
                const gradientMatrix: Float32Array = gradient.slice(gradientOffset, gradientOffset + outputMatrixSize);

                if (this.tensorRequiresGrad) {
                    const rightTranspose: Float32Array = transposeMatrix(rightMatrix, rightRows, rightColumns);
                    leftGradient.set(
                        matmulOperation(
                            gradientMatrix,
                            leftRows,
                            rightColumns,
                            rightTranspose,
                            rightColumns,
                            rightRows,
                        ),
                        leftOffset,
                    );
                }

                if (right.tensorRequiresGrad) {
                    const leftTranspose: Float32Array = transposeMatrix(leftMatrix, leftRows, leftColumns);
                    rightGradient.set(
                        matmulOperation(leftTranspose, leftColumns, leftRows, gradientMatrix, leftRows, rightColumns),
                        rightOffset,
                    );
                }
            }

            if (this.tensorRequiresGrad) {
                this.accumulateGradient(leftGradient);
            }

            if (right.tensorRequiresGrad) {
                right.accumulateGradient(rightGradient);
            }
        };

        return this.createOperationTensor(
            resultData,
            [batchSize, leftRows, rightColumns],
            requiresGrad,
            [this, right],
            backward,
        );
    }

    private matmulRank2(right: Tensor): Tensor {
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
        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (this.tensorRequiresGrad) {
                const rightTranspose: Float32Array = transposeMatrix(rightData, rightRows, rightColumns);
                this.accumulateGradient(
                    matmulOperation(gradient, leftRows, rightColumns, rightTranspose, rightColumns, rightRows),
                );
            }

            if (right.tensorRequiresGrad) {
                const leftTranspose: Float32Array = transposeMatrix(leftData, leftRows, leftColumns);
                right.accumulateGradient(
                    matmulOperation(leftTranspose, leftColumns, leftRows, gradient, leftRows, rightColumns),
                );
            }
        };

        return this.createOperationTensor(resultData, [leftRows, rightColumns], requiresGrad, [this, right], backward);
    }

    /**
     * Extracts a contiguous range from the final tensor dimension.
     * Supports [rows, columns] and [batch, rows, columns].
     */
    public sliceColumns(startColumn: number, endColumn: number): Tensor {
        if (this.rank !== 2 && this.rank !== 3) {
            throw new TensorShapeError(`sliceColumns() requires rank-2 or rank-3 input. Received rank ${this.rank}.`);
        }

        const columns: number = readShapeDimension(this.tensorShape, this.rank - 1);

        if (!Number.isInteger(startColumn) || !Number.isInteger(endColumn)) {
            throw new TensorIndexError('sliceColumns() boundaries must be integers.');
        }

        if (startColumn < 0 || endColumn > columns || startColumn >= endColumn) {
            throw new TensorIndexError(
                `Invalid column range [${startColumn}, ${endColumn}) for tensor with ${columns} columns.`,
            );
        }

        const width: number = endColumn - startColumn;
        const rowCount: number = this.tensorSize / columns;
        const source: Float32Array = this.toFloat32Array();
        const result: Float32Array = new Float32Array(rowCount * width);

        for (let row: number = 0; row < rowCount; row += 1) {
            for (let column: number = 0; column < width; column += 1) {
                const sourceOffset: number = row * columns + startColumn + column;
                const targetOffset: number = row * width + column;
                result[targetOffset] = readArrayValue(source, sourceOffset);
            }
        }

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            if (!this.tensorRequiresGrad) {
                return;
            }

            const inputGradient: Float32Array = new Float32Array(this.tensorSize);

            for (let row: number = 0; row < rowCount; row += 1) {
                for (let column: number = 0; column < width; column += 1) {
                    const sourceOffset: number = row * width + column;
                    const targetOffset: number = row * columns + startColumn + column;
                    inputGradient[targetOffset] = readArrayValue(gradient, sourceOffset);
                }
            }

            this.accumulateGradient(inputGradient);
        };

        const resultShape: number[] = [...this.tensorShape];
        resultShape[resultShape.length - 1] = width;
        return this.createOperationTensor(result, resultShape, this.tensorRequiresGrad, [this], backward);
    }

    /**
     * Concatenates rank-2 or rank-3 tensors along their final dimension.
     */
    public static concatColumns(tensors: readonly Tensor[]): Tensor {
        if (tensors.length === 0) {
            throw new TensorShapeError('concatColumns() requires at least one tensor.');
        }

        const first: Tensor | undefined = tensors[0];

        if (first === undefined) {
            throw new TensorShapeError('concatColumns() could not resolve the first tensor.');
        }

        if (first.rank !== 2 && first.rank !== 3) {
            throw new TensorShapeError(
                `concatColumns() requires rank-2 or rank-3 tensors. Received rank ${first.rank}.`,
            );
        }

        const firstShape: readonly number[] = first.tensorShape;
        const firstColumns: number = readShapeDimension(firstShape, first.rank - 1);
        const rowCount: number = first.tensorSize / firstColumns;
        const leadingShape: number[] = firstShape.slice(0, -1);
        let totalColumns: number = 0;
        let requiresGrad: boolean = false;
        const widths: number[] = new Array<number>(tensors.length);

        for (let index: number = 0; index < tensors.length; index += 1) {
            const tensor: Tensor | undefined = tensors[index];

            if (tensor === undefined) {
                throw new TensorShapeError(`concatColumns() tensor at index ${index} does not exist.`);
            }

            first.assertCompatibleTensor(tensor);

            if (tensor.rank !== first.rank) {
                throw new TensorShapeError(
                    `concatColumns() rank mismatch: expected ${first.rank} but tensor ${index} has rank ${tensor.rank}.`,
                );
            }

            const tensorShape: readonly number[] = tensor.tensorShape;
            for (let axis: number = 0; axis < leadingShape.length; axis += 1) {
                if (tensorShape[axis] !== leadingShape[axis]) {
                    throw new TensorShapeError(
                        `concatColumns() leading shape mismatch at tensor ${index}, axis ${axis}.`,
                    );
                }
            }

            const tensorColumns: number = readShapeDimension(tensorShape, tensor.rank - 1);
            widths[index] = tensorColumns;
            totalColumns += tensorColumns;
            requiresGrad = requiresGrad || tensor.tensorRequiresGrad;
        }

        const result: Float32Array = new Float32Array(rowCount * totalColumns);

        for (let row: number = 0; row < rowCount; row += 1) {
            let outputColumn: number = 0;

            for (let tensorIndex: number = 0; tensorIndex < tensors.length; tensorIndex += 1) {
                const tensor: Tensor | undefined = tensors[tensorIndex];
                const width: number | undefined = widths[tensorIndex];

                if (tensor === undefined || width === undefined) {
                    throw new TensorShapeError(`concatColumns() missing tensor metadata at index ${tensorIndex}.`);
                }

                const data: Float32Array = tensor.toFloat32Array();

                for (let column: number = 0; column < width; column += 1) {
                    result[row * totalColumns + outputColumn] = readArrayValue(data, row * width + column);
                    outputColumn += 1;
                }
            }
        }

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            let startColumn: number = 0;

            for (let tensorIndex: number = 0; tensorIndex < tensors.length; tensorIndex += 1) {
                const tensor: Tensor | undefined = tensors[tensorIndex];
                const width: number | undefined = widths[tensorIndex];

                if (tensor === undefined || width === undefined) {
                    throw new AutogradError(`concatColumns() missing autograd metadata at index ${tensorIndex}.`);
                }

                if (tensor.tensorRequiresGrad) {
                    const tensorGradient: Float32Array = new Float32Array(rowCount * width);

                    for (let row: number = 0; row < rowCount; row += 1) {
                        for (let column: number = 0; column < width; column += 1) {
                            tensorGradient[row * width + column] = readArrayValue(
                                gradient,
                                row * totalColumns + startColumn + column,
                            );
                        }
                    }

                    tensor.accumulateGradient(tensorGradient);
                }

                startColumn += width;
            }
        };

        return first.createOperationTensor(result, [...leadingShape, totalColumns], requiresGrad, tensors, backward);
    }

    /**
     * Stacks equally-shaped tensors along a new leading dimension.
     *
     * Example: two [sequence] tensors become [batch, sequence].
     */
    public static stack(tensors: readonly Tensor[]): Tensor {
        if (tensors.length === 0) {
            throw new TensorShapeError('stack() requires at least one tensor.');
        }

        const first: Tensor | undefined = tensors[0];

        if (first === undefined) {
            throw new TensorShapeError('stack() could not resolve the first tensor.');
        }

        const sampleSize: number = first.tensorSize;
        const result: Float32Array = new Float32Array(sampleSize * tensors.length);
        let requiresGrad: boolean = false;

        for (let index: number = 0; index < tensors.length; index += 1) {
            const tensor: Tensor | undefined = tensors[index];

            if (tensor === undefined) {
                throw new TensorShapeError(`stack() tensor at index ${index} does not exist.`);
            }

            first.assertCompatibleTensor(tensor);

            if (!tensorShapesEqual(first.tensorShape, tensor.tensorShape)) {
                throw new TensorShapeError(
                    `stack() requires equal tensor shapes. Expected [${first.tensorShape.join(', ')}] but tensor ${index} has [${tensor.tensorShape.join(', ')}].`,
                );
            }

            result.set(tensor.toFloat32Array(), index * sampleSize);
            requiresGrad = requiresGrad || tensor.tensorRequiresGrad;
        }

        const backward: BackwardFunction = (gradient: Float32Array): void => {
            for (let index: number = 0; index < tensors.length; index += 1) {
                const tensor: Tensor | undefined = tensors[index];

                if (tensor === undefined) {
                    throw new AutogradError(`stack() missing autograd tensor at index ${index}.`);
                }

                if (tensor.tensorRequiresGrad) {
                    const start: number = index * sampleSize;
                    tensor.accumulateGradient(gradient.slice(start, start + sampleSize));
                }
            }
        };

        return first.createOperationTensor(
            result,
            [tensors.length, ...first.tensorShape],
            requiresGrad,
            tensors,
            backward,
        );
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
     * Scales the currently accumulated gradient in place.
     * Used by numerical-safety utilities such as global-norm clipping.
     */
    public scaleGradient(factor: number): void {
        if (!Number.isFinite(factor) || factor < 0) {
            throw new InvalidGradientError(
                `Gradient scale factor must be a finite number greater than or equal to zero. Received ${factor}.`,
            );
        }

        const gradient: Float32Array | undefined = this.gradientStorage;

        if (gradient === undefined) {
            return;
        }

        for (let index: number = 0; index < gradient.length; index += 1) {
            const value: number = readArrayValue(gradient, index);
            const scaledValue: number = value * factor;

            if (!Number.isFinite(scaledValue)) {
                throw new InvalidGradientError(`Scaled gradient became non-finite at flat offset ${index}.`);
            }

            gradient[index] = scaledValue;
        }
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
