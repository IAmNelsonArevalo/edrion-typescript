import { WEBGPU_BUFFER_USAGE, WEBGPU_MAP_MODE_READ } from '@core/backend/webgpu/webgpu.constants';
import { MATRIX_MULTIPLY_WGSL, VECTOR_ADD_WGSL } from '@core/backend/webgpu/webgpu.kernels';
import { WebGPUBackend } from '@core/backend/webgpu/webgpu.backend';
import type { WebGPUBufferLike, WebGPUDeviceLike } from '@core/backend/webgpu/webgpu.types';

/** Async WebGPU kernel executor used by Edrion accelerated code paths. */
export class WebGPUKernelExecutor {
    private readonly backend: WebGPUBackend;

    public constructor(backend: WebGPUBackend) {
        this.backend = backend;
    }

    /** Executes element-wise Float32 vector addition on WebGPU. */
    public async add(left: Float32Array, right: Float32Array): Promise<Float32Array> {
        if (left.length !== right.length) {
            throw new RangeError(
                `WebGPU vector add requires equal lengths. Received ${left.length} and ${right.length}.`,
            );
        }
        if (left.length === 0) {
            return new Float32Array(0);
        }

        const device: WebGPUDeviceLike = this.backend.getDevice();
        const byteLength: number = left.byteLength;
        const leftBuffer: WebGPUBufferLike = createStorageBuffer(device, left);
        const rightBuffer: WebGPUBufferLike = createStorageBuffer(device, right);
        const outputBuffer: WebGPUBufferLike = device.createBuffer({
            size: byteLength,
            usage: WEBGPU_BUFFER_USAGE.STORAGE | WEBGPU_BUFFER_USAGE.COPY_SRC,
        });
        const readBuffer: WebGPUBufferLike = device.createBuffer({
            size: byteLength,
            usage: WEBGPU_BUFFER_USAGE.COPY_DST | WEBGPU_BUFFER_USAGE.MAP_READ,
        });

        try {
            const layout: unknown = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: 4, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: 4, buffer: { type: 'read-only-storage' } },
                    { binding: 2, visibility: 4, buffer: { type: 'storage' } },
                ],
            });
            const pipeline: unknown = device.createComputePipeline({
                layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
                compute: { module: device.createShaderModule({ code: VECTOR_ADD_WGSL }), entryPoint: 'main' },
            });
            const bindGroup: unknown = device.createBindGroup({
                layout,
                entries: [
                    { binding: 0, resource: { buffer: leftBuffer } },
                    { binding: 1, resource: { buffer: rightBuffer } },
                    { binding: 2, resource: { buffer: outputBuffer } },
                ],
            });
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(this.backend.calculateWorkgroups(left.length));
            pass.end();
            encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, byteLength);
            device.queue.submit([encoder.finish()]);
            const result: Float32Array = await readFloat32Buffer(readBuffer, left.length);
            this.backend.recordDispatch(left.byteLength + right.byteLength, result.byteLength);
            return result;
        } finally {
            leftBuffer.destroy();
            rightBuffer.destroy();
            outputBuffer.destroy();
            readBuffer.destroy();
        }
    }

    /** Executes rank-2 matrix multiplication on WebGPU. */
    public async matmul(
        left: Float32Array,
        leftRows: number,
        innerDimension: number,
        right: Float32Array,
        rightColumns: number,
    ): Promise<Float32Array> {
        validateMatrixDimensions(left, leftRows, innerDimension, right, rightColumns);
        if (leftRows === 0 || rightColumns === 0) {
            return new Float32Array(leftRows * rightColumns);
        }

        const device: WebGPUDeviceLike = this.backend.getDevice();
        const outputLength: number = leftRows * rightColumns;
        const outputByteLength: number = outputLength * Float32Array.BYTES_PER_ELEMENT;
        const leftBuffer: WebGPUBufferLike = createStorageBuffer(device, left);
        const rightBuffer: WebGPUBufferLike = createStorageBuffer(device, right);
        const outputBuffer: WebGPUBufferLike = device.createBuffer({
            size: outputByteLength,
            usage: WEBGPU_BUFFER_USAGE.STORAGE | WEBGPU_BUFFER_USAGE.COPY_SRC,
        });
        const dimensions: Uint32Array = new Uint32Array([leftRows, innerDimension, rightColumns, 0]);
        const dimensionsBuffer: WebGPUBufferLike = device.createBuffer({
            size: dimensions.byteLength,
            usage: 64 | WEBGPU_BUFFER_USAGE.COPY_DST,
        });
        device.queue.writeBuffer(dimensionsBuffer, 0, toGpuBufferSource(dimensions));
        const readBuffer: WebGPUBufferLike = device.createBuffer({
            size: outputByteLength,
            usage: WEBGPU_BUFFER_USAGE.COPY_DST | WEBGPU_BUFFER_USAGE.MAP_READ,
        });

        try {
            const layout: unknown = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: 4, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: 4, buffer: { type: 'read-only-storage' } },
                    { binding: 2, visibility: 4, buffer: { type: 'storage' } },
                    { binding: 3, visibility: 4, buffer: { type: 'uniform' } },
                ],
            });
            const pipeline: unknown = device.createComputePipeline({
                layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
                compute: { module: device.createShaderModule({ code: MATRIX_MULTIPLY_WGSL }), entryPoint: 'main' },
            });
            const bindGroup: unknown = device.createBindGroup({
                layout,
                entries: [
                    { binding: 0, resource: { buffer: leftBuffer } },
                    { binding: 1, resource: { buffer: rightBuffer } },
                    { binding: 2, resource: { buffer: outputBuffer } },
                    { binding: 3, resource: { buffer: dimensionsBuffer } },
                ],
            });
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(rightColumns / 8), Math.ceil(leftRows / 8));
            pass.end();
            encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputByteLength);
            device.queue.submit([encoder.finish()]);
            const result: Float32Array = await readFloat32Buffer(readBuffer, outputLength);
            this.backend.recordDispatch(left.byteLength + right.byteLength, result.byteLength);
            return result;
        } finally {
            leftBuffer.destroy();
            rightBuffer.destroy();
            outputBuffer.destroy();
            dimensionsBuffer.destroy();
            readBuffer.destroy();
        }
    }
}

function toGpuBufferSource(source: ArrayBufferView<ArrayBufferLike>): Uint8Array<ArrayBuffer> {
    const buffer: ArrayBuffer = new ArrayBuffer(source.byteLength);

    const target: Uint8Array<ArrayBuffer> = new Uint8Array(buffer);

    const sourceBytes: Uint8Array<ArrayBufferLike> = new Uint8Array(
        source.buffer,
        source.byteOffset,
        source.byteLength,
    );

    target.set(sourceBytes);

    return target;
}

function createStorageBuffer(device: WebGPUDeviceLike, values: Float32Array): WebGPUBufferLike {
    const buffer: WebGPUBufferLike = device.createBuffer({
        size: values.byteLength,
        usage: WEBGPU_BUFFER_USAGE.STORAGE | WEBGPU_BUFFER_USAGE.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, toGpuBufferSource(values));
    return buffer;
}

async function readFloat32Buffer(buffer: WebGPUBufferLike, length: number): Promise<Float32Array> {
    await buffer.mapAsync(WEBGPU_MAP_MODE_READ);
    const mapped: ArrayBuffer = buffer.getMappedRange();
    const copy: Float32Array = new Float32Array(length);
    copy.set(new Float32Array(mapped, 0, length));
    buffer.unmap();
    return copy;
}

function validateMatrixDimensions(
    left: Float32Array,
    leftRows: number,
    innerDimension: number,
    right: Float32Array,
    rightColumns: number,
): void {
    const dimensions: readonly number[] = [leftRows, innerDimension, rightColumns];
    for (let index: number = 0; index < dimensions.length; index += 1) {
        const value: number | undefined = dimensions[index];
        if (value === undefined || !Number.isInteger(value) || value <= 0) {
            throw new RangeError(
                `WebGPU matrix dimensions must be positive integers. Received: [${dimensions.join(', ')}].`,
            );
        }
    }
    if (left.length !== leftRows * innerDimension) {
        throw new RangeError('WebGPU left matrix data length does not match its declared dimensions.');
    }
    if (right.length !== innerDimension * rightColumns) {
        throw new RangeError('WebGPU right matrix data length does not match its declared dimensions.');
    }
}
