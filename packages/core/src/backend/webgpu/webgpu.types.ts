/** Minimal WebGPU structural types used by Edrion without a hard dependency on @webgpu/types. */
export interface WebGPUBufferLike {
    readonly size: number;
    mapAsync(mode: number): Promise<void>;
    getMappedRange(): ArrayBuffer;
    unmap(): void;
    destroy(): void;
}

export interface WebGPUQueueLike {
    writeBuffer(buffer: WebGPUBufferLike, bufferOffset: number, data: BufferSource): void;
    submit(commandBuffers: readonly unknown[]): void;
}

export interface WebGPUComputePassLike {
    setPipeline(pipeline: unknown): void;
    setBindGroup(index: number, bindGroup: unknown): void;
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    end(): void;
}

export interface WebGPUCommandEncoderLike {
    beginComputePass(): WebGPUComputePassLike;
    copyBufferToBuffer(source: WebGPUBufferLike, sourceOffset: number, destination: WebGPUBufferLike, destinationOffset: number, size: number): void;
    finish(): unknown;
}

export interface WebGPUDeviceLike {
    readonly queue: WebGPUQueueLike;
    readonly lost?: Promise<unknown>;
    createBuffer(descriptor: { readonly size: number; readonly usage: number; readonly mappedAtCreation?: boolean }): WebGPUBufferLike;
    createShaderModule(descriptor: { readonly code: string }): unknown;
    createBindGroupLayout(descriptor: unknown): unknown;
    createPipelineLayout(descriptor: unknown): unknown;
    createComputePipeline(descriptor: unknown): unknown;
    createBindGroup(descriptor: unknown): unknown;
    createCommandEncoder(): WebGPUCommandEncoderLike;
    destroy?(): void;
}

export interface WebGPUAdapterLike {
    requestDevice(): Promise<WebGPUDeviceLike>;
}

export interface WebGPUNavigatorLike {
    readonly gpu?: {
        requestAdapter(options?: { readonly powerPreference?: 'low-power' | 'high-performance' }): Promise<WebGPUAdapterLike | null>;
    };
}

export interface WebGPUBackendOptions {
    readonly powerPreference?: 'low-power' | 'high-performance';
    readonly navigator?: WebGPUNavigatorLike;
}

export interface WebGPUBackendMetrics {
    readonly dispatchCount: number;
    readonly bytesUploaded: number;
    readonly bytesDownloaded: number;
    readonly initializedAt: Date | undefined;
}
