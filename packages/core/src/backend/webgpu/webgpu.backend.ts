import type { EdrionBackend } from '@core/backend/backend.interface';
import type { BackendInitializationContext, EdrionBackendInfo, EdrionBackendName } from '@core/backend/backend.types';
import { WEBGPU_DEFAULT_WORKGROUP_SIZE } from '@core/backend/webgpu/webgpu.constants';
import { WebGPUUnavailableError } from '@core/backend/webgpu/webgpu.errors';
import type {
    WebGPUAdapterLike,
    WebGPUBackendMetrics,
    WebGPUBackendOptions,
    WebGPUDeviceLike,
    WebGPUNavigatorLike,
} from '@core/backend/webgpu/webgpu.types';

/** WebGPU execution backend for accelerated Edrion kernels. */
export class WebGPUBackend implements EdrionBackend {
    public readonly name: EdrionBackendName = 'webgpu';

    private readonly options: WebGPUBackendOptions;
    private initialized: boolean = false;
    private initializationContext: BackendInitializationContext | undefined = undefined;
    private adapter: WebGPUAdapterLike | undefined = undefined;
    private device: WebGPUDeviceLike | undefined = undefined;
    private dispatchCount: number = 0;
    private bytesUploaded: number = 0;
    private bytesDownloaded: number = 0;
    private initializedAt: Date | undefined = undefined;

    public constructor(options: WebGPUBackendOptions = {}) {
        this.options = options;
    }

    /** Detects WebGPU without initializing a device. */
    public static isSupported(navigatorOverride?: WebGPUNavigatorLike): boolean {
        const navigatorLike: WebGPUNavigatorLike | undefined = navigatorOverride ?? resolveNavigator();
        return navigatorLike?.gpu !== undefined;
    }

    /** Initializes the adapter and device. */
    public async initialize(context: BackendInitializationContext): Promise<void> {
        if (this.initialized) {
            return;
        }

        const navigatorLike: WebGPUNavigatorLike | undefined = this.options.navigator ?? resolveNavigator();
        const gpu = navigatorLike?.gpu;
        if (gpu === undefined) {
            throw new WebGPUUnavailableError();
        }

        const requestOptions: { powerPreference?: 'low-power' | 'high-performance' } = {};
        if (this.options.powerPreference !== undefined) {
            requestOptions.powerPreference = this.options.powerPreference;
        }

        const adapter: WebGPUAdapterLike | null = await gpu.requestAdapter(requestOptions);
        if (adapter === null) {
            throw new WebGPUUnavailableError();
        }

        const device: WebGPUDeviceLike = await adapter.requestDevice();
        this.adapter = adapter;
        this.device = device;
        this.initializationContext = context;
        this.initializedAt = new Date();
        this.initialized = true;
    }

    /** Releases device references and optional native resources. */
    public async dispose(): Promise<void> {
        if (!this.initialized) {
            return;
        }
        this.device?.destroy?.();
        this.adapter = undefined;
        this.device = undefined;
        this.initializationContext = undefined;
        this.initializedAt = undefined;
        this.initialized = false;
        await Promise.resolve();
    }

    public isInitialized(): boolean {
        return this.initialized;
    }

    public getInfo(): EdrionBackendInfo {
        return { name: this.name, initialized: this.initialized };
    }

    /** Returns the active WebGPU device. */
    public getDevice(): WebGPUDeviceLike {
        if (this.device === undefined) {
            throw new WebGPUUnavailableError();
        }
        return this.device;
    }

    /** Returns backend metrics used by production diagnostics. */
    public getMetrics(): WebGPUBackendMetrics {
        return {
            dispatchCount: this.dispatchCount,
            bytesUploaded: this.bytesUploaded,
            bytesDownloaded: this.bytesDownloaded,
            initializedAt: this.initializedAt,
        };
    }

    /** Records one accelerated kernel dispatch. */
    public recordDispatch(bytesUploaded: number, bytesDownloaded: number): void {
        this.dispatchCount += 1;
        this.bytesUploaded += bytesUploaded;
        this.bytesDownloaded += bytesDownloaded;
    }

    /** Calculates one-dimensional dispatch groups. */
    public calculateWorkgroups(elementCount: number, workgroupSize: number = WEBGPU_DEFAULT_WORKGROUP_SIZE): number {
        if (!Number.isInteger(elementCount) || elementCount < 0) {
            throw new RangeError(`WebGPU element count must be a non-negative integer. Received: ${elementCount}.`);
        }
        if (!Number.isInteger(workgroupSize) || workgroupSize <= 0) {
            throw new RangeError(`WebGPU workgroup size must be a positive integer. Received: ${workgroupSize}.`);
        }
        return Math.ceil(elementCount / workgroupSize);
    }
}

function resolveNavigator(): WebGPUNavigatorLike | undefined {
    const candidate: unknown = globalThis.navigator;
    return candidate as WebGPUNavigatorLike | undefined;
}
