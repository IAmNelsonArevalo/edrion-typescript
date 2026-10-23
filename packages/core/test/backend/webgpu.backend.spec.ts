import { describe, expect, it } from 'vitest';
import {
    WebGPUBackend,
    WebGPUUnavailableError,
    createDefaultBackendRegistry,
    type WebGPUDeviceLike,
    type WebGPUNavigatorLike,
} from '@core';

function createFakeDevice(): WebGPUDeviceLike {
    return {
        queue: {
            writeBuffer: (): void => undefined,
            submit: (): void => undefined,
        },
        createBuffer: (): never => {
            throw new Error('Kernel buffer creation is not expected during backend lifecycle tests.');
        },
        createShaderModule: (): never => {
            throw new Error('Kernel compilation is not expected during backend lifecycle tests.');
        },
        createBindGroupLayout: (): never => {
            throw new Error('Kernel layout creation is not expected during backend lifecycle tests.');
        },
        createPipelineLayout: (): never => {
            throw new Error('Kernel pipeline creation is not expected during backend lifecycle tests.');
        },
        createComputePipeline: (): never => {
            throw new Error('Kernel pipeline creation is not expected during backend lifecycle tests.');
        },
        createBindGroup: (): never => {
            throw new Error('Kernel binding is not expected during backend lifecycle tests.');
        },
        createCommandEncoder: (): never => {
            throw new Error('Kernel encoding is not expected during backend lifecycle tests.');
        },
    };
}

describe('WebGPUBackend FEATURE-017', (): void => {
    it('detects WebGPU support through the runtime navigator contract', (): void => {
        const navigatorLike: WebGPUNavigatorLike = {
            gpu: {
                requestAdapter: async () => null,
            },
        };

        expect(WebGPUBackend.isSupported(navigatorLike)).toBe(true);
        expect(WebGPUBackend.isSupported({})).toBe(false);
    });

    it('initializes and disposes a WebGPU device', async (): Promise<void> => {
        const device: WebGPUDeviceLike = createFakeDevice();
        const navigatorLike: WebGPUNavigatorLike = {
            gpu: {
                requestAdapter: async () => ({
                    requestDevice: async () => device,
                }),
            },
        };
        const backend: WebGPUBackend = new WebGPUBackend({ navigator: navigatorLike });

        await backend.initialize({ runtimeId: 'test', initializedAt: new Date() });
        expect(backend.isInitialized()).toBe(true);
        expect(backend.getDevice()).toBe(device);

        await backend.dispose();
        expect(backend.isInitialized()).toBe(false);
    });

    it('fails clearly when WebGPU is unavailable', async (): Promise<void> => {
        const backend: WebGPUBackend = new WebGPUBackend({ navigator: {} });

        await expect(backend.initialize({ runtimeId: 'test', initializedAt: new Date() })).rejects.toBeInstanceOf(
            WebGPUUnavailableError,
        );
    });

    it('calculates dispatch groups deterministically', (): void => {
        const backend: WebGPUBackend = new WebGPUBackend({ navigator: {} });

        expect(backend.calculateWorkgroups(0)).toBe(0);
        expect(backend.calculateWorkgroups(1)).toBe(1);
        expect(backend.calculateWorkgroups(64)).toBe(1);
        expect(backend.calculateWorkgroups(65)).toBe(2);
    });

    it('registers WebGPU in the standard backend registry', (): void => {
        expect(createDefaultBackendRegistry().getRegisteredBackends()).toEqual(['cpu', 'webgpu']);
    });
});
