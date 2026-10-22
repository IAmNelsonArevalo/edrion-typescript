import { describe, expect, it } from 'vitest';
import { CPUBackend } from '@core';

describe('CPUBackend', () => {
    it('should initialize', async () => {
        const backend = new CPUBackend();
        expect(backend.isInitialized()).toBe(false);

        await backend.initialize({
            runtimeId: 'test-runtime',
            observabilityEnabled: true,
            metadata: {},
        });

        expect(backend.isInitialized()).toBe(true);
        expect(backend.getInfo()).toEqual({
            name: 'cpu',
            device: 'cpu',
            initialized: true,
        });
    });

    it('should not initialize twice', async () => {
        const backend = new CPUBackend(),
            context = {
                runtimeId: 'test-runtime',
                observabilityEnabled: true,
                metadata: {},
            };

        await backend.initialize(context);
        await backend.initialize(context);
        expect(backend.isInitialized()).toBe(true);
    });

    it('should dispose backend resources', async () => {
        const backend = new CPUBackend();
        await backend.initialize({
            runtimeId: 'test-runtime',
            observabilityEnabled: true,
            metadata: {},
        });
        await backend.dispose();
        expect(backend.isInitialized()).toBe(false);
    });
});
