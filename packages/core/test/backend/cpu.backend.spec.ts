import { describe, expect, it } from 'vitest';
import { CPUBackend, type BackendInitializationContext } from '@core';

describe('CPUBackend', (): void => {
    it('Should initialize and dispose', async (): Promise<void> => {
        const backend: CPUBackend = new CPUBackend();
        const context: BackendInitializationContext = {
            runtimeId: 'runtime-test',
            initializedAt: new Date(),
        };

        await backend.initialize(context);

        expect(backend.isInitialized()).toBe(true);
        expect(backend.getInitializationContext()).toBe(context);
        expect(backend.getInfo()).toEqual({
            name: 'cpu',
            initialized: true,
        });

        await backend.dispose();

        expect(backend.isInitialized()).toBe(false);
        expect(backend.getInitializationContext()).toBeUndefined();
    });
});
