import { describe, expect, it } from 'vitest';
import { BackendRegistry, CPUBackend, Edrion, type EdrionBackend, type EdrionRuntimeInfo } from '@core';

describe('Edrion Runtime FEATURE-001', (): void => {
    it('Should start in created state', (): void => {
        const edrion: Edrion = new Edrion();

        expect(edrion.getStatus()).toEqual({
            state: 'created',
            ready: false,
        });
        expect(edrion.isInitialized()).toBe(false);
        expect(edrion.isReady()).toBe(false);
    });

    it('Should initialize CPU backend', async (): Promise<void> => {
        const edrion: Edrion = new Edrion();

        await edrion.initialize();

        expect(edrion.isReady()).toBe(true);
        expect(edrion.isInitialized()).toBe(true);
        expect(edrion.getBackend()?.name).toBe('cpu');
        expect(edrion.getMetrics().initializationCount).toBe(1);
    });

    it('Should shutdown runtime', async (): Promise<void> => {
        const edrion: Edrion = new Edrion();

        await edrion.initialize();
        await edrion.shutdown();

        expect(edrion.getStatus()).toEqual({
            state: 'stopped',
            ready: false,
        });
        expect(edrion.getBackend()).toBeUndefined();
        expect(edrion.getMetrics().shutdownCount).toBe(1);
    });

    it('Should expose runtime information', async (): Promise<void> => {
        const edrion: Edrion = new Edrion({
            runtime: {
                metadata: {
                    application: 'test-app',
                },
            },
        });

        await edrion.initialize();

        const information: EdrionRuntimeInfo = edrion.getRuntimeInfo();

        expect(information.version).toBe('0.6.0');
        expect(information.backend?.name).toBe('cpu');
        expect(information.executionContext.runtimeId.length).toBeGreaterThan(0);
        expect(information.executionContext.metadata['application']).toBe('test-app');
    });

    it('Should allow a custom backend registry', async (): Promise<void> => {
        const registry: BackendRegistry = new BackendRegistry();

        registry.register('custom', (): EdrionBackend => new CPUBackend());

        const edrion: Edrion = new Edrion(
            {
                backend: 'custom',
            },
            {
                backendRegistry: registry,
            },
        );

        await edrion.initialize();

        expect(edrion.isReady()).toBe(true);
    });

    it('Should remain idempotent when initialize is called twice', async (): Promise<void> => {
        const edrion: Edrion = new Edrion();

        await edrion.initialize();
        await edrion.initialize();

        expect(edrion.getMetrics().initializationCount).toBe(1);
    });
});
