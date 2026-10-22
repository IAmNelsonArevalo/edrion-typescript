import { describe, expect, it } from 'vitest';
import { BackendRegistry, Edrion, type EdrionBackend, type EdrionBackendInfo } from '@core';

describe('Edrion runtime', () => {
    it('should start in created state', () => {
        const edrion = new Edrion();
        expect(edrion.getStatus()).toBe('created');
        expect(edrion.isReady()).toBe(false);
    });

    it('should initialize the CPU backend', async () => {
        const edrion = new Edrion();
        await edrion.initialize();
        expect(edrion.getStatus()).toBe('ready');
        expect(edrion.isReady()).toBe(true);
        expect(edrion.getBackend()?.getInfo()).toEqual({
            name: 'cpu',
            device: 'cpu',
            initialized: true,
        });
    });

    it('should shutdown gracefully', async () => {
        const edrion = new Edrion();
        await edrion.initialize();
        await edrion.shutdown();
        expect(edrion.getStatus()).toBe('stopped');
        expect(edrion.isReady()).toBe(false);
    });

    it('should shutdown before initialization', async () => {
        const edrion = new Edrion();
        await edrion.shutdown();
        expect(edrion.getStatus()).toBe('stopped');
    });

    it('should not initialize twice when already ready', async () => {
        const edrion = new Edrion();
        await edrion.initialize();
        await edrion.initialize();
        expect(edrion.getMetrics().initializationCount).toBe(1);
    });

    it('should generate a unique runtime id', () => {
        const first = new Edrion();
        const second = new Edrion();
        expect(first.getExecutionContext().runtimeId).not.toBe(second.getExecutionContext().runtimeId);
    });

    it('should preserve runtime metadata', () => {
        const edrion = new Edrion({
            runtime: {
                metadata: {
                    application: 'language-teacher',
                    environment: 'test',
                },
            },
        });

        expect(edrion.getExecutionContext().metadata).toEqual({
            application: 'language-teacher',
            environment: 'test',
        });
    });

    it('should record lifecycle metrics', async () => {
        const edrion = new Edrion();
        await edrion.initialize();
        expect(edrion.getMetrics().initializationCount).toBe(1);
        expect(edrion.getMetrics().failureCount).toBe(0);
        await edrion.shutdown();
        expect(edrion.getMetrics().shutdownCount).toBe(1);
    });

    it('should expose runtime information', async () => {
        const edrion = new Edrion();
        await edrion.initialize();
        const info = edrion.getRuntimeInfo();
        expect(info.name).toBe('edrion-typescript');
        expect(info.version).toBe('0.2.0');
        expect(info.status).toBe('ready');
        expect(info.backend?.name).toBe('cpu');
    });

    it('should support custom backend registries', async () => {
        class TestBackend implements EdrionBackend {
            readonly name = 'test';
            private initialized = false;
            async initialize(): Promise<void> {
                this.initialized = true;
            }
            async dispose(): Promise<void> {
                this.initialized = false;
            }
            isInitialized(): boolean {
                return this.initialized;
            }
            getInfo(): EdrionBackendInfo {
                return {
                    name: this.name,
                    device: 'test-device',
                    initialized: this.initialized,
                };
            }
        }

        const registry = new BackendRegistry();
        registry.register('test', () => new TestBackend());
        const edrion = new Edrion(
            {
                backend: 'test',
            },
            {
                backendRegistry: registry,
            },
        );

        await edrion.initialize();
        expect(edrion.getBackend()?.name).toBe('test');
    });

    it('should enter failed state when backend initialization fails', async () => {
        class FailingBackend implements EdrionBackend {
            readonly name = 'failing';
            async initialize(): Promise<void> {
                throw new Error('Initialization failed.');
            }
            async dispose(): Promise<void> {}
            isInitialized(): boolean {
                return false;
            }
            getInfo(): EdrionBackendInfo {
                return {
                    name: this.name,
                    device: 'test',
                    initialized: false,
                };
            }
        }

        const registry = new BackendRegistry();
        registry.register('failing', () => new FailingBackend());
        const edrion = new Edrion(
            {
                backend: 'failing',
            },
            {
                backendRegistry: registry,
            },
        );

        await expect(edrion.initialize()).rejects.toThrow('Initialization failed.');
        expect(edrion.getStatus()).toBe('failed');
        expect(edrion.getMetrics().failureCount).toBe(1);
    });
});
