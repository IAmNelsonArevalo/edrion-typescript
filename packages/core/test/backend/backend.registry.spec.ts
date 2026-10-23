import { describe, expect, it } from 'vitest';
import { BackendRegistry, CPUBackend, createDefaultBackendRegistry, type EdrionBackend } from '@core';

describe('BackendRegistry', (): void => {
    it('Should register and create a backend', (): void => {
        const registry: BackendRegistry = new BackendRegistry();

        registry.register('custom', (): EdrionBackend => new CPUBackend());

        expect(registry.has('custom')).toBe(true);
        expect(registry.create('custom')).toBeInstanceOf(CPUBackend);
    });

    it('Should normalize backend names', (): void => {
        const registry: BackendRegistry = new BackendRegistry();

        registry.register(' Custom ', (): EdrionBackend => new CPUBackend());

        expect(registry.has('custom')).toBe(true);
        expect(registry.has('CUSTOM')).toBe(true);
    });

    it('Should reject duplicate backends', (): void => {
        const registry: BackendRegistry = new BackendRegistry();

        registry.register('cpu', (): EdrionBackend => new CPUBackend());

        expect((): void => {
            registry.register('CPU', (): EdrionBackend => new CPUBackend());
        }).toThrow('Backend "cpu" is already registered.');
    });

    it('Should provide CPU by default', (): void => {
        const registry: BackendRegistry = createDefaultBackendRegistry();

        expect(registry.has('cpu')).toBe(true);
        expect(registry.getRegisteredBackends()).toEqual(['cpu', 'webgpu']);
    });
});
