import { describe, expect, it } from 'vitest';
import { BackendRegistry, CPUBackend, createDefaultBackendRegistry } from '@core';

describe('BackendRegistry', () => {
    it('should register and create a backend', () => {
        const registry = new BackendRegistry();
        registry.register('cpu', () => new CPUBackend());
        expect(registry.has('cpu')).toBe(true);
        expect(registry.create('cpu')).toBeInstanceOf(CPUBackend);
    });

    it('should normalize backend names', () => {
        const registry = new BackendRegistry();
        registry.register('CPU', () => new CPUBackend());
        expect(registry.has(' cpu ')).toBe(true);
    });

    it('should reject duplicate backends', () => {
        const registry = new BackendRegistry();
        registry.register('cpu', () => new CPUBackend());
        expect(() => {
            registry.register('CPU', () => new CPUBackend());
        }).toThrow('Backend "cpu" is already registered.');
    });

    it('should reject unknown backends', () => {
        const registry = new BackendRegistry();
        expect(() => registry.create('unknown')).toThrow('Backend "unknown" is not registered.');
    });

    it('should reject empty backend names', () => {
        const registry = new BackendRegistry();
        expect(() => registry.register('   ', () => new CPUBackend())).toThrow('Backend name cannot be empty.');
    });

    it('should expose the default CPU backend', () => {
        const registry = createDefaultBackendRegistry();
        expect(registry.has('cpu')).toBe(true);
        expect(registry.list()).toContain('cpu');
    });
});
