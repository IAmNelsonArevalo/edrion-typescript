import type { EdrionBackend } from '@core/backend/backend.interface';
import { CPUBackend } from '@core/backend/cpu/cpu.backend';

/**
 * Factory capable of creating an Edrion backend.
 */
export type EdrionBackendFactory = () => EdrionBackend;

/**
 * Registry containing all computational backends
 * available to an Edrion runtime.
 */
export class BackendRegistry {
    private readonly factories = new Map<string, EdrionBackendFactory>();

    /**
     * Registers a computational backend.
     */
    register(name: string, factory: EdrionBackendFactory): void {
        const normalizedName = this.normalizeName(name);
        if (this.factories.has(normalizedName)) throw new Error(`Backend "${normalizedName}" is already registered.`);
        this.factories.set(normalizedName, factory);
    }

    /**
     * Creates a backend instance.
     */
    create(name: string): EdrionBackend {
        const normalizedName = this.normalizeName(name),
            factory = this.factories.get(normalizedName);

        if (!factory) throw new Error(`Backend "${normalizedName}" is not registered.`);

        return factory();
    }

    /**
     * Indicates whether a backend is registered.
     */
    has(name: string): boolean {
        const normalizedName = this.normalizeName(name);
        return this.factories.has(normalizedName);
    }

    /**
     * Returns all registered backend names.
     */
    list(): readonly string[] {
        return [...this.factories.keys()];
    }

    /**
     * Normalizes a backend name.
     */
    private normalizeName(name: string): string {
        const normalizedName = name.trim().toLowerCase();

        if (!normalizedName) throw new Error('Backend name cannot be empty.');

        return normalizedName;
    }
}

/**
 * Creates Edrion's default backend registry.
 */
export function createDefaultBackendRegistry(): BackendRegistry {
    const registry = new BackendRegistry();
    registry.register('cpu', () => new CPUBackend());
    return registry;
}
