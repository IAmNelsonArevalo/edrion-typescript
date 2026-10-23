import type { EdrionBackend } from '@core/backend/backend.interface';
import { CPUBackend } from '@core/backend/cpu/cpu.backend';
import { WebGPUBackend } from '@core/backend/webgpu/webgpu.backend';

/** Factory capable of creating an Edrion backend. */
export type EdrionBackendFactory = () => EdrionBackend;

/**
 * Registry responsible for backend registration, discovery and creation.
 */
export class BackendRegistry {
    private readonly factories: Map<string, EdrionBackendFactory> = new Map<string, EdrionBackendFactory>();

    /** Registers a backend factory. */
    public register(name: string, factory: EdrionBackendFactory): void {
        const normalizedName: string = normalizeBackendName(name);

        if (this.factories.has(normalizedName)) {
            throw new Error(`Backend "${normalizedName}" is already registered.`);
        }

        this.factories.set(normalizedName, factory);
    }

    /** Indicates whether a backend is registered. */
    public has(name: string): boolean {
        const normalizedName: string = normalizeBackendName(name);

        return this.factories.has(normalizedName);
    }

    /** Creates a fresh backend instance. */
    public create(name: string): EdrionBackend {
        const normalizedName: string = normalizeBackendName(name);
        const factory: EdrionBackendFactory | undefined = this.factories.get(normalizedName);

        if (factory === undefined) {
            throw new Error(`Backend "${normalizedName}" is not registered.`);
        }

        return factory();
    }

    /** Returns every registered backend name. */
    public getRegisteredBackends(): readonly string[] {
        return Array.from(this.factories.keys());
    }
}

/** Creates the standard Edrion backend registry. */
export function createDefaultBackendRegistry(): BackendRegistry {
    const registry: BackendRegistry = new BackendRegistry();

    registry.register('cpu', (): EdrionBackend => new CPUBackend());
    registry.register('webgpu', (): EdrionBackend => new WebGPUBackend());

    return registry;
}

function normalizeBackendName(name: string): string {
    const normalizedName: string = name.trim().toLowerCase();

    if (normalizedName.length === 0) {
        throw new Error('Backend name cannot be empty.');
    }

    return normalizedName;
}
