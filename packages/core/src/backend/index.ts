export type { EdrionBackend } from './backend.interface';
export type { BackendInitializationContext, EdrionBackendInfo, EdrionBackendName } from './backend.types';
export { BackendRegistry, createDefaultBackendRegistry, type EdrionBackendFactory } from './backend.registry';
export { CPUBackend } from './cpu/cpu.backend';

export * from './webgpu';
