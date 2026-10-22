import { describe, expect, it } from 'vitest';

import { Edrion } from '@core';

import { EdrionService } from '@nest';

describe('EdrionService', () => {
    it('should expose the core runtime', () => {
        const runtime = new Edrion(),
            service = new EdrionService(runtime);
        expect(service.getRuntime()).toBe(runtime);
    });

    it('should initialize Edrion on NestJS module initialization', async () => {
        const runtime = new Edrion(),
            service = new EdrionService(runtime);

        await service.onModuleInit();
        expect(service.isReady()).toBe(true);
        expect(service.getStatus()).toBe('ready');
    });

    it('should shutdown Edrion when NestJS module is destroyed', async () => {
        const runtime = new Edrion(),
            service = new EdrionService(runtime);

        await service.onModuleInit();
        await service.onModuleDestroy();
        expect(service.getStatus()).toBe('stopped');
    });

    it('should expose runtime information', async () => {
        const runtime = new Edrion(),
            service = new EdrionService(runtime);

        await service.onModuleInit();
        expect(service.getRuntimeInfo().backend?.name).toBe('cpu');
    });

    it('should expose runtime metrics', async () => {
        const runtime = new Edrion(),
            service = new EdrionService(runtime);
        await service.onModuleInit();
        expect(service.getMetrics().initializationCount).toBe(1);
    });
});
