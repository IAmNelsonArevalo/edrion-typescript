import { describe, expect, it } from 'vitest';
import { Edrion } from '@edrion/core';
import { EdrionService } from '@nest';

describe('EdrionService', (): void => {
    it('Should initialize Edrion', async (): Promise<void> => {
        const runtime: Edrion = new Edrion();
        const service: EdrionService = new EdrionService(runtime);

        await service.onModuleInit();

        expect(service.isInitialized()).toBe(true);
    });

    it('Should shutdown Edrion', async (): Promise<void> => {
        const runtime: Edrion = new Edrion();
        const service: EdrionService = new EdrionService(runtime);

        await service.onModuleInit();
        await service.onModuleDestroy();

        expect(service.isInitialized()).toBe(false);
    });
});
