import 'reflect-metadata';
import { NestFactory, type INestApplicationContext } from '@nestjs/core';
import { EdrionService } from '@edrion/nestjs';
import { afterEach, describe, expect, it } from 'vitest';
import { AppModule } from '../src/app.module';

describe('NestJS consumer integration', (): void => {
    let application: INestApplicationContext | undefined;

    afterEach(async (): Promise<void> => {
        if (application !== undefined) {
            await application.close();
            application = undefined;
        }
    });

    it('boots Edrion, exposes diagnostics and shuts down through the Nest lifecycle', async (): Promise<void> => {
        application = await NestFactory.createApplicationContext(AppModule, { logger: false });
        const edrion: EdrionService = application.get(EdrionService);

        expect(edrion.isReady()).toBe(true);
        expect(edrion.getVersion()).toBe('1.6.0');

        await application.close();
        application = undefined;

        expect(edrion.getStatus().state).toBe('stopped');
    });
});
