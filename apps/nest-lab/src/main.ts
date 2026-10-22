import 'reflect-metadata';
import { NestFactory } from '@nestjs/core';
import { EdrionService } from '@edrion/nestjs';
import { AppModule } from './app.module';

/** Boots a real Nest application context and validates the Edrion lifecycle. */
async function bootstrap(): Promise<void> {
    const application = await NestFactory.createApplicationContext(AppModule, {
        logger: false,
    });
    const edrion: EdrionService = application.get(EdrionService);

    if (!edrion.isReady()) {
        throw new Error('Edrion did not reach the ready state inside NestJS.');
    }

    console.log(JSON.stringify(edrion.getHealthReport(), null, 2));
    await application.close();
}

void bootstrap();
