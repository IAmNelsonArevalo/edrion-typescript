import { Module } from '@nestjs/common';
import { EdrionModule } from '@edrion/nestjs';

/** Real NestJS consumer used as the Edrion integration laboratory. */
@Module({
    imports: [
        EdrionModule.forRoot({
            backend: 'cpu',
        }),
    ],
})
export class AppModule {}
