import { Module, type DynamicModule, type FactoryProvider, type Provider } from '@nestjs/common';
import { Edrion } from '@edrion/core';
import { EDRION_INSTANCE, EDRION_OPTIONS } from '@nest/constants/edrion.constants';
import { EdrionService } from '@nest/edrion.service';
import type {
    EdrionModuleAsyncOptions,
    EdrionModuleOptions,
    EdrionOptionsFactory,
} from '@nest/interfaces/edrion-module-options.interface';

/**
 * Official NestJS integration module for Edrion.
 */
@Module({})
export class EdrionModule {
    /** Registers Edrion synchronously. */
    public static forRoot(options: EdrionModuleOptions = {}): DynamicModule {
        const optionsProvider: Provider<EdrionModuleOptions> = {
            provide: EDRION_OPTIONS,
            useValue: options,
        };

        const instanceProvider: FactoryProvider<Edrion> = {
            provide: EDRION_INSTANCE,
            inject: [EDRION_OPTIONS],
            useFactory: (edrionOptions: EdrionModuleOptions): Edrion => {
                return new Edrion(edrionOptions);
            },
        };

        return {
            module: EdrionModule,
            providers: [optionsProvider, instanceProvider, EdrionService],
            exports: [EDRION_INSTANCE, EdrionService],
        };
    }

    /** Registers Edrion asynchronously. */
    public static forRootAsync(options: EdrionModuleAsyncOptions): DynamicModule {
        const asyncProviders: Provider[] = EdrionModule.createAsyncProviders(options);

        const instanceProvider: FactoryProvider<Edrion> = {
            provide: EDRION_INSTANCE,
            inject: [EDRION_OPTIONS],
            useFactory: (edrionOptions: EdrionModuleOptions): Edrion => {
                return new Edrion(edrionOptions);
            },
        };

        const dynamicModule: DynamicModule = {
            module: EdrionModule,
            providers: [...asyncProviders, instanceProvider, EdrionService],
            exports: [EDRION_INSTANCE, EdrionService],
        };

        if (options.imports !== undefined) {
            dynamicModule.imports = options.imports;
        }

        return dynamicModule;
    }

    private static createAsyncProviders(options: EdrionModuleAsyncOptions): Provider[] {
        if (options.useFactory !== undefined) {
            const provider: FactoryProvider<EdrionModuleOptions> = {
                provide: EDRION_OPTIONS,
                useFactory: options.useFactory,
            };

            if (options.inject !== undefined) {
                provider.inject = options.inject;
            }

            return [provider];
        }

        if (options.useClass !== undefined) {
            const factoryProvider: Provider<EdrionOptionsFactory> = {
                provide: options.useClass,
                useClass: options.useClass,
            };

            const optionsProvider: FactoryProvider<EdrionModuleOptions> = {
                provide: EDRION_OPTIONS,
                inject: [options.useClass],
                useFactory: async (factory: EdrionOptionsFactory): Promise<EdrionModuleOptions> => {
                    return factory.createEdrionOptions();
                },
            };

            return [factoryProvider, optionsProvider];
        }

        throw new Error('EdrionModule.forRootAsync() requires useFactory or useClass.');
    }
}
