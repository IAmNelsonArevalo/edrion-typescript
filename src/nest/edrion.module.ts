import { type DynamicModule, Module, type Provider } from "@nestjs/common";

import { Edrion } from "../core/index.js";

import {
  EDRION_INSTANCE,
  EDRION_OPTIONS,
} from "./constants/edrion.constants.js";

import type {
  EdrionModuleAsyncOptions,
  EdrionModuleOptions,
  EdrionOptionsFactory,
} from "./interfaces/edrion-module-options.interface.js";

import { EdrionService } from "./edrion.service.js";

/**
 * Main NestJS integration module for Edrion.
 */
@Module({})
export class EdrionModule {
  /**
   * Registers Edrion using synchronous configuration.
   *
   * @example
   *
   * EdrionModule.forRoot({
   *   backend: 'cpu',
   *   observability: {
   *     enabled: true,
   *   },
   * })
   */
  static forRoot(options: EdrionModuleOptions = {}): DynamicModule {
    const optionsProvider: Provider = {
      provide: EDRION_OPTIONS,

      useValue: options,
    };

    const instanceProvider: Provider = {
      provide: EDRION_INSTANCE,

      useFactory: (resolvedOptions: EdrionModuleOptions): Edrion => {
        return new Edrion(resolvedOptions);
      },

      inject: [EDRION_OPTIONS],
    };

    return {
      module: EdrionModule,

      providers: [optionsProvider, instanceProvider, EdrionService],

      exports: [EDRION_INSTANCE, EdrionService],
    };
  }

  /**
   * Registers Edrion using asynchronous configuration.
   */
  static forRootAsync(options: EdrionModuleAsyncOptions): DynamicModule {
    const configurationProviders = this.createAsyncProviders(options);

    const instanceProvider: Provider = {
      provide: EDRION_INSTANCE,

      useFactory: (resolvedOptions: EdrionModuleOptions): Edrion => {
        return new Edrion(resolvedOptions);
      },

      inject: [EDRION_OPTIONS],
    };

    return {
      module: EdrionModule,

      imports: options.imports ?? [],

      providers: [...configurationProviders, instanceProvider, EdrionService],

      exports: [EDRION_INSTANCE, EdrionService],
    };
  }

  /**
   * Creates providers required for asynchronous configuration.
   */
  private static createAsyncProviders(
    options: EdrionModuleAsyncOptions,
  ): Provider[] {
    if (options.useFactory) {
      return [
        {
          provide: EDRION_OPTIONS,

          useFactory: options.useFactory,

          inject: options.inject ?? [],
        },
      ];
    }

    if (options.useClass) {
      const factoryClass = options.useClass;

      return [
        {
          provide: factoryClass,

          useClass: factoryClass,
        },

        {
          provide: EDRION_OPTIONS,

          useFactory: async (
            factory: EdrionOptionsFactory,
          ): Promise<EdrionModuleOptions> => {
            return factory.createEdrionOptions();
          },

          inject: [factoryClass],
        },
      ];
    }

    throw new Error(
      "EdrionModule.forRootAsync() requires useFactory or useClass.",
    );
  }
}
