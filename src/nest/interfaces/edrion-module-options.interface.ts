import type { FactoryProvider, ModuleMetadata, Type } from '@nestjs/common';
import type { EdrionOptions } from '@core';

/**
 * Configuration accepted by EdrionModule.forRoot().
 */
export type EdrionModuleOptions = EdrionOptions;

/**
 * Factory contract used by class-based
 * asynchronous Edrion configuration.
 */
export interface EdrionOptionsFactory {
    createEdrionOptions(): EdrionModuleOptions | Promise<EdrionModuleOptions>;
}

/**
 * Configuration accepted by EdrionModule.forRootAsync().
 */
export interface EdrionModuleAsyncOptions extends Pick<ModuleMetadata, 'imports'> {
    /**
     * Factory function used to obtain Edrion configuration.
     */
    useFactory?: FactoryProvider<EdrionModuleOptions>['useFactory'];

    /**
     * Dependencies injected into useFactory.
     */
    inject?: FactoryProvider<EdrionModuleOptions>['inject'];

    /**
     * Class implementing EdrionOptionsFactory.
     */
    useClass?: Type<EdrionOptionsFactory>;
}
