import type { FactoryProvider, ModuleMetadata, Type } from '@nestjs/common';
import type { EdrionOptions } from '@edrion/core';

/** NestJS Edrion module configuration. */
export type EdrionModuleOptions = EdrionOptions;

/** Factory contract used by asynchronous Edrion configuration. */
export interface EdrionOptionsFactory {
    createEdrionOptions(): Promise<EdrionModuleOptions> | EdrionModuleOptions;
}

/** Async Edrion module options. */
export interface EdrionModuleAsyncOptions extends Pick<ModuleMetadata, 'imports'> {
    readonly useFactory?: FactoryProvider<EdrionModuleOptions>['useFactory'];
    readonly inject?: FactoryProvider<EdrionModuleOptions>['inject'];
    readonly useClass?: Type<EdrionOptionsFactory>;
}
