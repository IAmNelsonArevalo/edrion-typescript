import type { ModuleMetadata, Type } from "@nestjs/common";

import type { EdrionOptions } from "../../core/index.js";

/**
 * Configuration accepted by EdrionModule.forRoot().
 */
export type EdrionModuleOptions = EdrionOptions;

/**
 * Factory contract for class-based asynchronous configuration.
 */
export interface EdrionOptionsFactory {
  createEdrionOptions(): EdrionModuleOptions | Promise<EdrionModuleOptions>;
}

/**
 * Configuration accepted by EdrionModule.forRootAsync().
 */
export interface EdrionModuleAsyncOptions extends Pick<
  ModuleMetadata,
  "imports"
> {
  /**
   * Factory function used to resolve Edrion configuration.
   */
  useFactory?: (
    ...args: never[]
  ) => EdrionModuleOptions | Promise<EdrionModuleOptions>;

  /**
   * Dependencies injected into useFactory.
   */
  inject?: Array<string | symbol | Type<unknown>>;

  /**
   * Class implementing EdrionOptionsFactory.
   */
  useClass?: Type<EdrionOptionsFactory>;
}
