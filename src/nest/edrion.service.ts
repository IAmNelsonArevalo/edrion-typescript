import { Inject, Injectable } from "@nestjs/common";

import { Edrion, type EdrionRuntimeInfo } from "../core/index.js";

import { EDRION_INSTANCE } from "./constants/edrion.constants.js";

/**
 * NestJS facade around the core Edrion runtime.
 *
 * NestJS applications should inject this service
 * when interacting with Edrion.
 */
@Injectable()
export class EdrionService {
  /**
   * Creates the NestJS Edrion facade.
   */
  constructor(
    @Inject(EDRION_INSTANCE)
    private readonly runtime: Edrion,
  ) {}

  /**
   * Returns the underlying framework-agnostic runtime.
   */
  getRuntime(): Edrion {
    return this.runtime;
  }

  /**
   * Returns runtime metadata.
   */
  getRuntimeInfo(): EdrionRuntimeInfo {
    return this.runtime.getRuntimeInfo();
  }
}
