import {
  type EdrionOptions,
  type NormalizedEdrionOptions,
  normalizeEdrionOptions,
} from "../config/edrion-options.js";

/**
 * Public metadata describing the current Edrion runtime.
 */
export interface EdrionRuntimeInfo {
  /**
   * Runtime name.
   */
  name: string;

  /**
   * Runtime package version.
   */
  version: string;

  /**
   * Active computational backend.
   */
  backend: string;

  /**
   * Indicates whether observability is enabled.
   */
  observabilityEnabled: boolean;
}

/**
 * Framework-agnostic Edrion runtime.
 *
 * NestJS, Express, Node.js and future adapters
 * must consume this class instead of implementing
 * Edrion-specific core behavior.
 */
export class Edrion {
  private readonly options: NormalizedEdrionOptions;

  /**
   * Creates a new Edrion runtime.
   */
  constructor(options: EdrionOptions = {}) {
    this.options = normalizeEdrionOptions(options);
  }

  /**
   * Returns the normalized runtime configuration.
   */
  getOptions(): Readonly<NormalizedEdrionOptions> {
    return this.options;
  }

  /**
   * Returns runtime information useful for
   * diagnostics and observability.
   */
  getRuntimeInfo(): EdrionRuntimeInfo {
    return {
      name: "edrion-typescript",

      version: "0.1.0",

      backend: this.options.backend,

      observabilityEnabled: this.options.observability.enabled,
    };
  }
}
