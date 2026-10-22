/**
 * Computational backends supported by Edrion.
 *
 * Feature 000 only exposes the CPU identifier.
 * The actual computational backend abstraction
 * will be introduced in Feature 001.
 */
export type EdrionBackend = "cpu";

/**
 * Edrion observability configuration.
 */
export interface EdrionObservabilityOptions {
  /**
   * Enables runtime observability.
   */
  enabled?: boolean;
}

/**
 * Main public runtime configuration.
 */
export interface EdrionOptions {
  /**
   * Computational backend used by Edrion.
   */
  backend?: EdrionBackend;

  /**
   * Runtime observability options.
   */
  observability?: EdrionObservabilityOptions;
}

/**
 * Internal normalized runtime configuration.
 */
export interface NormalizedEdrionOptions {
  backend: EdrionBackend;

  observability: {
    enabled: boolean;
  };
}

/**
 * Default Edrion runtime configuration.
 */
export const DEFAULT_EDRION_OPTIONS: Readonly<NormalizedEdrionOptions> = {
  backend: "cpu",

  observability: {
    enabled: true,
  },
};

/**
 * Normalizes optional Edrion configuration.
 */
export function normalizeEdrionOptions(
  options: EdrionOptions = {},
): NormalizedEdrionOptions {
  return {
    backend: options.backend ?? DEFAULT_EDRION_OPTIONS.backend,
    observability: {
      enabled:
        options.observability?.enabled ??
        DEFAULT_EDRION_OPTIONS.observability.enabled,
    },
  };
}
