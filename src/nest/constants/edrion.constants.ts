/**
 * NestJS injection token containing the
 * framework-agnostic Edrion runtime instance.
 */
export const EDRION_INSTANCE = Symbol("EDRION_INSTANCE");

/**
 * NestJS injection token containing
 * the Edrion module configuration.
 */
export const EDRION_OPTIONS = Symbol("EDRION_OPTIONS");
