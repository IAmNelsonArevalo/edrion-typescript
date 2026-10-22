/**
 * Backend identifier.
 *
 * It intentionally remains extensible so custom and future Edrion
 * backends can be registered without changing the public type.
 */
export type EdrionBackendName = string;

/**
 * Context provided to a backend when it is initialized.
 */
export interface BackendInitializationContext {
    readonly runtimeId: string;
    readonly initializedAt: Date;
}

/**
 * Public information exposed by an Edrion backend.
 */
export interface EdrionBackendInfo {
    readonly name: EdrionBackendName;
    readonly initialized: boolean;
}
