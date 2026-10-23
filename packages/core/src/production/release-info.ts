/** Stable Edrion framework version exposed by the public API. */
export const EDRION_VERSION: string = '1.4.0';

/** Release channel used by this build. */
export type EdrionReleaseChannel = 'stable';

/** Immutable release information for the running framework. */
export interface EdrionReleaseInfo {
    readonly version: string;
    readonly channel: EdrionReleaseChannel;
    readonly apiStability: 'stable';
}

/** Returns immutable information describing this Edrion release. */
export function getEdrionReleaseInfo(): EdrionReleaseInfo {
    return {
        version: EDRION_VERSION,
        channel: 'stable',
        apiStability: 'stable',
    };
}
