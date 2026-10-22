/** Stable capability identifiers exposed by Edrion v1.0.0. */
export type EdrionCapability =
    | 'autograd'
    | 'backend:cpu'
    | 'generation'
    | 'losses'
    | 'neural-network'
    | 'numerical-safety'
    | 'optimizers'
    | 'serialization'
    | 'tensor'
    | 'tokenizer'
    | 'training'
    | 'transformer';

/** Immutable capability report. */
export interface EdrionCapabilities {
    readonly capabilities: readonly EdrionCapability[];
}

const STABLE_CAPABILITIES: readonly EdrionCapability[] = Object.freeze([
    'tensor',
    'autograd',
    'backend:cpu',
    'neural-network',
    'losses',
    'optimizers',
    'transformer',
    'tokenizer',
    'generation',
    'training',
    'serialization',
    'numerical-safety',
]);

/** Returns the stable capabilities available in Edrion v1.0.0. */
export function getEdrionCapabilities(): EdrionCapabilities {
    return {
        capabilities: [...STABLE_CAPABILITIES],
    };
}
