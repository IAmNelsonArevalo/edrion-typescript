/** Stable capability identifiers exposed by Edrion v1.4.0. */
export type EdrionCapability =
    | 'autograd'
    | 'backend:cpu'
    | 'tensor:batched-matmul'
    | 'transformer:batched'
    | 'data:dataset'
    | 'data:dataloader'
    | 'generation'
    | 'losses'
    | 'neural-network'
    | 'numerical-safety'
    | 'optimizers'
    | 'serialization'
    | 'tensor'
    | 'tokenizer'
    | 'tokenizer:bpe'
    | 'tokenizer:dataset'
    | 'transformer:rope'
    | 'transformer:rmsnorm'
    | 'transformer:swiglu'
    | 'transformer:modern-architecture'
    | 'training'
    | 'training:batched-data'
    | 'transformer';

/** Immutable capability report. */
export interface EdrionCapabilities {
    readonly capabilities: readonly EdrionCapability[];
}

const STABLE_CAPABILITIES: readonly EdrionCapability[] = Object.freeze([
    'tensor',
    'autograd',
    'backend:cpu',
    'tensor:batched-matmul',
    'transformer:batched',
    'data:dataset',
    'data:dataloader',
    'neural-network',
    'losses',
    'optimizers',
    'transformer',
    'tokenizer',
    'tokenizer:bpe',
    'tokenizer:dataset',
    'transformer:rope',
    'transformer:rmsnorm',
    'transformer:swiglu',
    'transformer:modern-architecture',
    'generation',
    'training',
    'training:batched-data',
    'serialization',
    'numerical-safety',
]);

/** Returns the stable capabilities available in Edrion v1.4.0. */
export function getEdrionCapabilities(): EdrionCapabilities {
    return {
        capabilities: [...STABLE_CAPABILITIES],
    };
}
