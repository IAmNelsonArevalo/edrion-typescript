/** Stable capability identifiers exposed by Edrion v1.6.0. */
export type EdrionCapability =
    | 'autograd'
    | 'backend:cpu'
    | 'tensor:batched-matmul'
    | 'transformer:batched'
    | 'data:dataset'
    | 'data:dataloader'
    | 'generation'
    | 'generation:kv-cache'
    | 'generation:streaming'
    | 'generation:inference-metrics'
    | 'losses'
    | 'neural-network'
    | 'numerical-safety'
    | 'optimizers'
    | 'optimizers:adamw'
    | 'training:lr-schedulers'
    | 'training:gradient-accumulation'
    | 'training:evaluation'
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
    'optimizers:adamw',
    'training:lr-schedulers',
    'training:gradient-accumulation',
    'training:evaluation',
    'transformer',
    'tokenizer',
    'tokenizer:bpe',
    'tokenizer:dataset',
    'transformer:rope',
    'transformer:rmsnorm',
    'transformer:swiglu',
    'transformer:modern-architecture',
    'generation',
    'generation:kv-cache',
    'generation:streaming',
    'generation:inference-metrics',
    'training',
    'training:batched-data',
    'serialization',
    'numerical-safety',
]);

/** Returns the stable capabilities available in Edrion v1.6.0. */
export function getEdrionCapabilities(): EdrionCapabilities {
    return {
        capabilities: [...STABLE_CAPABILITIES],
    };
}
