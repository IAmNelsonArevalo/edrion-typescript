/** Stable capability identifiers exposed by Edrion v2.0.0. */
export type EdrionCapability =
    | 'autograd'
    | 'backend:cpu'
    | 'backend:webgpu'
    | 'backend:webgpu-kernels'
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
    | 'precision:mixed'
    | 'precision:float16'
    | 'precision:bfloat16'
    | 'training:loss-scaling'
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
    | 'language-model:causal-lm'
    | 'language-model:pretraining'
    | 'language-model:perplexity'
    | 'language-model:generation'
    | 'large-model:sharded-checkpoints'
    | 'large-model:memory-planning'
    | 'large-model:profiling'
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
    'backend:webgpu',
    'backend:webgpu-kernels',
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
    'language-model:causal-lm',
    'language-model:pretraining',
    'language-model:perplexity',
    'language-model:generation',
    'large-model:sharded-checkpoints',
    'large-model:memory-planning',
    'large-model:profiling',
    'training',
    'training:batched-data',
    'serialization',
    'numerical-safety',
    'precision:mixed',
    'precision:float16',
    'precision:bfloat16',
    'training:loss-scaling',
]);

/** Returns the stable capabilities available in Edrion v2.0.0. */
export function getEdrionCapabilities(): EdrionCapabilities {
    return {
        capabilities: [...STABLE_CAPABILITIES],
    };
}
