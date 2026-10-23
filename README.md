# Edrion TypeScript — v1.5.0

Edrion is a TypeScript-first deep-learning framework with its own Tensor engine, Autograd, neural-network modules, batched Transformer stack, Dataset/DataLoader pipeline, BPE tokenization, training, serialization and numerical-safety layer.

## Current Version

`v1.5.0 — FEATURE-015: Advanced Training Engine`

## Advanced Training Engine

FEATURE-015 adds the optimization and evaluation primitives required for serious language-model pretraining:

- AdamW with decoupled weight decay
- runtime-adjustable learning rates
- linear warmup, cosine decay, and warmup + cosine schedules
- gradient accumulation across real tensor batches
- optional global-norm gradient clipping
- validation/evaluation loops
- training, validation, learning-rate, and gradient-norm metrics

The existing Transformer, BPE tokenizer, Dataset/DataLoader and serialization stack remain available unchanged.

## Modern Transformer Foundation

FEATURE-014 moves Edrion closer to the architecture required by the future Edrion LM.

```text
Token IDs
   ↓
Token Embedding
   ↓
Modern Transformer Blocks × N
   ├─ RMSNorm
   ├─ Multi-Head Attention
   │    └─ RoPE on Q/K
   ├─ Residual
   ├─ RMSNorm
   ├─ SwiGLU
   └─ Residual
   ↓
Final RMSNorm
   ↓
Language Model Head
   ↓
Vocabulary Logits
```

### RoPE

Modern attention uses Rotary Position Embeddings instead of learned absolute positional embeddings. Edrion's RoPE implementation is differentiable and supports both:

```text
[sequence, headDimension]
[batch, sequence, headDimension]
```

### RMSNorm

Modern blocks replace LayerNorm with RMSNorm, using a trainable scaling parameter and normalization over the final feature dimension.

### SwiGLU

The modern feed-forward path is:

```text
gate = SiLU(Wgate(x))
up   = Wup(x)
out  = Wdown(gate * up)
```

### Compatibility

For v1.x backward compatibility, `TransformerModel` keeps the legacy default. Enable the new architecture explicitly with:

```ts
architecture: 'modern';
```

The previous path remains available as:

```ts
architecture: 'legacy';
```

## Production capabilities added in v1.4.0

```text
transformer:rope
transformer:rmsnorm
transformer:swiglu
transformer:modern-architecture
```

## Validation

```text
Core strict TypeScript typecheck    PASS
RoPE forward/backward               PASS
Batched RoPE                        PASS
RMSNorm                             PASS
SwiGLU                              PASS
Modern batched Transformer          PASS
Full Transformer backward           PASS
Legacy compatibility                PASS
```

See:

```text
FEATURE-014.md
FEATURE-014-VALIDATION.md
PR-FEATURE-014.md
```

## Packages

```text
@edrion/core        1.4.0
@edrion/nestjs      1.4.0
edrion-typescript   1.4.0
@edrion/nest-lab    1.4.0
```

## Branch

```bash
git switch -c EDR-TS/feature/014-modern-transformer
```

## Commit

```bash
git add .
git commit -m "feat(EDR-TS-014): modernize transformer architecture"
```

## Quality Gate

```bash
pnpm install
pnpm check
```

## Roadmap

| Version | Feature                                        | Status  |
| ------- | ---------------------------------------------- | ------- |
| v1.0.0  | FEATURE-010 — Production Foundation            | ✅      |
| v1.1.0  | FEATURE-011 — QA Fixes + Batched Tensor Engine | ✅      |
| v1.2.0  | FEATURE-012 — Dataset + DataLoader             | ✅      |
| v1.3.0  | FEATURE-013 — Advanced Tokenization / BPE      | ✅      |
| v1.4.0  | FEATURE-014 — Modern Transformer Architecture  | ✅      |
| v1.5.0  | FEATURE-015 — Advanced Training Engine         | Next    |
| v1.6.0  | FEATURE-016 — KV Cache + Streaming Generation  | Planned |
| v1.7.0  | FEATURE-017 — WebGPU Backend                   | Planned |
| v1.8.0  | FEATURE-018 — Mixed Precision                  | Planned |
| v1.9.0  | FEATURE-019 — Large Model Infrastructure       | Planned |
| v2.0.0  | Edrion Language Model Foundation               | Planned |

# Edrion Nest Lab

Executable NestJS consumer used by Edrion's integration QA.

It validates that `@edrion/nestjs` works outside its own package boundary, including dependency injection, module lifecycle, runtime initialization, health reporting and shutdown.

```bash
pnpm --filter @edrion/nest-lab test
pnpm --filter @edrion/nest-lab build
```

## Production capabilities added in v1.5.0

```text
optimizers:adamw
training:lr-schedulers
training:gradient-accumulation
training:evaluation
```
