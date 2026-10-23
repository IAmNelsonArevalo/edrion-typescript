# Edrion TypeScript — v1.1.0

Edrion is a TypeScript-first deep-learning framework with its own Tensor engine, Autograd, neural-network modules, Transformer stack, training pipeline, serialization and numerical-safety layer.

## Current Version

`v1.1.0 — FEATURE-011: QA Fixes + Batched Tensor Engine`

## FEATURE-011 goals

FEATURE-011 closes the two QA gaps detected in v1.0.0 and removes one of the largest architectural limits in the first stable release: Transformer execution was restricted to one sequence at a time.

### QA fixes

- pins an exact Corepack-compatible pnpm version with `packageManager: "pnpm@10.0.0"`;
- adds `scripts/verify-package-manager.mjs` and makes it part of `pnpm check`;
- converts `apps/nest-lab` from documentation-only scaffolding into a real executable NestJS consumer;
- adds a NestJS integration test covering DI, initialization, health reporting and shutdown.

### Batched Tensor Engine

`Tensor.matmul()` now supports:

```text
[M, K] @ [K, N]             -> [M, N]
[B, M, K] @ [B, K, N]       -> [B, M, N]
```

Batched matmul remains connected to Autograd and computes gradients for both operands.

`Tensor.transpose()` now supports rank-3 batched matrices by swapping their final two dimensions.

`sliceColumns()` and `concatColumns()` now support both rank-2 and rank-3 tensors, preserving Autograd connectivity.

### Batched neural-network execution

`Linear` now accepts any rank greater than or equal to 1 as long as the final dimension matches `inFeatures`. Leading dimensions are preserved.

Examples:

```text
[features]                    -> [outFeatures]
[sequence, features]          -> [sequence, outFeatures]
[batch, sequence, features]   -> [batch, sequence, outFeatures]
```

### Batched Transformer

The Transformer can now execute:

```text
[sequence]
```

and:

```text
[batch, sequence]
```

Token input:

```text
[batch, sequence]
```

becomes:

```text
[batch, sequence, embedding]
```

and finally:

```text
[batch, sequence, vocabulary]
```

Multi-head attention and scaled dot-product attention also accept batched hidden states.

## Production capabilities

v1.1.0 exposes two additional runtime capabilities:

```text
tensor:batched-matmul
transformer:batched
```

## Validation performed for FEATURE-011

The FEATURE-011 snapshot was independently compiled with strict TypeScript settings and exercised with a runtime harness that verified:

- rank-3 batched matmul values;
- batched matmul gradients;
- rank-3 transpose;
- batched multi-head attention;
- gradient propagation through batched attention;
- Transformer logits with shape `[batch, sequence, vocabulary]`;
- v1.1.0 production capability metadata.

The independent runtime harness returned `PASS`.

## Packages

```text
@edrion/core        1.1.0
@edrion/nestjs      1.1.0
edrion-typescript   1.1.0
@edrion/nest-lab    1.1.0
```

## Branch

```bash
git switch -c EDR-TS/feature/011-batched-tensor-engine
```

## Commit

```bash
git add .
git commit -m "feat(EDR-TS-011): add batched tensor engine and resolve v1 QA gaps"
```

## Quality Gate

```bash
pnpm install
pnpm check
```

`pnpm check` now includes package-manager reproducibility verification and the real NestJS consumer integration suite.

## Roadmap

| Version | Feature                                        | Status  |
| ------- | ---------------------------------------------- | ------- |
| v1.0.0  | FEATURE-010 — Production Foundation            | ✅      |
| v1.1.0  | FEATURE-011 — QA Fixes + Batched Tensor Engine | ✅      |
| v1.2.0  | FEATURE-012 — Dataset + DataLoader             | Next    |
| v1.3.0  | FEATURE-013 — Advanced Tokenization / BPE      | Planned |
| v1.4.0  | FEATURE-014 — Modern Transformer Architecture  | Planned |
| v1.5.0  | FEATURE-015 — Advanced Training Engine         | Planned |
| v1.6.0  | FEATURE-016 — KV Cache + Streaming Generation  | Planned |
| v1.7.0  | FEATURE-017 — WebGPU Backend                   | Planned |
| v1.8.0  | FEATURE-018 — Mixed Precision                  | Planned |
| v1.9.0  | FEATURE-019 — Large Model Infrastructure       | Planned |
| v2.0.0  | Edrion Language Model Foundation               | Planned |
