# Edrion TypeScript — FEATURE-004 / v0.5.0

Edrion is a TypeScript-first AI and deep-learning framework designed around a framework-independent mathematical core.

FEATURE-004 introduces the first neural-network abstraction layer on top of the differentiable Tensor engine delivered in FEATURE-003.

## Current Version

`v0.5.0 — FEATURE-004: Neural Network Core`

## Included in FEATURE-004

### Module system

- `Module`
- recursive parameter registration
- child-module registration
- `parameters()`
- `namedParameters()`
- `zeroGrad()`
- recursive `train()` / `eval()` modes

### Trainable parameters

- `Parameter extends Tensor`
- `requiresGrad: true` by construction
- compatible with FEATURE-003 autograd

### Layers

- `Linear`
- `Embedding`
- `LayerNormalization`

### Activations

- `ReLU`
- `Sigmoid`
- `Tanh`
- `GELU`
- `Softmax`

### Tensor addition

FEATURE-004 adds an exact differentiable `Tensor.relu()` primitive so ReLU does not rely on an unstable approximation.

## Linear

```ts
import { Linear, Tensor } from '@edrion/core';

const layer: Linear = new Linear(4, 8);

const input: Tensor = new Tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 4], {
    requiresGrad: true,
});

const output: Tensor = layer.forward(input);
const loss: Tensor = output.mean();

loss.backward();

console.log(layer.weight.grad?.shape);
```

## Embedding

Embedding receives integer token IDs represented by a Tensor and returns dense vectors while preserving gradients for the embedding matrix.

```ts
import { Embedding, Tensor } from '@edrion/core';

const embedding: Embedding = new Embedding(1000, 128);

const tokenIds: Tensor = new Tensor([10, 24, 10, 99], [4]);

const vectors: Tensor = embedding.forward(tokenIds);

console.log(vectors.shape);
// [4, 128]
```

## Layer Normalization

```ts
import { LayerNormalization, Tensor } from '@edrion/core';

const normalization: LayerNormalization = new LayerNormalization(128);

const output: Tensor = normalization.forward(input);
```

Layer normalization currently normalizes the final tensor dimension and supports optional affine parameters.

## Activations

```ts
import { GELU, ReLU, Sigmoid, Softmax, Tanh } from '@edrion/core';
```

Example:

```ts
const activation: GELU = new GELU();
const output: Tensor = activation.forward(input);
```

## Module Parameters

```ts
const parameters = model.parameters();
const namedParameters = model.namedParameters();

model.zeroGrad();
model.train();
model.eval();
```

Nested modules are traversed recursively.

## Architecture

```text
@edrion/core
│
├── autograd
├── backend
├── metrics
├── runtime
├── tensor
│   ├── operations
│   └── autograd-enabled Tensor
│
└── nn
    ├── Module
    ├── Parameter
    ├── layers
    │   ├── Linear
    │   ├── Embedding
    │   └── LayerNormalization
    └── activations
        ├── ReLU
        ├── Sigmoid
        ├── Tanh
        ├── GELU
        └── Softmax
```

## Important Design Decisions

- Neural-network layers reuse Tensor operations and autograd instead of implementing a second gradient engine.
- `Linear` uses `matmul()` plus broadcasting from FEATURE-003.
- `Embedding` uses one-hot projection through `matmul()` so gradients naturally accumulate into its trainable weight matrix.
- `LayerNormalization` is composed from differentiable Tensor operations.
- Softmax numerical stabilization is intentionally reserved for FEATURE-009, where numerical-safety policies are introduced consistently across the framework.

## Branch

```bash
git switch -c EDR-TS/feature/004-neural-network-core
```

## Commit

```bash
git add .
git commit -m "feat(EDR-TS-004): add neural network core"
```

## Pull Request

```text
EDR-TS-004: Neural Network Core
```

## Quality Gate

```bash
pnpm install
pnpm typecheck
pnpm typecheck:test
pnpm lint
pnpm test
pnpm build
pnpm check
```

## Feature Roadmap

| Version | Feature                                          | Status  |
| ------- | ------------------------------------------------ | ------- |
| v0.1.0  | FEATURE-000 — NestJS / Monorepo Foundation       | ✅      |
| v0.2.0  | FEATURE-001 — Core Runtime + Backend Abstraction | ✅      |
| v0.3.0  | FEATURE-002 — Tensor Core                        | ✅      |
| v0.4.0  | FEATURE-003 — Tensor Operations + Autograd       | ✅      |
| v0.5.0  | FEATURE-004 — Neural Network Core                | ✅      |
| v0.6.0  | FEATURE-005 — Losses + Optimizers                | Next    |
| v0.7.0  | FEATURE-006 — Transformer Foundation             | Planned |
| v0.8.0  | FEATURE-007 — Tokenizer + Generation             | Planned |
| v0.9.0  | FEATURE-008 — Training + Serialization           | Planned |
| v0.10.x | FEATURE-009 — Stability Lab + Numerical Safety   | Planned |
| v1.0.0  | FEATURE-010 — Production Foundation              | Planned |

## Version

- `@edrion/core`: `0.5.0`
- `@edrion/nestjs`: `0.5.0`
- `edrion-typescript`: `0.5.0`
