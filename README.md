<div align="center">

# EDRION

## TypeScript

### AI runtime and Deep Learning foundations for the Edrion ecosystem

**Building intelligence from the core.**

</div>

---

# About Edrion

**Edrion** is an artificial intelligence ecosystem focused on building the foundations required to create, train, evaluate, specialize and execute intelligent models.

The project is designed around the idea that modern artificial intelligence should not be restricted to a single programming language, framework or runtime.

`edrion-typescript` brings the Edrion ecosystem to TypeScript through a strongly typed, framework-independent AI runtime designed to evolve toward:

- Tensor computation
- Automatic differentiation
- Deep Learning
- Neural Networks
- Transformers
- Tokenization
- Training
- Inference
- Model serialization
- Metrics and observability
- Language models
- Multimodal AI
- WebAssembly
- WebGPU

NestJS is currently the first officially supported framework integration.

---

# Edrion Ecosystem

```text
Edrion
│
├── edrion-go
│   └── Go AI / Deep Learning Framework
│
├── edrion-typescript
│   └── TypeScript AI Runtime
│
├── edrion-core-go-api
│   └── Edrion Model Platform API
│
└── Future Products
    ├── Language Models
    ├── Domain Models
    ├── Language Teacher
    └── Multimodal Applications
```

The long-term goal is to allow the different Edrion runtimes to share concepts, model artifacts and AI capabilities while taking advantage of the strengths of each programming ecosystem.

---

# Current Status

Current version:

```text
v0.3.0
```

Current milestone:

```text
FEATURE-002 — Tensor Core
```

Completed milestones:

```text
v0.1.0
FEATURE-000 — NestJS Foundation ✅

v0.2.0
FEATURE-001 — Core Runtime + Backend Abstraction ✅

v0.3.0
FEATURE-002 — Tensor Core ✅
```

At this stage, Edrion TypeScript has moved beyond runtime infrastructure and now includes its first mathematical data structure: the Tensor.

---

# Current Capabilities

`v0.3.0` provides:

- Framework-independent Edrion runtime
- NestJS integration
- Runtime lifecycle
- Backend registry
- CPU backend lifecycle
- Custom backend support
- Runtime execution context
- Runtime metadata
- Runtime metrics
- Tensor Core
- Float32 tensor storage
- Tensor shapes
- Tensor strides
- Multidimensional indexing
- Scalar tensors
- Tensor reshape
- Tensor cloning
- Tensor factories
- Strong tensor validation
- Strict TypeScript support
- ESM build
- CommonJS build
- Type declarations
- Unit testing

---

# Architecture

The current architecture is divided into runtime, backend and tensor layers.

```text
┌───────────────────────────────┐
│      NestJS Application       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        EdrionModule           │
│        EdrionService          │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│         Edrion Runtime        │
│                               │
│ Runtime State                 │
│ Execution Context             │
│ Runtime Metrics               │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        Backend Registry       │
└───────────────┬───────────────┘
                │
                ▼
          ┌───────────┐
          │ CPU       │
          │ Backend   │
          └─────┬─────┘
                │
                ▼
┌───────────────────────────────┐
│          Tensor Core          │
│                               │
│ Shape                         │
│ Strides                       │
│ Storage                       │
│ DType                         │
│ Indexing                      │
│ Reshape                       │
└───────────────────────────────┘
```

---

# Design Principles

## Framework Independence

The core runtime must remain independent from application frameworks.

```text
NestJS
   │
   ▼
Edrion Integration
   │
   ▼
Edrion Runtime
   │
   ▼
Edrion Core
```

Future adapters can reuse the same runtime:

```text
                  Edrion Core
                      │
         ┌────────────┼────────────┐
         │            │            │
       NestJS       Express      Node.js
                                   │
                                GraphQL
```

---

## Backend Independence

Tensor computation is designed to evolve toward backend-independent execution.

```text
Tensor
  │
  ▼
Operations
  │
  ▼
Backend Registry
  │
  ├── CPU
  ├── WASM
  └── WebGPU
```

In `v0.3.0`, tensors already store a backend identifier, while actual mathematical kernels will be introduced later.

---

## Observability First

Observability is considered part of the Edrion architecture from the beginning.

Current runtime metrics include:

- initialization count
- shutdown count
- failure count
- initialization timestamp
- shutdown timestamp
- uptime

Future observability will expand toward:

- tensor operations
- tensor allocations
- model parameters
- layers
- embeddings
- attention heads
- token usage
- training metrics
- inference metrics
- execution duration
- memory consumption

---

# Installation

Using pnpm:

```bash
pnpm add edrion-typescript
```

Using npm:

```bash
npm install edrion-typescript
```

Using Yarn:

```bash
yarn add edrion-typescript
```

---

# Tensor Core

`v0.3.0` introduces the first Edrion Tensor implementation.

A Tensor is a multidimensional container of numerical values.

Example:

```ts
import { Tensor } from 'edrion-typescript';

const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
```

Conceptually:

```text
[
  [1, 2],
  [3, 4]
]
```

---

# Tensor Properties

```ts
tensor.shape;
```

```text
[2, 2]
```

```ts
tensor.strides;
```

```text
[2, 1]
```

```ts
tensor.rank;
```

```text
2
```

```ts
tensor.size;
```

```text
4
```

```ts
tensor.dtype;
```

```text
float32
```

```ts
tensor.backend;
```

```text
cpu
```

---

# Float32 Storage

Tensor values are stored internally using:

```ts
Float32Array;
```

instead of regular JavaScript arrays.

Example:

```ts
const tensor = new Tensor(new Float32Array([1, 2, 3]), [3]);
```

Using typed arrays creates a better foundation for future high-performance execution using:

- WebAssembly
- Worker Threads
- Web Workers
- WebGPU

---

# Shapes

Tensor shape describes the number of dimensions and their sizes.

Vector:

```text
[3]
```

Matrix:

```text
[2, 3]
```

Rank 3 Tensor:

```text
[2, 3, 4]
```

Scalar:

```text
[]
```

---

# Tensor Size

Tensor size represents the total number of elements.

For:

```text
shape = [2, 3, 4]
```

the size is:

```text
2 × 3 × 4 = 24
```

Edrion exposes:

```ts
tensor.size;
```

---

# Tensor Rank

Tensor rank represents the number of dimensions.

```text
[]         → rank 0
[3]        → rank 1
[2, 3]     → rank 2
[2, 3, 4]  → rank 3
```

Access it through:

```ts
tensor.rank;
```

---

# Strides

Edrion currently uses row-major contiguous tensor storage.

For:

```text
shape = [2, 3, 4]
```

the strides are:

```text
[12, 4, 1]
```

These strides allow multidimensional indices to be converted into linear memory positions.

For example:

```ts
tensor.get(1, 2, 3);
```

uses:

```text
1 × 12
+
2 × 4
+
3 × 1

= 23
```

so the requested value lives at linear storage offset:

```text
23
```

---

# Tensor Indexing

Tensor values can be read using `get()`.

```ts
const tensor = new Tensor([1, 2, 3, 4], [2, 2]);

tensor.get(0, 1);
```

Returns:

```text
2
```

Values can be updated using `set()`.

```ts
tensor.set(10, 0, 1);
```

Then:

```ts
tensor.get(0, 1);
```

returns:

```text
10
```

---

# Tensor Validation

Tensor construction validates that storage size matches shape size.

This is invalid:

```ts
new Tensor(new Float32Array([1, 2, 3]), [2, 2]);
```

because:

```text
data length = 3
shape size  = 4
```

Edrion throws a `TensorDataError`.

Tensor indexing also performs bounds validation.

---

# Scalar Tensors

Edrion supports rank-zero tensors.

```ts
const tensor = Tensor.scalar(7);
```

Properties:

```ts
tensor.shape;
// []

tensor.rank;
// 0

tensor.size;
// 1

tensor.isScalar;
// true

tensor.get();
// 7
```

Scalar tensors are important for future loss functions and automatic differentiation.

A future training loss may look conceptually like:

```text
Tensor
shape = []
value = 0.428
```

---

# Tensor Factories

## Zeros

```ts
const tensor = Tensor.zeros([2, 3]);
```

Produces:

```text
[
  [0, 0, 0],
  [0, 0, 0]
]
```

---

## Ones

```ts
const tensor = Tensor.ones([2, 2]);
```

Produces:

```text
[
  [1, 1],
  [1, 1]
]
```

---

## Full

```ts
const tensor = Tensor.full([2, 2], 5);
```

Produces:

```text
[
  [5, 5],
  [5, 5]
]
```

---

## Scalar

```ts
const tensor = Tensor.scalar(10);
```

Produces a rank-zero Tensor.

---

# Functional Tensor API

Tensor factories can also be imported directly.

```ts
import { full, ones, scalar, zeros } from 'edrion-typescript';
```

Example:

```ts
const a = zeros([2, 2]);

const b = ones([3]);

const c = full([2, 2], 8);

const d = scalar(0.5);
```

---

# Reshape

Tensor dimensions can be changed without changing the number of elements.

```ts
const tensor = new Tensor([1, 2, 3, 4], [2, 2]);

const reshaped = tensor.reshape([4]);
```

Original:

```text
[
  [1, 2],
  [3, 4]
]
```

Reshaped:

```text
[1, 2, 3, 4]
```

The total tensor size must remain unchanged.

This is invalid:

```ts
tensor.reshape([3, 3]);
```

because a Tensor with four elements cannot become a Tensor containing nine elements.

---

# Clone

Tensors can be copied independently.

```ts
const original = Tensor.ones([2, 2]);

const copy = original.clone();

copy.set(10, 0, 0);
```

Changing `copy` does not mutate `original`.

---

# Array Conversion

Nested representation:

```ts
tensor.toArray();
```

Example:

```ts
[
    [1, 2],
    [3, 4],
];
```

Flat representation:

```ts
tensor.toFlatArray();
```

Result:

```ts
[1, 2, 3, 4];
```

Typed representation:

```ts
tensor.toFloat32Array();
```

returns an independent `Float32Array`.

---

# DType

The initial supported Tensor data type is:

```text
float32
```

The architecture is designed to support additional types in the future, such as:

```text
float64
int32
bool
```

They are intentionally not introduced yet to keep the early tensor engine focused.

---

# Backend Metadata

Every Tensor has an associated backend identifier.

Default:

```text
cpu
```

Example:

```ts
const tensor = new Tensor([1, 2], [2], {
    backend: 'cpu',
});
```

In this release, backend information is metadata.

Future tensor operations will use the runtime backend system to execute computational kernels.

---

# NestJS Integration

Edrion can still be registered through NestJS:

```ts
import { Module } from '@nestjs/common';

import { EdrionModule } from 'edrion-typescript';

@Module({
    imports: [
        EdrionModule.forRoot({
            backend: 'cpu',

            observability: {
                enabled: true,
            },

            runtime: {
                metadata: {
                    application: 'my-ai-application',
                },
            },
        }),
    ],
})
export class AppModule {}
```

---

# Framework-Independent Runtime

NestJS is not required to use Edrion.

```ts
import { Edrion } from 'edrion-typescript';

const edrion = new Edrion({
    backend: 'cpu',
});

await edrion.initialize();

console.log(edrion.getRuntimeInfo());

await edrion.shutdown();
```

---

# Internal Imports

The source code uses aliases such as:

```ts
import { Tensor } from '@core';

import { EdrionService } from '@nest';
```

Primary aliases:

```text
@core
@core/*

@nest
@nest/*

@test/*
```

This avoids deeply nested imports and keeps the codebase maintainable as the framework expands.

---

# Project Structure

```text
src/
├── core/
│   ├── backend/
│   ├── config/
│   ├── metrics/
│   ├── runtime/
│   │
│   ├── tensor/
│   │   ├── dtype/
│   │   │   ├── dtype.ts
│   │   │   └── index.ts
│   │   │
│   │   ├── shape/
│   │   │   ├── shape.ts
│   │   │   ├── strides.ts
│   │   │   └── index.ts
│   │   │
│   │   ├── storage/
│   │   │   ├── tensor-storage.ts
│   │   │   └── index.ts
│   │   │
│   │   ├── tensor.ts
│   │   ├── tensor.factory.ts
│   │   ├── tensor.utils.ts
│   │   ├── tensor.errors.ts
│   │   └── index.ts
│   │
│   └── index.ts
│
├── nest/
│   ├── constants/
│   ├── interfaces/
│   ├── edrion.module.ts
│   ├── edrion.service.ts
│   └── index.ts
│
└── index.ts
```

Tests:

```text
test/
├── backend/
├── core/
├── nest/
│
└── tensor/
    ├── dtype.spec.ts
    ├── shape.spec.ts
    ├── strides.spec.ts
    ├── tensor.spec.ts
    ├── tensor.indexing.spec.ts
    ├── tensor.reshape.spec.ts
    └── tensor.factory.spec.ts
```

---

# Development

Install dependencies:

```bash
pnpm install
```

Type checking:

```bash
pnpm typecheck
```

Tests:

```bash
pnpm test
```

Lint:

```bash
pnpm lint
```

Formatting:

```bash
pnpm format
```

Build:

```bash
pnpm build
```

Complete validation:

```bash
pnpm check
```

---

# Quality Gate

Every feature must pass:

```text
TypeScript
    ↓
ESLint
    ↓
Vitest
    ↓
Production Build
```

before being merged or released.

---

# Roadmap

## v0.1.0 — NestJS Foundation

```text
FEATURE-000 ✅
```

Introduced the initial TypeScript project and official NestJS integration.

---

## v0.2.0 — Core Runtime + Backend Abstraction

```text
FEATURE-001 ✅
```

Introduced:

- backend abstraction
- backend registry
- CPU backend lifecycle
- runtime lifecycle
- runtime state
- execution context
- runtime metadata
- runtime metrics
- custom backend support

---

## v0.3.0 — Tensor Core

```text
FEATURE-002 ✅
```

Introduced:

- Tensor
- DType
- Float32 storage
- Shapes
- Strides
- Scalar tensors
- Indexing
- Reshape
- Clone
- Tensor factories
- Tensor validation

---

## v0.4.0 — Tensor Operations + Autograd

```text
FEATURE-003
```

Planned:

- addition
- subtraction
- multiplication
- division
- power
- reductions
- broadcasting foundation
- computational graph
- `requiresGrad`
- gradients
- `backward()`
- gradient accumulation
- `zeroGrad()`

---

## v0.5.0 — Neural Network Core

```text
FEATURE-004
```

Planned:

- Module
- Parameter
- Linear
- activations
- Embedding
- LayerNorm

---

## v0.6.0 — Losses + Optimizers

```text
FEATURE-005
```

Planned:

- Cross Entropy
- MSE
- SGD
- Adam
- optimizer state
- numerical validation

---

## v0.7.0 — Transformer Foundation

```text
FEATURE-006
```

Planned:

- Attention
- Multi-Head Attention
- causal masking
- Feed Forward Networks
- Transformer Blocks

---

## v0.8.0 — Tokenizer + Generation

```text
FEATURE-007
```

Planned:

- vocabulary
- tokenizer
- BPE
- encode
- decode
- autoregressive generation

---

## v0.9.0 — Training + Serialization

```text
FEATURE-008
```

Planned:

- Trainer
- epochs
- seeds
- checkpoints
- training metrics
- model serialization
- model loading
- artifact persistence

---

## v1.0.0 — Production Foundation

```text
FEATURE-009
```

Target architecture:

```text
Tensor
   ↓
Autograd
   ↓
Neural Networks
   ↓
Transformer
   ↓
Tokenizer
   ↓
Training
   ↓
Serialization
   ↓
Inference
   ↓
NestJS
```

---

# Future

Beyond the initial `v1.0.0` foundation, Edrion TypeScript is expected to expand toward:

- Conversation Engine
- Stability Lab
- Express Adapter
- Node.js Adapter
- GraphQL Integration
- WebAssembly backend
- WebGPU backend
- Edrion Go artifact compatibility
- Multimodal foundations
- Image models
- Audio models
- Video models

---

# Long-Term Vision

Edrion is not intended to be only a wrapper around existing AI frameworks.

The objective is to build reusable foundations for:

```text
Tensor Computation
        ↓
Automatic Differentiation
        ↓
Neural Networks
        ↓
Transformers
        ↓
Training
        ↓
Inference
        ↓
Language Models
        ↓
Multimodal Intelligence
```

For TypeScript specifically, this creates a path toward:

```text
Server AI
   +
Browser AI
   +
WebAssembly
   +
WebGPU
```

allowing Edrion components and models to eventually operate across backend and frontend environments.

---

# Contributing

Before submitting changes:

```bash
pnpm check
```

Contributions should preserve the project's architectural principles:

- framework independence
- backend independence
- modularity
- strict typing
- clear public APIs
- strong validation
- automated testing
- observability from the beginning
- minimal unnecessary coupling

---

# License

This project is licensed under the **MIT License**.

Copyright © 2026 Nelson Arevalo.

See the `LICENSE` file for the complete license text.

---

<div align="center">

# EDRION

**Building intelligence from the core.**

`Go • TypeScript • AI • Deep Learning`

</div>
