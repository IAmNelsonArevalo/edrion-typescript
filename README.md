<div align="center">

# EDRION

## TypeScript

### AI framework and NestJS integration for the Edrion ecosystem

**Building intelligence from the core.**

</div>

---

## About Edrion

**Edrion** is an artificial intelligence ecosystem focused on building the foundations required to create, train, evaluate, specialize and execute intelligent models.

The project is designed around the idea that modern artificial intelligence should not be limited to a single programming language or runtime.

`edrion-typescript` brings the Edrion ecosystem to TypeScript, providing a strongly typed foundation designed to evolve toward:

- Deep Learning
- Tensor operations
- Automatic differentiation
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

NestJS is the first officially supported framework integration.

---

## Edrion Ecosystem

Edrion is designed as an ecosystem rather than a single package.

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

The long-term goal is to allow the different Edrion runtimes to share concepts, model formats and AI capabilities while taking advantage of the strengths of each language and platform.

---

## Current Status

Current version:

```text
v0.1.0
```

Current milestone:

```text
FEATURE-000 — NestJS Foundation
```

`v0.1.0` establishes the architectural and runtime foundation of Edrion TypeScript.

It does **not yet contain the tensor engine, neural networks or training system**.

Those capabilities are introduced progressively through future features.

---

## Why TypeScript?

TypeScript gives Edrion access to an enormous ecosystem spanning:

```text
Backend
├── NestJS
├── Express
└── Node.js

Frontend
├── Browser
├── React
├── Next.js
└── Web Workers

Future AI execution
├── WebAssembly
└── WebGPU
```

This creates interesting possibilities for Edrion, including running trained models directly inside JavaScript environments and, eventually, inside web browsers.

---

## Design Principles

Edrion TypeScript follows several architectural principles.

### Framework Independence

The Edrion core must never depend directly on NestJS.

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

This allows future integrations to reuse the same runtime:

```text
                   Edrion Core
                       │
         ┌─────────────┼─────────────┐
         │             │             │
       NestJS        Express       Node.js
                                      │
                                   GraphQL
```

---

### Strong Typing

Edrion is built using strict TypeScript configuration.

The project aims to avoid unsafe runtime behavior by relying heavily on:

- explicit interfaces
- strong type contracts
- strict compiler settings
- validated configuration
- predictable public APIs

---

### Modularity

Each AI capability will live inside an isolated module.

Future core structure:

```text
src/core/
├── tensor/
├── autograd/
├── nn/
├── losses/
├── optimizers/
├── tokenizers/
├── training/
├── inference/
├── generation/
├── serialization/
└── metrics/
```

This prevents responsibilities from becoming tightly coupled as the framework grows.

---

### Observability First

Observability is considered part of the Edrion architecture from the beginning.

Future versions will expose information such as:

- tensor operations
- model parameters
- layers
- training metrics
- inference metrics
- token usage
- execution duration
- memory usage
- model dimensions
- attention heads
- embeddings
- checkpoints

Observability should not be introduced as an afterthought.

---

### Extensibility

The runtime is designed to support multiple computational backends.

The first backend identifier is:

```text
cpu
```

The architecture will eventually support:

```text
CPU
 │
 ├── Native TypeScript
 │
 ├── Worker Threads
 │
 ├── WebAssembly
 │
 └── WebGPU
```

---

## Installation

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

## NestJS Requirements

When using Edrion with NestJS, your application should have:

```bash
pnpm add \
  @nestjs/common \
  @nestjs/core \
  reflect-metadata \
  rxjs
```

These dependencies are declared as peer dependencies by Edrion.

---

# NestJS Integration

## Register Edrion

Import `EdrionModule` into your NestJS application:

```ts
import {
  Module,
} from '@nestjs/common';

import {
  EdrionModule,
} from 'edrion-typescript';

@Module({
  imports: [
    EdrionModule.forRoot({
      backend: 'cpu',

      observability: {
        enabled: true,
      },
    }),
  ],
})
export class AppModule {}
```

This creates a single Edrion runtime managed by NestJS Dependency Injection.

---

## Inject EdrionService

```ts
import {
  Injectable,
} from '@nestjs/common';

import {
  EdrionService,
} from 'edrion-typescript';

@Injectable()
export class AIService {
  constructor(
    private readonly edrion:
      EdrionService,
  ) {}

  getRuntimeInfo() {
    return this.edrion.getRuntimeInfo();
  }
}
```

Example result:

```json
{
  "name": "edrion-typescript",
  "version": "0.1.0",
  "backend": "cpu",
  "observabilityEnabled": true
}
```

---

# Async NestJS Configuration

Edrion also supports asynchronous configuration.

This is useful when the configuration depends on:

- environment variables
- NestJS ConfigModule
- secrets
- remote configuration
- application-specific providers

Example:

```ts
import {
  Module,
} from '@nestjs/common';

import {
  ConfigModule,
  ConfigService,
} from '@nestjs/config';

import {
  EdrionModule,
} from 'edrion-typescript';

@Module({
  imports: [
    ConfigModule.forRoot(),

    EdrionModule.forRootAsync({
      imports: [
        ConfigModule,
      ],

      inject: [
        ConfigService,
      ],

      useFactory: (
        configService: ConfigService,
      ) => ({
        backend: 'cpu',

        observability: {
          enabled:
            configService.get<boolean>(
              'EDRION_OBSERVABILITY',
              true,
            ),
        },
      }),
    }),
  ],
})
export class AppModule {}
```

---

# Framework-Independent Usage

NestJS is an integration layer.

The core Edrion runtime can be used directly:

```ts
import {
  Edrion,
} from 'edrion-typescript';

const edrion =
  new Edrion({
    backend: 'cpu',

    observability: {
      enabled: true,
    },
  });

console.log(
  edrion.getRuntimeInfo(),
);
```

This architecture is intentional.

Future Express and Node.js integrations will use the same `Edrion` runtime.

---

# Runtime Configuration

Current configuration:

```ts
interface EdrionOptions {
  backend?: 'cpu';

  observability?: {
    enabled?: boolean;
  };
}
```

Example:

```ts
const edrion =
  new Edrion({
    backend: 'cpu',

    observability: {
      enabled: false,
    },
  });
```

Default configuration:

```ts
{
  backend: 'cpu',

  observability: {
    enabled: true,
  },
}
```

---

# Public API

The package currently exports:

```ts
import {
  Edrion,
  EdrionModule,
  EdrionService,

  EDRION_INSTANCE,
  EDRION_OPTIONS,

  normalizeEdrionOptions,
  DEFAULT_EDRION_OPTIONS,
} from 'edrion-typescript';
```

Type definitions are exported as well:

```ts
import type {
  EdrionOptions,
  EdrionBackend,
  EdrionRuntimeInfo,
  EdrionModuleOptions,
  EdrionModuleAsyncOptions,
  EdrionOptionsFactory,
} from 'edrion-typescript';
```

Consumers should import everything through the public package entry point.

Avoid importing internal files directly.

---

# Project Structure

```text
edrion-typescript/
│
├── src/
│   ├── core/
│   │   ├── config/
│   │   │   └── edrion-options.ts
│   │   │
│   │   ├── runtime/
│   │   │   └── edrion.ts
│   │   │
│   │   └── index.ts
│   │
│   ├── nest/
│   │   ├── constants/
│   │   │   └── edrion.constants.ts
│   │   │
│   │   ├── interfaces/
│   │   │   └── edrion-module-options.interface.ts
│   │   │
│   │   ├── edrion.module.ts
│   │   ├── edrion.service.ts
│   │   └── index.ts
│   │
│   └── index.ts
│
├── test/
│   ├── core/
│   │   └── edrion.spec.ts
│   │
│   └── nest/
│       └── edrion.service.spec.ts
│
├── LICENSE
├── package.json
├── tsconfig.json
├── tsconfig.build.json
├── tsup.config.ts
├── vitest.config.ts
├── eslint.config.mjs
├── prettier.config.mjs
└── README.md
```

---

# Architecture

The initial architecture intentionally separates the AI runtime from framework-specific integrations.

```text
┌──────────────────────────────┐
│      NestJS Application      │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│        EdrionModule          │
│        EdrionService         │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│           Edrion             │
│          Runtime             │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│         Edrion Core          │
│                              │
│ Tensor                       │
│ Autograd                     │
│ Neural Networks              │
│ Optimizers                   │
│ Transformers                 │
│ Training                     │
│ Inference                    │
└──────────────────────────────┘
```

The bottom layers will be progressively implemented in future versions.

---

# Development

Clone the repository:

```bash
git clone https://github.com/IAmNelsonArevalo/edrion-typescript.git
```

Enter the project:

```bash
cd edrion-typescript
```

Install dependencies:

```bash
pnpm install
```

---

## Type Checking

```bash
pnpm typecheck
```

---

## Tests

```bash
pnpm test
```

Watch mode:

```bash
pnpm test:watch
```

---

## Lint

```bash
pnpm lint
```

Automatic fixes:

```bash
pnpm lint:fix
```

---

## Format

```bash
pnpm format
```

Validate formatting:

```bash
pnpm format:check
```

---

## Build

```bash
pnpm build
```

The package generates:

```text
dist/
├── index.js
├── index.cjs
├── index.d.ts
└── source maps
```

Edrion supports both:

```text
ES Modules
CommonJS
```

as well as generated TypeScript declaration files.

---

# Quality Gate

Before merging a feature or publishing a release:

```bash
pnpm check
```

This executes:

```text
TypeScript validation
        ↓
ESLint
        ↓
Tests
        ↓
Production build
```

A feature should not be considered complete unless the complete quality gate passes.

---

# Version Roadmap

## v0.1.0 — NestJS Foundation

```text
FEATURE-000
```

Includes:

- TypeScript project foundation
- ESM
- CommonJS
- declaration files
- Edrion runtime
- runtime configuration
- observability configuration
- CPU backend identifier
- NestJS module
- NestJS service
- synchronous configuration
- asynchronous configuration
- Dependency Injection
- tests
- linting
- formatting
- production build

---

## v0.2.0 — Core Runtime + Backend Abstraction

```text
FEATURE-001
```

Planned:

- backend interface
- backend registry
- backend lifecycle
- CPU backend foundation
- runtime state
- execution context
- runtime metrics foundation

---

## v0.3.0 — Tensor Core

```text
FEATURE-002
```

Planned:

- Tensor
- shape
- strides
- dtype
- Float32Array storage
- indexing
- reshape
- tensor creation
- tensor validation

---

## v0.4.0 — Tensor Operations + Autograd

```text
FEATURE-003
```

Planned:

- mathematical operations
- computational graph
- gradients
- backward propagation
- zeroGrad
- gradient accumulation

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
- metrics
- seeds
- checkpoints
- model serialization
- model loading
- artifact persistence

---

## v1.0.0 — Production Foundation

```text
FEATURE-009
```

Target:

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

At this point, Edrion TypeScript should provide a complete initial AI framework lifecycle.

---

# Future Integrations

After the first production foundation, Edrion TypeScript is expected to expand toward:

```text
Express Adapter

Node.js Adapter

GraphQL Integration

Conversation Engine

Stability Lab

WebAssembly Backend

WebGPU Backend

Edrion Go Artifact Compatibility

Multimodal Foundation
```

---

# Long-Term Vision

Edrion is not intended to become only another wrapper around an existing AI framework.

The long-term objective is to build its own reusable foundations for:

```text
Tensor computation
        ↓
Automatic differentiation
        ↓
Neural networks
        ↓
Transformers
        ↓
Training
        ↓
Inference
        ↓
Language models
        ↓
Multimodal intelligence
```

while allowing those capabilities to be consumed naturally from different programming ecosystems.

For TypeScript specifically, the long-term opportunity includes:

```text
Server AI
   +
Browser AI
   +
WebAssembly
   +
WebGPU
```

allowing Edrion models and components to eventually operate across backend and frontend environments.

---

# Contributing

Edrion is currently under active development.

Before submitting changes:

```bash
pnpm check
```

New capabilities should follow the existing architectural principles:

- keep the core framework-independent
- avoid unnecessary coupling
- preserve strict typing
- include tests
- expose only intentional public APIs
- maintain observability considerations
- document meaningful behavior

---

# License

This project is licensed under the **MIT License**.

Copyright © 2026 Nelson Arevalo.

See the `LICENSE` file for the complete license text.

---

<div align="center">

## EDRION

**Building intelligence from the core.**

`Go • TypeScript • AI • Deep Learning`

</div>