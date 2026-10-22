<div align="center">

# EDRION

## TypeScript

### AI runtime and framework integrations for the Edrion ecosystem

**Building intelligence from the core.**

</div>

---

## About Edrion

**Edrion** is an artificial intelligence ecosystem focused on building the foundations required to create, train, evaluate, specialize and execute intelligent models.

The project is designed around the idea that modern artificial intelligence should not be limited to a single programming language, framework or runtime.

`edrion-typescript` brings the Edrion ecosystem to TypeScript and provides a strongly typed, framework-independent AI runtime designed to evolve toward:

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

NestJS is the first officially supported framework integration, while the core runtime remains independent from NestJS.

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

The long-term objective is to allow different Edrion runtimes to share concepts, artifacts and model capabilities while taking advantage of the strengths of each ecosystem.

---

# Current Status

Current version:

```text
v0.2.0
```

Current milestone:

```text
FEATURE-001 — Core Runtime + Backend Abstraction
```

Completed milestones:

```text
v0.1.0
FEATURE-000 — NestJS Foundation ✅

v0.2.0
FEATURE-001 — Core Runtime + Backend Abstraction ✅
```

The project now contains the runtime and backend infrastructure required before introducing Tensor computation.

---

# Current Capabilities

`v0.2.0` provides:

- Framework-independent Edrion runtime
- NestJS integration
- Synchronous NestJS configuration
- Asynchronous NestJS configuration
- Dependency Injection
- Runtime lifecycle management
- Runtime state
- Runtime identifiers
- Execution context
- Runtime metadata
- Runtime metrics
- Backend abstraction
- Backend registry
- CPU backend lifecycle
- Custom backend support
- Graceful shutdown
- Failure tracking
- TypeScript aliases
- ESM build
- CommonJS build
- Type declaration generation
- Unit testing
- Strict TypeScript validation

---

# Why TypeScript?

TypeScript provides Edrion access to an ecosystem spanning backend, frontend and browser environments.

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

Future AI Execution
├── WebAssembly
└── WebGPU
```

This creates the possibility of using Edrion not only as a server-side AI framework but eventually as an AI runtime capable of operating inside browser environments.

---

# Design Principles

## Framework Independence

The core of Edrion must remain independent from application frameworks.

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

Future integrations should consume the same runtime:

```text
                  Edrion Core
                      │
         ┌────────────┼────────────┐
         │            │            │
       NestJS       Express      Node.js
                                   │
                                GraphQL
```

NestJS is an adapter around Edrion, not the Edrion core itself.

---

## Backend Independence

Computational execution is also separated from the runtime.

```text
Edrion Runtime
      │
      ▼
Backend Registry
      │
 ┌────┼─────────┐
 │    │         │
 ▼    ▼         ▼
CPU   WASM    WebGPU
✅   Future    Future
```

The runtime does not need to know the internal implementation of each backend.

Backends implement a common contract and are resolved through the backend registry.

---

## Strong Typing

Edrion uses strict TypeScript configuration.

The project relies on:

- explicit interfaces
- strict compiler rules
- type-safe configuration
- well-defined contracts
- predictable public APIs
- type declaration generation

---

## Modularity

AI functionality will be separated into dedicated modules.

The expected core evolution is:

```text
src/core/
├── backend/
├── config/
├── metrics/
├── runtime/
├── tensor/
├── autograd/
├── nn/
├── losses/
├── optimizers/
├── tokenizers/
├── training/
├── inference/
├── generation/
└── serialization/
```

Each area should remain focused on a single responsibility.

---

## Observability First

Observability is part of the architecture from the beginning.

Current runtime metrics include:

- initialization count
- shutdown count
- failure count
- initialization timestamp
- shutdown timestamp
- runtime uptime

Future versions will expand observability to include:

- tensor operations
- parameter counts
- layers
- neurons
- dimensions
- embeddings
- attention heads
- training metrics
- inference metrics
- token usage
- execution duration
- memory usage
- checkpoints

Observability is considered a core feature rather than an optional late-stage addition.

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

# NestJS Requirements

When using Edrion with NestJS:

```bash
pnpm add \
  @nestjs/common \
  @nestjs/core \
  reflect-metadata \
  rxjs
```

These packages are peer dependencies of `edrion-typescript`.

---

# NestJS Integration

Register Edrion inside your application:

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
                    application: 'my-application',

                    environment: 'production',
                },
            },
        }),
    ],
})
export class AppModule {}
```

NestJS automatically initializes the Edrion runtime when the module starts.

---

# Injecting EdrionService

```ts
import { Injectable } from '@nestjs/common';

import { EdrionService } from 'edrion-typescript';

@Injectable()
export class AIService {
    constructor(private readonly edrion: EdrionService) {}

    getRuntimeInfo() {
        return this.edrion.getRuntimeInfo();
    }
}
```

---

# Runtime Lifecycle in NestJS

NestJS lifecycle events are connected to Edrion automatically.

```text
NestJS startup
      │
      ▼
OnModuleInit
      │
      ▼
Edrion.initialize()
      │
      ▼
Backend.initialize()
      │
      ▼
READY
```

During shutdown:

```text
NestJS shutdown
      │
      ▼
OnModuleDestroy
      │
      ▼
Edrion.shutdown()
      │
      ▼
Backend.dispose()
      │
      ▼
STOPPED
```

---

# Async NestJS Configuration

Edrion supports asynchronous configuration.

```ts
import { Module } from '@nestjs/common';

import { ConfigModule, ConfigService } from '@nestjs/config';

import { EdrionModule } from 'edrion-typescript';

@Module({
    imports: [
        ConfigModule.forRoot(),

        EdrionModule.forRootAsync({
            imports: [ConfigModule],

            inject: [ConfigService],

            useFactory: (configService: ConfigService) => ({
                backend: 'cpu',

                observability: {
                    enabled: configService.get<boolean>('EDRION_OBSERVABILITY', true),
                },

                runtime: {
                    metadata: {
                        application: 'my-application',
                    },
                },
            }),
        }),
    ],
})
export class AppModule {}
```

---

# Framework-Independent Usage

Edrion does not require NestJS.

```ts
import { Edrion } from 'edrion-typescript';

const edrion = new Edrion({
    backend: 'cpu',

    observability: {
        enabled: true,
    },

    runtime: {
        metadata: {
            application: 'standalone-example',
        },
    },
});

await edrion.initialize();

console.log(edrion.getRuntimeInfo());

await edrion.shutdown();
```

---

# Runtime Information

After initialization:

```ts
const info = edrion.getRuntimeInfo();
```

Example:

```json
{
    "name": "edrion-typescript",
    "version": "0.2.0",
    "runtimeId": "c1edb45a-...",
    "status": "ready",
    "backend": {
        "name": "cpu",
        "device": "cpu",
        "initialized": true
    },
    "observabilityEnabled": true,
    "createdAt": "2026-10-21T...",
    "metrics": {
        "initializationCount": 1,
        "shutdownCount": 0,
        "failureCount": 0,
        "lastInitializedAt": "2026-10-21T...",
        "uptimeMs": 25
    }
}
```

---

# Runtime States

The Edrion runtime currently supports:

```text
created
   │
   ▼
initializing
   │
   ▼
ready
   │
   ▼
stopping
   │
   ▼
stopped
```

If initialization or shutdown fails:

```text
failed
```

Runtime transitions are validated internally to avoid invalid lifecycle states.

---

# Backend System

Every Edrion backend implements the same contract.

```ts
interface EdrionBackend {
    readonly name: string;

    initialize(context: BackendInitializationContext): Promise<void>;

    dispose(): Promise<void>;

    isInitialized(): boolean;

    getInfo(): EdrionBackendInfo;
}
```

The default registry currently contains:

```text
cpu
```

---

# Custom Backends

Applications may provide their own backend registry.

```ts
import { BackendRegistry, Edrion, type EdrionBackend } from 'edrion-typescript';

class CustomBackend implements EdrionBackend {
    readonly name = 'custom';

    private initialized = false;

    async initialize() {
        this.initialized = true;
    }

    async dispose() {
        this.initialized = false;
    }

    isInitialized() {
        return this.initialized;
    }

    getInfo() {
        return {
            name: this.name,

            device: 'custom-device',

            initialized: this.initialized,
        };
    }
}

const registry = new BackendRegistry();

registry.register('custom', () => new CustomBackend());

const edrion = new Edrion(
    {
        backend: 'custom',
    },

    {
        backendRegistry: registry,
    },
);

await edrion.initialize();
```

This architecture allows future computational backends without coupling the runtime directly to their implementations.

---

# Runtime Metrics

Runtime metrics can be accessed with:

```ts
const metrics = edrion.getMetrics();
```

Current metrics:

```ts
{
    initializationCount: number;
    shutdownCount: number;
    failureCount: number;

    lastInitializedAt: Date | undefined;

    lastShutdownAt: Date | undefined;

    uptimeMs: number;
}
```

This system will later be expanded to track tensor, model, training and inference activity.

---

# Import Architecture

The source code uses internal aliases.

```ts
import { Edrion } from '@core';

import { BackendRegistry } from '@core/backend';

import { EdrionService } from '@nest';
```

This avoids imports such as:

```ts
../../../core/runtime/edrion
```

and keeps imports maintainable as Edrion grows.

The primary aliases are:

```text
@core
@core/*

@nest
@nest/*

@test/*
```

Relative imports are still appropriate inside a single module.

---

# Public API

Consumers should always use the public package entrypoint:

```ts
import { Edrion, BackendRegistry, CPUBackend, EdrionModule, EdrionService, EDRION_INSTANCE, EDRION_OPTIONS } from 'edrion-typescript';
```

Types are also exported:

```ts
import type {
    EdrionOptions,
    EdrionBackend,
    EdrionBackendInfo,
    EdrionRuntimeInfo,
    EdrionRuntimeStatus,
    EdrionRuntimeMetricsSnapshot,
    EdrionExecutionContext,
    EdrionModuleOptions,
    EdrionModuleAsyncOptions,
} from 'edrion-typescript';
```

Consumers should not import internal source files directly.

---

# Current Architecture

```text
┌─────────────────────────────────────┐
│         NestJS Application          │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│            EdrionModule             │
│            EdrionService            │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│            Edrion Runtime           │
│                                     │
│ Runtime State                       │
│ Execution Context                   │
│ Runtime Metrics                     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│           Backend Registry          │
└──────────────────┬──────────────────┘
                   │
                   ▼
             ┌──────────┐
             │   CPU    │
             │ Backend  │
             └────┬─────┘
                  │
                  ▼
              Tensor Core
              FEATURE-002
```

---

# Project Structure

```text
edrion-typescript/
│
├── src/
│   ├── core/
│   │   ├── backend/
│   │   │   ├── cpu/
│   │   │   ├── backend.interface.ts
│   │   │   ├── backend.registry.ts
│   │   │   ├── backend.types.ts
│   │   │   └── index.ts
│   │   │
│   │   ├── config/
│   │   │   └── edrion-options.ts
│   │   │
│   │   ├── metrics/
│   │   │   ├── runtime-metrics.ts
│   │   │   └── index.ts
│   │   │
│   │   ├── runtime/
│   │   │   ├── edrion.ts
│   │   │   ├── execution-context.ts
│   │   │   ├── runtime-state.ts
│   │   │   └── index.ts
│   │   │
│   │   └── index.ts
│   │
│   ├── nest/
│   │   ├── constants/
│   │   ├── interfaces/
│   │   ├── edrion.module.ts
│   │   ├── edrion.service.ts
│   │   └── index.ts
│   │
│   └── index.ts
│
├── test/
│   ├── backend/
│   ├── core/
│   └── nest/
│
├── LICENSE
├── README.md
├── package.json
├── tsconfig.json
├── tsconfig.build.json
├── tsup.config.ts
├── vitest.config.ts
├── eslint.config.mjs
└── prettier.config.mjs
```

---

# Development

Clone:

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

## Tests

```bash
pnpm test
```

Watch:

```bash
pnpm test:watch
```

## Lint

```bash
pnpm lint
```

Automatic fixes:

```bash
pnpm lint:fix
```

## Format

```bash
pnpm format
```

## Build

```bash
pnpm build
```

## Complete Quality Gate

```bash
pnpm check
```

This validates:

```text
TypeScript
    ↓
ESLint
    ↓
Vitest
    ↓
Production Build
```

A feature should not be merged until the complete quality gate passes.

---

# Roadmap

## v0.1.0 — NestJS Foundation

```text
FEATURE-000 ✅
```

Introduced:

- TypeScript project foundation
- ESM
- CommonJS
- Declaration files
- Edrion runtime
- Runtime configuration
- NestJS module
- NestJS service
- `forRoot`
- `forRootAsync`
- Dependency Injection
- Observability configuration
- Tests
- Build tooling

---

## v0.2.0 — Core Runtime + Backend Abstraction

```text
FEATURE-001 ✅
```

Introduced:

- Backend contract
- Backend registry
- CPU backend
- Custom backends
- Runtime state
- Runtime lifecycle
- Execution context
- Runtime ID
- Runtime metadata
- Runtime metrics
- Failure tracking
- Graceful shutdown
- NestJS lifecycle integration
- Source aliases

---

## v0.3.0 — Tensor Core

```text
FEATURE-002
```

Planned:

- Tensor
- Shape
- Strides
- DType
- Float32Array storage
- Scalar tensors
- Tensor creation
- Index calculation
- Reshape
- Clone
- Tensor validation
- Backend integration

---

## v0.4.0 — Tensor Operations + Autograd

```text
FEATURE-003
```

Planned:

- Tensor mathematical operations
- Computational graph
- Gradient tracking
- Backward propagation
- Gradient accumulation
- zeroGrad

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
- optimizer states
- numerical validation

---

## v0.7.0 — Transformer Foundation

```text
FEATURE-006
```

Planned:

- Attention
- Multi-Head Attention
- Causal Mask
- Feed Forward Networks
- Transformer Blocks

---

## v0.8.0 — Tokenizer + Generation

```text
FEATURE-007
```

Planned:

- Vocabulary
- Tokenizer
- BPE
- Encoding
- Decoding
- Autoregressive generation

---

## v0.9.0 — Training + Serialization

```text
FEATURE-008
```

Planned:

- Trainer
- Epochs
- Seeds
- Training metrics
- Model serialization
- Checkpoints
- Model loading
- Artifact persistence

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

Beyond `v1.0.0`, Edrion TypeScript is expected to expand toward:

- Conversation Engine
- Stability Lab
- Express Adapter
- Node.js Adapter
- GraphQL Integration
- WebAssembly backend
- WebGPU backend
- Edrion Go artifact compatibility
- Multimodal AI foundations
- Image models
- Audio models
- Video models

---

# Long-Term Vision

Edrion is not intended to become only a wrapper around existing AI frameworks.

The long-term objective is to build reusable foundations for:

```text
Tensor computation
        ↓
Automatic differentiation
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

The TypeScript runtime creates an additional opportunity:

```text
Server AI
    +
Browser AI
    +
WebAssembly
    +
WebGPU
```

allowing Edrion models and components to eventually execute across server and client environments.

---

# Contributing

Edrion is under active development.

Before submitting changes:

```bash
pnpm check
```

New features should follow the architectural principles of the project:

- keep the core framework-independent
- separate responsibilities
- avoid unnecessary coupling
- preserve strict typing
- include tests
- maintain intentional public APIs
- consider observability from the beginning
- document meaningful behavior

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
