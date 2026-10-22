# Edrion TypeScript — FEATURE-002 / v0.4.0

FEATURE-002 introduces the Tensor Core for Edrion.

## Includes

- Everything from FEATURE-000 and FEATURE-001
- `float32` DType foundation
- Tensor shape validation
- Scalar shape `[]`
- Row-major strides
- `Float32Array` internal storage
- Safe multidimensional indexing
- `Tensor` metadata: shape, strides, rank, size, dtype, backend
- `reshape()` with independent storage
- `clone()`
- `toArray()`, `toFlatArray()`, `toFloat32Array()`
- Factories: `zeros`, `ones`, `full`, `scalar`
- Tensor-specific domain errors
- Tests for dtype, shape, storage, scalar tensors, factories and Tensor behavior

FEATURE-002 intentionally does **not** include mathematical operations or autograd. Those begin in FEATURE-003.

## Branch

```bash
git switch -c EDR-TS/feature/002-tensor-core
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

## Commit

```bash
git add .
git commit -m "feat(EDR-TS-002): add tensor core"
```

## Version

- `@edrion/core`: `0.3.0`
- `@edrion/nestjs`: `0.3.0`
- `edrion-typescript`: `0.3.0`

## FEATURE-003 — Tensor Operations + Autograd

See [`FEATURE-003.md`](./FEATURE-003.md).
