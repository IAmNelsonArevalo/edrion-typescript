# Edrion TypeScript — v1.3.0

Edrion is a TypeScript-first deep-learning framework with its own Tensor engine, Autograd, neural-network modules, Transformer stack, data/training pipeline, tokenization, serialization and numerical-safety layer.

## Current Version

`v1.3.0 — FEATURE-013: Advanced Tokenization / BPE`

## FEATURE-013 goals

FEATURE-013 introduces Edrion's first trainable subword tokenizer and connects text corpora directly to the Dataset/DataLoader pipeline.

### BPE owned by Edrion

Core now exposes:

```text
BpeTokenizer
BpeMerge
BpeTrainingStats
TokenizedTextDataset
```

Unlike the original `BasicTokenizer`, BPE learns its vocabulary from the corpus:

```text
Corpus
  ↓
Unicode alphabet
  ↓
Pair frequencies
  ↓
Repeated BPE merges
  ↓
Subword vocabulary
```

Example:

```ts
const tokenizer = new BpeTokenizer({
    targetVocabularySize: 8192,
    minimumPairFrequency: 2,
    lowercase: true,
});

tokenizer.fit(corpus);
```

### Deterministic training

The same corpus and configuration produce the same merge order and vocabulary. Pair-frequency ties use a deterministic lexical rule.

### Unicode and exact normalized decode

FEATURE-013 operates on Unicode code points and keeps whitespace and punctuation in the symbol stream. This allows encoded text to decode back to the same normalized text without the heuristic spacing rules used by `BasicTokenizer`.

### Dataset integration

Tokenizer training can consume:

```text
Dataset<string>
```

through:

```ts
tokenizer.fitDataset(dataset);
```

and text datasets can be adapted lazily:

```text
Dataset<string>
      ↓
TokenizedTextDataset
      ↓
Dataset<readonly number[]>
```

This prepares Edrion for language-model corpus preprocessing without introducing filesystem or Node dependencies into `@edrion/core`.

### Persistence

BPE state includes:

```text
configuration
vocabulary
ordered merge table
```

and can be serialized/restored for training and inference.

### Compatibility

`BasicTokenizer` remains available. `BpeTokenizer` implements the existing `Tokenizer` interface, so the generation engine can consume it directly.

## Production capabilities

v1.3.0 adds:

```text
tokenizer:bpe
tokenizer:dataset
```

while preserving all v1.2.0 data and batched-training capabilities.

## Validation

FEATURE-013 passed:

```text
Core strict TypeScript typecheck    PASS
BPE runtime training                PASS
Unicode round-trip                  PASS
Serialization round-trip            PASS
Dataset integration                 PASS
Deterministic training              PASS
```

See:

```text
FEATURE-013.md
FEATURE-013-VALIDATION.md
PR-FEATURE-013.md
```

## Packages

```text
@edrion/core        1.3.0
@edrion/nestjs      1.3.0
edrion-typescript   1.3.0
@edrion/nest-lab    1.3.0
```

## Branch

```bash
git switch -c EDR-TS/feature/013-bpe-tokenizer
```

## Commit

```bash
git add .
git commit -m "feat(EDR-TS-013): add BPE tokenizer and dataset tokenization"
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
| v1.4.0  | FEATURE-014 — Modern Transformer Architecture  | Next    |
| v1.5.0  | FEATURE-015 — Advanced Training Engine         | Planned |
| v1.6.0  | FEATURE-016 — KV Cache + Streaming Generation  | Planned |
| v1.7.0  | FEATURE-017 — WebGPU Backend                   | Planned |
| v1.8.0  | FEATURE-018 — Mixed Precision                  | Planned |
| v1.9.0  | FEATURE-019 — Large Model Infrastructure       | Planned |
| v2.0.0  | Edrion Language Model Foundation               | Planned |
