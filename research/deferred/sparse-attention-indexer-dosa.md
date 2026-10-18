# DoSA: certificate-based conditional Top-K scoring for sparse-attention indexers

- **Topic ID:** sparse-attention-indexer-dosa
- **Status:** WAIT
- **Next review:** 2026-12-13
- **Course change:** none
- **Candidates:** C-20261014-02

## Summary

Speeds up the Top-K indexer of DeepSeek-style sparse attention by scoring conditionally with certificates that skip blocks that cannot enter the Top-K.

## Evidence

- https://arxiv.org/abs/2610.13990 — single paper, no code, SGLang baselines (authors' measurements).

## What would change the decision

Code release plus adoption in a serving stack; the course would first need native sparse attention (see block-sparse-attention-pisa, sparse-attention-decoding-ffd).

## History

- 2026-10-18 — WAIT — single paper, no code, and its prerequisite (native sparse attention) is not taught — weekly/2026-W42.md — candidates C-20261014-02
