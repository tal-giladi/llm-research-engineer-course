# The COUNT: n-gram count embeddings stored in host memory as a pretraining addition

- **Topic ID:** ngram-count-embeddings-host-memory
- **Status:** WAIT
- **Next review:** 2026-12-13
- **Course change:** none
- **Candidates:** C-20261015-02

## Summary

Adds n-gram count embeddings kept in CPU memory to pretraining, claimed to be compute-cheap and adaptable without training.

## Evidence

- https://arxiv.org/abs/2610.20456 — single paper, no code.
- Related signal: the Qwen4 preview recorded in W39 also used an offloadable N-gram lookup embedding table.

## What would change the decision

Independent reproduction or a second major model using host-memory n-gram tables with a published ablation; then a Module 12 extension.

## History

- 2026-10-18 — WAIT — interesting systems trade-off, single paper — weekly/2026-W42.md — candidates C-20261015-02
