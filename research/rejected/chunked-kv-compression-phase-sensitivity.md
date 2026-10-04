# Periodic weak spots from chunked KV-cache compression

- **Topic ID:** chunked-kv-compression-phase-sensitivity
- **Status:** REJECT
- **Next review:** on new evidence
- **Course change:** none
- **Candidates:** C-20261001-01

## Summary

Chunked KV-cache compression creates position-periodic retrieval gaps (up to 40 points) tied to a token's phase within a compression chunk; average benchmark scores hide them.

## Evidence

- https://arxiv.org/abs/2609.36322 — single paper, authors' own pretraining experiments.

## What would change the decision

Reopen if the course adds a KV-cache compression lesson (it currently teaches GQA and hybrid linear attention, not chunked compression) or if production compression schemes are shown to have the same phase effect.

## History

- 2026-10-04 — REJECT — an evaluation caveat about a technique the course does not teach; no home — weekly/2026-W40.md — candidates C-20261001-01
