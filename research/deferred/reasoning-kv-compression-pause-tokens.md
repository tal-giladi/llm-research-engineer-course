# AbridgeKV: compressing completed reasoning blocks into learned pause-token KV entries

- **Topic ID:** reasoning-kv-compression-pause-tokens
- **Status:** WAIT
- **Next review:** 2026-12-13
- **Course change:** none
- **Candidates:** C-20261018-03

## Summary

Trains the model to summarize finished reasoning blocks into a few pause-token KV entries and evict the originals during long-CoT decoding.

## Evidence

- https://arxiv.org/abs/2610.20641 — single paper, no code; vLLM numbers from the authors' harness.

## What would change the decision

A second independent paper or a merged serving-stack implementation.

## History

- 2026-10-18 — WAIT — single paper, requires training, no code — weekly/2026-W42.md — candidates C-20261018-03
