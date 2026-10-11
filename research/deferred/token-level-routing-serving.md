# TokenRouter: serving systems for token-level multi-model routing

- **Topic ID:** token-level-routing-serving
- **Status:** WAIT
- **Next review:** 2026-12-06
- **Course change:** none
- **Candidates:** C-20261009-04

## Summary

A serving runtime for routing individual generated tokens between several LLMs: request-centric programming over model-centric execution, one subserver per model, asynchronous dispatch and a throughput-model-driven delayed-batching scheduler.

## Evidence

- https://arxiv.org/abs/2610.12242 — NeurIPS 2026 per the abstract page; 2.01-64.15x decoding throughput vs existing systems (authors' numbers); code reported public, repo URL not confirmed.

## What would change the decision

Adoption of token-level routing in a production serving stack or model release, or a course decision to add serving-systems material (see agentic-inference-serving-dynamo).

## History

- 2026-10-11 — WAIT — peer-reviewed systems paper but token-level routing itself is not established practice and the course has no serving module to host it — weekly/2026-W41.md — candidates C-20261009-04
