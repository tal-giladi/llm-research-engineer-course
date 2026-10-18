# RoleQ: role-aligned scales for post-training ternary quantization

- **Topic ID:** ternary-post-training-quantization
- **Status:** WAIT
- **Next review:** 2026-12-13
- **Course change:** none
- **Candidates:** C-20261018-02

## Summary

Post-training ternary (~1.59-bit) quantization with scales aligned to each weight's role.

## Evidence

- https://arxiv.org/abs/2610.21808 — single paper, no code found.
- Curriculum gap noted for the author: the course has no inference post-training-quantization lesson (GPTQ/AWQ/INT8/INT4).

## What would change the decision

Code plus independent results; but first the course needs a foundational PTQ lesson, which is a structural decision for the author rather than a weekly ADD.

## History

- 2026-10-18 — WAIT — single paper and the foundational PTQ material it builds on is not in the course — weekly/2026-W42.md — candidates C-20261018-02
