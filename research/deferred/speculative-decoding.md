# Speculative decoding (cluster): length inflation under relaxed verification, target-probability reuse, target-regenerated baselines, RL-trained drafters, draft KV staleness

- **Topic ID:** speculative-decoding
- **Status:** WAIT
- **Next review:** 2026-12-20
- **Course change:** none
- **Candidates:** C-20261019-03

## Summary

Five independent October 2026 papers on speculative decoding, each a different sub-problem. The base technique (draft, then verify with rejection sampling that preserves the target distribution) is standard in vLLM/SGLang/TensorRT-LLM but is not taught anywhere in the course.

## Evidence

- https://arxiv.org/abs/2610.23818 — relaxed verification: 1.34x accepted tokens per round but 1.54x longer completions (authors).
- https://arxiv.org/abs/2610.29210, https://arxiv.org/abs/2610.29013, https://arxiv.org/abs/2610.29527, https://arxiv.org/abs/2610.31098 — four further single papers (daily 2026-10-22/23).

## What would change the decision

A course-author decision to add an inference/serving foundation (speculative decoding builds directly on 06.3 sampling and 06.3/12.2 KV cache). With that foundation, a lossless-verification lesson is mature enough to ADD and the length-inflation result becomes its evaluation caveat.

## History

- 2026-10-25 — WAIT — active multi-group research on a foundational inference technique the course does not teach; adding the foundation is a structural decision for the author (as for serving and PTQ in W41/W42) — weekly/2026-W43.md — candidates C-20261019-03
