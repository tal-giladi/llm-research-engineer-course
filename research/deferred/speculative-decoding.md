# Speculative decoding (cluster): length inflation under relaxed verification, target-probability reuse, target-regenerated baselines, RL-trained drafters, draft KV staleness

- **Topic ID:** speculative-decoding
- **Status:** WAIT
- **Next review:** 2026-11-15
- **Course change:** none
- **Candidates:** C-20261019-03, C-20261031-03, C-20261102-01, C-20261103-01, C-20261103-02, C-20261103-03, C-20261105-02, C-20261106-02

## Summary

Five independent October 2026 papers on speculative decoding, each a different sub-problem. The base technique (draft, then verify with rejection sampling that preserves the target distribution) is standard in vLLM/SGLang/TensorRT-LLM but is not taught anywhere in the course.

## Evidence

- https://arxiv.org/abs/2610.23818 — relaxed verification: 1.34x accepted tokens per round but 1.54x longer completions (authors).
- https://arxiv.org/abs/2610.29210, https://arxiv.org/abs/2610.29013, https://arxiv.org/abs/2610.29527, https://arxiv.org/abs/2610.31098 — four further single papers (daily 2026-10-22/23).
- https://arxiv.org/abs/2610.38981 — Why speculative decoding works better than predicted on sparse MoE models (NeurIPS 2026): predictive model of MoE speedup; no third-party reproduction yet (W44).
- The base technique (lossless draft-and-verify) is now taught in lessons/module-08/lesson-06.md (08.6, added 2026-11-01).
- https://arxiv.org/abs/2511.22972 — FLy, training-free loosely speculative decoding (ICLR 2026 per arXiv comments): entropy-gated, windowed relaxed verification; shipped in vLLM v0.33.0 (2026-11-02) as opt-in `rejection_sample_method="fly"`, documented by vLLM as intentionally lossy, with vLLM's own re-measurement (min 96.3-97.9% accuracy retention, 1.04-1.94x speedups in its setups) at https://docs.vllm.ai/en/latest/features/speculative_decoding/fly/ (as recorded by the daily run 2026-11-03). No SGLang/TensorRT-LLM adoption yet.
- https://arxiv.org/abs/2608.02438 — XPress: causal Jacobi refiner for block-diffusion drafters (DFlash), merged into vLLM core by its first author (PR #54448), shipped in v0.33.0 (authors' and PR benchmarks only).
- https://arxiv.org/abs/2610.37778 — MDraft: multi-draft on-policy distillation for diffusion drafters; single paper, no code.
- https://arxiv.org/abs/2610.27815 — ProVe: proposer-verifier self-speculative block decoding; single paper, no code.
- https://arxiv.org/abs/2611.02846 — SpecOPD: branch-parallel on-policy distillation for drafters, acceptance-length objective; single paper, no code.
- https://arxiv.org/abs/2611.06425 and https://github.com/fudan-generative-vision/D3D — D3D: standalone diffusion drafter enabling pipelined drafting; code with SGLang integration, no reproduction.

## What would change the decision

A course-author decision to add an inference/serving foundation (speculative decoding builds directly on 06.3 sampling and 06.3/12.2 KV cache). With that foundation, a lossless-verification lesson is mature enough to ADD and the length-inflation result becomes its evaluation caveat.

Update 2026-11-01: the foundation was added as 08.6 (author scope decision 2026-10-28), including the lossless rule and a one-line caveat on relaxed verification. Each refinement in this cluster (relaxed verification, drafter training, MoE speedup models) is now judged as a possible extension of 08.6 on its own evidence: independent reproduction, or adoption in vLLM/SGLang/TensorRT-LLM.

Update 2026-11-08: FLy now meets the adoption half of this bar (vLLM core, documented, re-measured by vLLM). At the 2026-11-15 review, judge an 08.6 extension on lossy vs lossless verification (what is guaranteed, measured accuracy cost) against Keep Sampling Mask for that week's ADD. Diffusion drafters (DFlash, XPress, D3D, MDraft) are covered by the author's 2026-11-03 non-autoregressive decision: keep WAIT, do not ADD.

## History

- 2026-10-25 — WAIT — active multi-group research on a foundational inference technique the course does not teach; adding the foundation is a structural decision for the author (as for serving and PTQ in W41/W42) — weekly/2026-W43.md — candidates C-20261019-03
- 2026-11-01 — WAIT — the foundation now exists (08.6 teaches lossless draft-and-verify); the cluster's refinements are re-judged against it at an earlier review, as the author's 2026-10-28 scope decision asks — weekly/2026-W44.md — candidates C-20261031-03
- 2026-11-08 — WAIT — six new entries; FLy (peer-reviewed, shipped and re-measured in vLLM core) is the first refinement to meet this file's adoption bar and is judged as a possible lossy-verification extension of 08.6 at the scheduled 2026-11-15 review; the diffusion-drafter entries (XPress, MDraft, D3D) stay WAIT under the 2026-11-03 non-autoregressive scope decision — weekly/2026-W45.md — candidates C-20261102-01, C-20261103-01, C-20261103-02, C-20261103-03, C-20261105-02, C-20261106-02
