# ReplaySSM: cache SSM inputs, not state, for hybrid-model decoding

- **Topic ID:** replayssm-ssm-input-caching
- **Status:** WAIT
- **Next review:** 2026-11-15
- **Course change:** none
- **Candidates:** C-20261014-01

## Summary

Represents a Mamba-2 / Gated DeltaNet decode state as a checkpoint plus a buffer of recent inputs, so most decode steps read but do not write the state, speculative rollback is a pointer move, and a high-precision checkpoint can drive the recurrence while a low-precision copy serves reads.

## Evidence

- https://arxiv.org/abs/2610.13778 — paper, v1 2026-10-09 (Princeton/NVIDIA, Tri Dao); 1.20-1.47x decode at batch 256, ~1.8x vLLM spec-decode throughput (authors' measurements).
- https://github.com/Johnny-Liou/ReplaySSM — README lists merged PRs in TensorRT-LLM, vLLM and FlashInfer (one TensorRT-LLM GDN PR by NVIDIA independently) and several still open; merge status read from the README, not checked PR by PR.
- Production claims (GLM-5.3-Flash, Kimi-K3) come from the README / NVIDIA blog (company claim); the README's own Kimi-K3 note says K3 'cites ReplaySSM as a concurrent approach', weaker than 'runs in production'.

## What would change the decision

Promote to ADD when (a) the integrations are confirmed in tagged releases of vLLM/SGLang/TensorRT-LLM and (b) the course has the serving foundations this presupposes (decode memory traffic, speculative decoding) or the lesson can teach them compactly as an extension of 12.5. Also re-check after the paper has been out for a few weeks for independent benchmarks.

## History

- 2026-10-18 — WAIT — strongest serving candidate of the week with real stack integration, but nine days old, adoption details partly self-reported and inconsistent, and only one ADD fits this week; short re-review — weekly/2026-W42.md — candidates C-20261014-01
