# ReplaySSM: cache SSM inputs, not state, for hybrid-model decoding

- **Topic ID:** replayssm-ssm-input-caching
- **Status:** WAIT
- **Next review:** 2027-01-03
- **Course change:** none
- **Candidates:** C-20261014-01

## Summary

Represents a Mamba-2 / Gated DeltaNet decode state as a checkpoint plus a buffer of recent inputs, so most decode steps read but do not write the state, speculative rollback is a pointer move, and a high-precision checkpoint can drive the recurrence while a low-precision copy serves reads.

## Evidence

- https://arxiv.org/abs/2610.13778 — paper, v1 2026-10-09 (Princeton/NVIDIA, Tri Dao); 1.20-1.47x decode at batch 256, ~1.8x vLLM spec-decode throughput (authors' measurements).
- https://github.com/Johnny-Liou/ReplaySSM — README lists merged PRs in TensorRT-LLM, vLLM and FlashInfer (one TensorRT-LLM GDN PR by NVIDIA independently) and several still open; merge status read from the README, not checked PR by PR.
- Production claims (GLM-5.3-Flash, Kimi-K3) come from the README / NVIDIA blog (company claim); the README's own Kimi-K3 note says K3 'cites ReplaySSM as a concurrent approach', weaker than 'runs in production'.
- https://tridao.me/blog/2026/replayssm/ — author blog (search result). Search summaries (secondary, https://discuss.pytorch.kr/t/replayssm-ssm-nvidia/12174) say tagged vLLM releases (v0.27.0 notes: 'ReplaySSM caching for faster Mamba2 standard decode') cover Mamba-2 only, behind an experimental flag, while the Gated DeltaNet path and low-precision kernels live in a fork. Not verified against the release pages.

## What would change the decision

Promote to ADD when (a) the integrations are confirmed in tagged releases of vLLM/SGLang/TensorRT-LLM and (b) the course has the serving foundations this presupposes (decode memory traffic, speculative decoding) or the lesson can teach them compactly as an extension of 12.5. Also re-check after the paper has been out for a few weeks for independent benchmarks.

## History

- 2026-10-18 — WAIT — strongest serving candidate of the week with real stack integration, but nine days old, adoption details partly self-reported and inconsistent, and only one ADD fits this week; short re-review — weekly/2026-W42.md — candidates C-20261014-01
- 2026-11-15 — WAIT — 08.6 now provides the serving foundation, but tagged-release support appears to cover Mamba-2 only (experimental flag), not the Gated DeltaNet the course teaches; recheck later — weekly/2026-W46.md — candidates none new
