# Keep Sampling Mask: unbiased importance ratios under top-k/top-p rollouts (DeepSeek-V3.2 practice, TRL v1.17.0 default, vLLM support)

- **Topic ID:** rl-sampling-mask-unbiased-is
- **Status:** WAIT
- **Next review:** 2026-11-15
- **Course change:** none
- **Candidates:** C-20261030-01

## Summary

When rollouts are sampled with top-k/top-p, the engine's token probabilities are normalized over the truncated candidate set while the trainer normalizes over the full vocabulary, so importance ratios are biased (TRL reports ~1e-4 instead of ~1 at top_p=0.9). Keep Sampling Mask returns the sampling mask from the engine and has the trainer renormalize over the same set. TRL v1.17.0 also makes score centering with masked IS the default.

## Evidence

- https://github.com/huggingface/trl/releases/tag/v1.17.0 — TRL v1.17.0 (29 Oct 2026): trainers normalize over the candidate set each token was sampled from (Keep Sampling Mask from DeepSeek-V3.2); requires top_k > 0 with top_p < 1 under vLLM IS correction; score centering with masked IS default (importance_sampling_min 0.5, max 5.0), 15-20% generation-throughput cost (maintainer claim).
- vLLM inference-side support (Model Runner V2) per the daily record of 2026-10-30.
- DeepSeek-V3.2 technical report practice (company claim).

## What would change the decision

Qualifies on evidence (two independent OSS stacks, a major lab's practice) and fits as an advanced-extension section of 16.4, which already teaches that importance weights cannot fix top-k/top-p truncation. Waiting only because of the weekly ADD budget (the author-approved serving foundation took this week's slot). Next week: ADD as a 16.4 extension unless a stronger candidate competes; re-check whether verl/OpenRLHF adopted it.

## History

- 2026-11-01 — WAIT — strong, adopted fix that extends 16.4 directly; one-ADD budget used by the author-approved serving foundation; next review next week — weekly/2026-W44.md — candidates C-20261030-01
- 2026-11-08 — WAIT — still qualifies; this week's one ADD went to multi-teacher on-policy distillation, which had stronger multi-lab adoption and an independent reproduction; first in line next week — weekly/2026-W45.md — candidates none new
