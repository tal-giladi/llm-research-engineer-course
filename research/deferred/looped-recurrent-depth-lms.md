# Looped (recurrent-depth) language models: training recipes, fixed-point efficiency, loop-native residuals

- **Topic ID:** looped-recurrent-depth-lms
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20261005-01, C-20261007-01, C-20261013-01, C-20261015-01, C-20261105-01

## Summary

Architectures that reuse a shared block of layers several times per token to buy depth with compute instead of parameters (Universal Transformer / Huginn lineage). This week: a token-efficient training and dense-to-looped conversion recipe, fixed-point-based KV sharing and cheaper RL updates, and a loop-native residual that keeps accuracy rising to 20,000+ test-time iterations on puzzles.

## Evidence

- https://arxiv.org/abs/2610.00673 — training recipe, 1.4B LoopLM vs dense (authors' numbers), no code.
- https://arxiv.org/abs/2610.06833 — 'Looped Models Done Right, Part II', 100M-1.6B, code https://github.com/ifm-ai/xllm-loop (as recorded by the daily run).
- https://arxiv.org/abs/2610.11570 — InfiLoop, 7M-parameter puzzle model, code https://github.com/pixeli99/InfiLoop.
- Same-week related C items: Recurrent Looped Transformer (arXiv 2610.07591), SanSi (arXiv 2610.07730), DLoop (daily 2026-10-09).
- https://arxiv.org/abs/2610.12582 — Elastic Latent Recurrence: removes the one-loop tax; single paper, no code.
- https://arxiv.org/abs/2610.19975 — LatentLoop: asymmetric compute across passes in looped MoE; scaling-style study, no code.
- https://arxiv.org/abs/2611.03998 and https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 — Dory1.0-4B (NVIDIA Nemotron Labs): open-weight looped model, 8 + 20 shared (1-4 loops) + 8 layers on an nGPT backbone, 6T tokens, elastic depth; benchmark superiority over Qwen3.5-9B/Gemma 4 is a company claim with no independent evaluation; weights only, no training code; ~16 community format conversions.
- Same-week C items: ARLoop (arXiv 2611.05234), LoopVQ (arXiv 2611.05304), Loop Before You Leap (arXiv 2611.06206, BabyLM 2026).

## What would change the decision

Independent reproduction of one group's claims by another; a looped model released by a major lab or supported in a widely used training/serving stack; or a consolidating result that settles which depth prior / injection / residual scheme to teach. A clean toy-scale lesson (shared block + input injection + adaptive exit) could then extend Module 12.

## History

- 2026-10-11 — WAIT — very active cluster (three independent groups, two with code) but every result is days old, sources address different sub-problems without converging on one recipe, and there is no reproduction or adoption — weekly/2026-W41.md — candidates C-20261005-01, C-20261007-01
- 2026-10-18 — WAIT — two more independent single papers on different sub-problems; still no reproduction or shared recipe; next review date unchanged — weekly/2026-W42.md — candidates C-20261013-01, C-20261015-01
- 2026-11-08 — WAIT — the 'major-lab release' condition is now met by NVIDIA's open-weight Dory1.0-4B, but its numbers are self-reported, there is no training code and no independent evaluation, and the recipe questions (injection, residual, exit) are still unsettled; re-judge at the scheduled 2026-11-29 review with any independent evaluation that appears — weekly/2026-W45.md — candidates C-20261105-01
