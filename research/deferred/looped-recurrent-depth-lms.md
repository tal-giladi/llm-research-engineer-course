# Looped (recurrent-depth) language models: training recipes, fixed-point efficiency, loop-native residuals

- **Topic ID:** looped-recurrent-depth-lms
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20261005-01, C-20261007-01

## Summary

Architectures that reuse a shared block of layers several times per token to buy depth with compute instead of parameters (Universal Transformer / Huginn lineage). This week: a token-efficient training and dense-to-looped conversion recipe, fixed-point-based KV sharing and cheaper RL updates, and a loop-native residual that keeps accuracy rising to 20,000+ test-time iterations on puzzles.

## Evidence

- https://arxiv.org/abs/2610.00673 — training recipe, 1.4B LoopLM vs dense (authors' numbers), no code.
- https://arxiv.org/abs/2610.06833 — 'Looped Models Done Right, Part II', 100M-1.6B, code https://github.com/ifm-ai/xllm-loop (as recorded by the daily run).
- https://arxiv.org/abs/2610.11570 — InfiLoop, 7M-parameter puzzle model, code https://github.com/pixeli99/InfiLoop.
- Same-week related C items: Recurrent Looped Transformer (arXiv 2610.07591), SanSi (arXiv 2610.07730), DLoop (daily 2026-10-09).

## What would change the decision

Independent reproduction of one group's claims by another; a looped model released by a major lab or supported in a widely used training/serving stack; or a consolidating result that settles which depth prior / injection / residual scheme to teach. A clean toy-scale lesson (shared block + input injection + adaptive exit) could then extend Module 12.

## History

- 2026-10-11 — WAIT — very active cluster (three independent groups, two with code) but every result is days old, sources address different sub-problems without converging on one recipe, and there is no reproduction or adoption — weekly/2026-W41.md — candidates C-20261005-01, C-20261007-01
