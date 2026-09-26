# GVPO: Group Variance Policy Optimization

- **Topic ID:** gvpo
- **Status:** WAIT
- **Next review:** 2026-11-26
- **Course change:** taught only as directional design-space reading in lessons/frontier/update-02.md (code/src/llmre/rl/gvpo.py); not presented as standard practice
- **Candidates:** none (added before the weekly process existed)

## Summary

Replaces GRPO's importance ratio + clipping with a weight derived from the closed-form optimum of
KL-constrained reward maximization; equivalent to an MSE between group-centered implicit reward
and group-centered reward; unique optimum; extends to on-policy distillation.

## Evidence

- GVPO (NeurIPS 2025) — https://arxiv.org/abs/2504.19599
- GVPO++ extension — http://arxiv.org/abs/2609.21432v1
- No known frontier-lab or major open-source RL stack adoption as of 2026-09-26.

## What would change the decision

Independent reproduction at LLM scale, or adoption as an option in a major RL stack (verl,
OpenRLHF, TRL) or a frontier technical report → review for a full Module 16 lesson.

## History

- 2026-09-26 — WAIT — single paper line, no independent validation or adoption yet; kept as design-space reading in F.2. — n/a
