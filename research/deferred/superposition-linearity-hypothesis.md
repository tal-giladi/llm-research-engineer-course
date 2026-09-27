# Superposition Linearity Hypothesis (two continuations from one forward pass)

- **Topic ID:** superposition-linearity-hypothesis
- **Status:** WAIT
- **Next review:** 2026-11-22
- **Course change:** none
- **Candidates:** C-20260927-01

## Summary

Claims that averaging the inputs of two text streams makes a transformer's output distribution
approximately a linear combination of the two streams' next-token distributions, that this
property weakens during pretraining and can be restored by fine-tuning, and uses it to decode two
continuations at once.

## Evidence

- https://arxiv.org/abs/2609.29845 — single academic paper, published 2026-09-24; no code confirmed, no independent reproduction yet.

## What would change the decision

Independent reproduction on another model family, released code, and citation uptake. Separately,
the course has no interpretability module; a decision to add one (a curriculum-structure question,
not this paper's) would be the natural home for a hands-on linearity demo.

## History

- 2026-09-27 — WAIT — interesting, cheaply reproducible claim but three days old, unreplicated, and no interpretability module to place it in — weekly/2026-W39.md — candidates C-20260927-01
