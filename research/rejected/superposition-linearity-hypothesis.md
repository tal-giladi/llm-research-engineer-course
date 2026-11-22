# Superposition Linearity Hypothesis (two continuations from one forward pass)

- **Topic ID:** superposition-linearity-hypothesis
- **Status:** REJECT
- **Next review:** on new evidence
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

Decision 2026-11-22: a web search found only aggregator summaries and an unofficial, unverified repository; no independent reproduction on another model family and no official code. The course has no interpretability module (a scope question for Tal, not for this paper). Reopen on an independent reproduction plus released code, or if Tal adds an interpretability area.

## History

- 2026-09-27 — WAIT — interesting, cheaply reproducible claim but three days old, unreplicated, and no interpretability module to place it in — weekly/2026-W39.md — candidates C-20260927-01
- 2026-11-22 — REJECT — eight weeks on, no independent reproduction or official code found (only an unverified third-party repository), and the course still has no interpretability module to host it — weekly/2026-W47.md — candidates none new
