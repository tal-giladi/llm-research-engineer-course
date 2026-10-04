# On-policy distillation variants: co-evolving teacher (DCE/SRCL) and same-family scaling laws

- **Topic ID:** on-policy-distillation-variants
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none (the base technique is taught in lesson 17.4)
- **Candidates:** C-20260928-02, C-20261001-02

## Summary

Two specific extensions of on-policy distillation: a teacher that improves alongside the student over recursive rounds plus training on shorter verified rewrites (DCE + SRCL), and a scaling study claiming peak student quality improves with teacher scale only up to about the student's own scale.

## Evidence

- https://arxiv.org/abs/2609.30652 — DCE/SRCL, single paper; compared only against an on-policy self-distillation baseline, not GRPO/DAPO.
- https://arxiv.org/abs/2609.32722 — same-family on-policy distillation scaling properties; single paper, no code confirmed.

## What would change the decision

Independent reproduction of the teacher-scale saturation result, or adoption of a co-evolving-teacher loop in a public training stack or technical report, would make either a short extension of 17.4.

## History

- 2026-10-04 — WAIT — single-paper variants of a technique now taught in 17.4; no reproduction or adoption — weekly/2026-W40.md — candidates C-20260928-02, C-20261001-02
