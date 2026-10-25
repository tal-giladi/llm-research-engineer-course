# A fix for softmax and cross-entropy: Softmax Collapse and Softmax Cancellation

- **Topic ID:** softmax-cross-entropy-numerical-fixes
- **Status:** WAIT
- **Next review:** 2026-11-22
- **Course change:** none
- **Candidates:** C-20261023-01

## Summary

Claims two numerical bugs in standard softmax/log-softmax/cross-entropy: rounding can zero the dominant target logit's gradient (Collapse), and distillation losses can have nonzero gradient for identical student/teacher logits (Cancellation); corrected implementations reportedly remove Slingshot instabilities and reduce distillation drift.

## Evidence

- https://arxiv.org/abs/2610.31642 — Barron, Gnanapragasam, Tschannen (Barron at Google DeepMind per the daily run); no code.

## What would change the decision

Code or a full-text check showing the bugs apply to the course's from-scratch implementations (01.3/01.4, 02.3, 17.4's distill.py), or independent reproduction; then a dated extension of 01.4 and 17.4. Short review because it touches foundational code.

## History

- 2026-10-25 — WAIT — credible and foundational, but a one-day-old preprint with no code; short re-review — weekly/2026-W43.md — candidates C-20261023-01
