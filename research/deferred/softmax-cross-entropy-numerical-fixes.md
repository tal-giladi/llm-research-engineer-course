# A fix for softmax and cross-entropy: Softmax Collapse and Softmax Cancellation

- **Topic ID:** softmax-cross-entropy-numerical-fixes
- **Status:** WAIT
- **Next review:** 2027-01-03
- **Course change:** none
- **Candidates:** C-20261023-01, C-20261117-01

## Summary

Claims two numerical bugs in standard softmax/log-softmax/cross-entropy: rounding can zero the dominant target logit's gradient (Collapse), and distillation losses can have nonzero gradient for identical student/teacher logits (Cancellation); corrected implementations reportedly remove Slingshot instabilities and reduce distillation drift.

## Evidence

- https://arxiv.org/abs/2610.31642 — Barron, Gnanapragasam, Tschannen (Barron at Google DeepMind per the daily run); no code.

## What would change the decision

Update 2026-11-22: C-20261117-01 (https://arxiv.org/abs/2611.18511, Brandon Amos) is a second, independent line extending the analysis to forward/reverse KL, entropy, JSD and the K1/K2/K3 estimators, with JAX listings and property-based tests in the paper (no repository). Checked against the course: `llmre.reasoning.distill.forward_kl` (17.4) in fp32 with student logits = teacher + (0, 2^-25, 0) returns exactly 0.0 and a gradient of (0, 0, 5.96e-8), while fp64 gives about -6e-17 and (-6.6e-10, 5.5e-9, -4.9e-9): the failure mode applies to the course code. What is still missing is evidence that this changes training outcomes (the gradients involved are ~1e-9) and adoption in a framework (PyTorch/JAX/TRL/verl). Either of those, or released code with a measured effect on a distillation or KL-penalized RL run, would make a dated extension of 01.3/17.4 an ADD.


Code or a full-text check showing the bugs apply to the course's from-scratch implementations (01.3/01.4, 02.3, 17.4's distill.py), or independent reproduction; then a dated extension of 01.4 and 17.4. Short review because it touches foundational code.

## History

- 2026-10-25 — WAIT — credible and foundational, but a one-day-old preprint with no code; short re-review — weekly/2026-W43.md — candidates C-20261023-01
- 2026-11-22 — WAIT — a second, independent author (Amos, C-20261117-01) extends the analysis to KL/JSD/K1-K3 with stable formulas, and a check confirms the course's fp32 `forward_kl` returns 0 and a wrong gradient for a 2^-25 logit gap; still no training-outcome evidence, no released code, no framework adoption — weekly/2026-W47.md — candidates C-20261117-01
