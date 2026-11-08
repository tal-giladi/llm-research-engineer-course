# How real and how transferable is Muon's edge over AdamW (tuning gap, fine-tuning reversal)

- **Topic ID:** muon-adamw-gap-robustness
- **Status:** WAIT
- **Next review:** 2027-01-03
- **Course change:** none
- **Candidates:** C-20261104-02

## Summary

Two new papers qualify the Muon-vs-AdamW story taught in 03.4. One finds that ~90% of the untuned gap behind Moonlight's 'reuse AdamW settings' claim disappears after joint LR x WD tuning at small scale, with a small residual advantage that grows with size; the other argues and simulates that Muon-pretrained checkpoints can fine-tune worse than AdamW ones at large data scale.

## Evidence

- https://arxiv.org/abs/2611.02529 — 'Tune the Baseline First' (single author, 1.9M-15M active-param MoE ladder, one T4 GPU); code https://github.com/ethxin0011/weight_decay_ladder.
- https://arxiv.org/abs/2611.03633 — 'Pretraining Data Scale Reverses the Muon-AdamW Fine-Tuning Gap' (Li, Gai, Risteski, Springer, Raghunathan): linear associative-memory theory plus simulation; no code.

## What would change the decision

A tuned comparison at >= 100M-1B scale by an independent group, or a second paper confirming the fine-tuning reversal empirically. Then a dated caveat paragraph in 03.4 next to the Moonlight company claim (not a new lesson). 03.4 already flags that claim and cites a budget-matched benchmark.

## History

- 2026-11-08 — WAIT — two small single papers that qualify, not overturn, what 03.4 teaches (and 03.4 already marks the claim as a company claim); worth tracking as one topic so later papers are not scattered — weekly/2026-W45.md — candidates C-20261104-02
