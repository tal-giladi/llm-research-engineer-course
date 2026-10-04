# Language drift in chain-of-thought under RLVR

- **Topic ID:** rlvr-cot-language-drift
- **Status:** WAIT
- **Next review:** 2026-11-29
- **Course change:** none
- **Candidates:** C-20261002-01

## Summary

Proves that RLVR (unlike SFT) permits unbounded drift of CoT language on genuinely novel tasks, and that constraining drift necessarily costs expected reward (a performance/monitorability trade-off).

## Evidence

- https://arxiv.org/abs/2610.02015 — single theory-plus-experiment paper (Sullivan, Koller).
- The underlying phenomenon (R1-Zero language mixing) is already taught in 17.3 as a documented limitation.

## What would change the decision

Independent scrutiny of the proofs or empirical replication; then a short extension of 17.3 or F.2 on why RLVR drifts and what a language-consistency reward costs.

## History

- 2026-10-04 — WAIT — novel theory on a phenomenon the course already mentions, but unreviewed and single paper — weekly/2026-W40.md — candidates C-20261002-01
