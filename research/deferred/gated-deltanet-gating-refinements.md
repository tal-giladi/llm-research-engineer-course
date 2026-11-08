# Gating and readout refinements for Gated DeltaNet-style linear attention (Muir)

- **Topic ID:** gated-deltanet-gating-refinements
- **Status:** WAIT
- **Next review:** 2027-01-03
- **Course change:** none
- **Candidates:** C-20261107-02

## Summary

Refinements of the GDN family taught in 12.5: a low-rank erase gate shared with other gating (state updates are less sensitive to erasing than writing), a matched-filter readout gate that reweights slow-decaying, large-norm state rows, and a parallel update for large chunks.

## Evidence

- https://arxiv.org/abs/2611.08376 — 166M-1.34B models, perplexity 1.8-2.0% below GDN-2 (authors); no code.

## What would change the decision

Code plus independent reproduction, or adoption in a released hybrid model or in flash-linear-attention. Then a further-reading pointer or a short extension of 12.5.

## History

- 2026-11-08 — WAIT — single paper without code refining an architecture the course already teaches — weekly/2026-W45.md — candidates C-20261107-02
