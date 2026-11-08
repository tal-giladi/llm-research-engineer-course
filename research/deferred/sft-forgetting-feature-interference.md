# SFT capability loss as feature interference (AEGIS loss)

- **Topic ID:** sft-forgetting-feature-interference
- **Status:** WAIT
- **Next review:** 2027-01-03
- **Course change:** none
- **Candidates:** C-20261105-04

## Summary

A geometric account of why narrow SFT degrades general capability: task data disturbs the near-orthogonal feature geometry from pretraining and pushes unrelated activations across decision boundaries. The proposed AEGIS loss bounds how far activation margins toward competing tokens can drift.

## Evidence

- https://arxiv.org/abs/2611.05003 — 3 model families x 14 SFT tasks, ~1.3x SFT compute (authors); no code, no venue.

## What would change the decision

Code and an independent reproduction, or adoption of the loss in a fine-tuning library. The mechanistic account would fit as a research connection in the SFT module (14.x) next to existing forgetting mitigations.

## History

- 2026-11-08 — WAIT — single paper, no code; interesting mechanistic claim but not yet testable by others — weekly/2026-W45.md — candidates C-20261105-04
