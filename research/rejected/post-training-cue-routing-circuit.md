# Many Biases, One Circuit: post-training amplifies a pretrained cue-following attention circuit

- **Topic ID:** post-training-cue-routing-circuit
- **Status:** REJECT
- **Next review:** on new evidence
- **Course change:** none
- **Candidates:** C-20261020-01

## Summary

Four multiple-choice cue-following biases route through the same few attention heads, latent in base models, activated by ~200 generic SFT examples and strengthened by RL; a six-layer attention-pattern swap between RL and pre-RL checkpoints moves cue-following most of the way.

## Evidence

- https://arxiv.org/abs/2610.22937 — single paper, no code, no venue.

## What would change the decision

An interpretability-module decision by the course author, plus an independent replication of the head-swap result or a code release covering more model families.

## History

- 2026-10-25 — REJECT — single paper on circuit-level interpretability, which the course has no module for; same reasoning as circuit-discovery-objective-gap (W40) — weekly/2026-W43.md — candidates C-20261020-01
