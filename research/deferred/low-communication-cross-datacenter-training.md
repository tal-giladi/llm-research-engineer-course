# Low-communication and cross-datacenter training (DiLoCo family, compute islands, butterfly mixing)

- **Topic ID:** low-communication-cross-datacenter-training
- **Status:** WAIT
- **Next review:** 2026-12-13
- **Course change:** none
- **Candidates:** C-20261108-03

## Summary

Training one model across separate clusters with infrequent synchronization: local inner-optimizer steps, an outer optimizer on the averaged parameter deltas every H steps (DiLoCo), streaming/fragment-wise sync, compressed deltas, and gossip/butterfly mixing instead of all-reduce. Module 09 currently teaches only synchronous, colocated methods (09.1-09.3).

## Evidence

- https://arxiv.org/abs/2311.08105 — DiLoCo (Google DeepMind, verified by the daily run).
- OpenDiLoCo and INTELLECT-1 (Prime Intellect, open source, multi-continent runs) — per the daily record; not re-verified this week.
- https://www.coreauto.com/hitchhikers-guide-to-training-on-compute-islands — Core Automation case study (two-site 32B MoE; company claim; host not reachable; reviewed from a copy Tal provided).

## What would change the decision

This is a new topic area for Module 09 (a possible 09.4), so the author decides whether it belongs (question in weekly/2026-W46.md). If yes, it can be an ADD built from the DiLoCo-family literature (Google DeepMind DiLoCo + Streaming DiLoCo, Prime Intellect's open reproductions), with Core Automation's numbers only as a labelled company-claim example.

## History

- 2026-11-15 — WAIT — underlying DiLoCo family is independently reproduced over three years, but it is a new area for Module 09, and the triggering post is a company blog with unreproduced numbers; scope question to the author — weekly/2026-W46.md — candidates C-20261108-03
