# Online test-time training / self-training of agents and its instability (ASCENT, self-generated feedback, ReSAIL)

- **Topic ID:** agent-test-time-training-stability
- **Status:** WAIT
- **Next review:** 2026-12-06
- **Course change:** none
- **Candidates:** C-20261006-01, C-20261007-02, C-20261008-03

## Summary

Agents that keep updating their weights while deployed, from their own trajectories (self-distillation, self-generated feedback). Three independent papers in one week report that such loops destabilize and propose different fixes.

## Evidence

- https://arxiv.org/abs/2610.05303 — ASCENT: online TTT via self-distillation, three benchmark suites, single group.
- https://arxiv.org/abs/2610.05076 — self-generated feedback destabilizes test-time training (general mechanism paper), C-20261007-02.
- https://arxiv.org/abs/2609.39306 — ReSAIL, a third independent line on instability of iterative agent self-training, C-20261008-03.

## What would change the decision

A shared, reproduced explanation of the instability (e.g. the feedback-loop mechanism confirmed by a second group) or adoption of agent TTT in a public agent framework or model report. Then it could become an extension of Module 19 on agents.

## History

- 2026-10-11 — WAIT — convergent signal from three groups on the failure mode, but each fix is single-paper and the mechanism is not settled — weekly/2026-W41.md — candidates C-20261006-01, C-20261007-02, C-20261008-03
