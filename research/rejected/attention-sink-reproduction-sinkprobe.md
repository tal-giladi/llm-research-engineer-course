# SinkProbe: published attention-sink fixes fail to reproduce at small scale

- **Topic ID:** attention-sink-reproduction-sinkprobe
- **Status:** REJECT
- **Next review:** on new evidence
- **Course change:** none
- **Candidates:** C-20260928-03

## Summary

A diagnostic suite (sink mass, massive activations, positional recall) applied to four small models finds that the cited gated-attention reduction of first-token attention does not reproduce at the authors' scale, and attributes sinks to the training objective rather than the architecture.

## Evidence

- https://arxiv.org/abs/2609.08574 — single paper (v2), small scale only.
- Checked lessons/module-12/lesson-05.md: it does not state the 46.7%→4.8% gated-attention sink figure or teach attention sinks, so no caveat (FIX) is needed.

## What would change the decision

Reopen if the course adds attention-sink material (e.g. StreamingLLM-style sink tokens) or if the negative result is reproduced at larger scale and changes how gated attention is described.

## History

- 2026-10-04 — REJECT — the course makes none of the challenged claims and does not teach attention sinks; nothing to correct or extend — weekly/2026-W40.md — candidates C-20260928-03
