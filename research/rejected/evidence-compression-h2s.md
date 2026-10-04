# H2S: highlight-then-summarize evidence compression for long-context QA

- **Topic ID:** evidence-compression-h2s
- **Status:** REJECT
- **Next review:** on new evidence
- **Course change:** none
- **Candidates:** C-20260929-02

## Summary

Two-stage trained pipeline that extracts question-relevant evidence spans from a long document, compresses them into a question-conditioned summary, then answers; SFT plus an RL stage on a new 6.6K-example dataset.

## Evidence

- https://arxiv.org/abs/2609.31382 — single paper with code and data (https://github.com/X-Luffy/Highlight-Then-Summarize); no reproduction or adoption.

## What would change the decision

Reopen if trained evidence compression becomes a standard component of long-context systems (adoption by a major lab or framework) or the course adds a retrieval/long-context-QA lesson that needs a worked trained-compression example.

## History

- 2026-10-04 — REJECT — narrow application pipeline (extract-then-summarize-then-answer) for one task family; no underlying mechanism the course lacks — weekly/2026-W40.md — candidates C-20260929-02
