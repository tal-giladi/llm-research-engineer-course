# Encoded but not decoded: probe vs LM-head vs behavior gap (syntax)

- **Topic ID:** probe-behavior-gap-syntax
- **Status:** REJECT
- **Next review:** on new evidence
- **Course change:** none
- **Candidates:** C-20260927-02

## Summary

A three-level evaluation (probe recoverability → LM-head readout → behavior) on a trilingual
control-dependency syntax benchmark across 7 models, finding probe > LM-head > behavior with the
gap localized to specific layers by activation patching.

## Evidence

- https://arxiv.org/abs/2609.29848 — single paper (AACL-IJCNLP 2026), one narrow syntactic phenomenon; no independent reproduction.

## What would change the decision

The general lesson ("a probe shows what is linearly decodable, not what the model uses") is
already the established probing-methodology caveat, so the paper adds a narrow data point rather
than a concept. Reopen only if the three-level framework becomes a standard evaluation protocol
used across tasks, or if the course adds an interpretability module that needs a concrete worked
case study of probing pitfalls.

## History

- 2026-09-27 — REJECT — narrow single-task paper whose methodological point is not new; not worth curriculum space on its own — weekly/2026-W39.md — candidates C-20260927-02
