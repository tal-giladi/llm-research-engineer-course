# Experiment report — <short title>

> Copy this file to `reports/YYYY-MM-DD-<slug>.md`, fill every section, and commit it
> **with** the config and the exact commit hash of the code that produced the numbers.
> One report = one question. If you are answering two questions, write two reports.

**Date:** YYYY-MM-DD · **Author:** <name> · **Code commit:** `<git sha>` · **Config:** `configs/<file>.yaml`

## QUESTION
The single, specific, falsifiable question this experiment answers (e.g. "Does RMSNorm lower final validation loss vs LayerNorm at 124M params on 1B tokens?"). One sentence.

## HYPOTHESIS
What you expect to happen and *why*, stated before you look at the results — the mechanism, not just the direction.

## METHOD
Exactly what you did: model size, data, tokens, steps, hardware, and anything a reader would need to reproduce the run bit-for-bit. Link the config.

## BASELINE
The reference point you are comparing against (the unmodified system or the published number), run under identical conditions so the comparison is fair.

## VARIABLES
The **one** thing you changed (independent variable) and everything you deliberately held fixed (controls). If more than one thing changed, you cannot attribute the effect.

## RESULTS
The raw numbers, reported as **mean ± std across N seeds** (never a single seed) with a confidence interval; include the metric, N, and a table or plot.

## ANALYSIS
What the numbers mean: is the difference larger than seed noise (do the confidence intervals overlap)? Is the effect the size the hypothesis predicted?

## LIMITATIONS
What this experiment does *not* show: confounds, scale it was not tested at, metrics not measured, and why the conclusion might not generalize.

## CONCLUSION
The one-line answer to the QUESTION, qualified by the ANALYSIS and LIMITATIONS. State it even when the result is negative — a clean negative result is a result.

## NEXT EXPERIMENT
The single most informative follow-up this result motivates (a new QUESTION for the next report).
