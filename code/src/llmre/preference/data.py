"""Building a preference dataset: ratings -> pairs, filters, agreement, bias checks.

This is Module 15, lesson 15.5. Lessons 15.1-15.3 trained on rows of the form
``{"prompt", "chosen", "rejected"}``. This module produces those rows from the
shapes preference data is actually collected in:

* **rated completions** (UltraFeedback, HelpSteer): one prompt, several
  completions, each with a score -> :func:`binarize`.
* **pairwise judgments** (HH-RLHF, Chatbot Arena, LLM-as-judge): two
  completions and a verdict -> :func:`pair_from_judgment`, made robust to
  position bias by :func:`debiased_verdict`.

Then quality controls: drop ties and small margins, measure annotator agreement
(:func:`percent_agreement`, :func:`cohens_kappa`), measure length bias
(:func:`length_bias`) and rebalance it (:func:`length_balanced_subset`).

Plain Python on the CPU: rows are ``dict``s, scores are ``float``s.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Iterable


def binarize(
    prompt: str,
    completions: list[str],
    scores: list[float],
    *,
    strategy: str = "best_vs_random",
    min_margin: float = 0.0,
    rng: random.Random | None = None,
) -> dict | None:
    """Turn one prompt with N scored completions into one preference pair.

    Strategies (both used in public datasets):

    * ``"best_vs_worst"``: chosen = highest score, rejected = lowest score.
      Largest margin, but the rejected answer is often trivially bad, so the
      model learns little from it.
    * ``"best_vs_random"``: chosen = highest score, rejected = a random other
      completion with a strictly lower score. This is how
      ``HuggingFaceH4/ultrafeedback_binarized`` was built from UltraFeedback.

    Args:
        prompt: the user prompt.
        completions: N candidate responses.
        scores: N scores, higher is better (e.g. mean of rated aspects).
        strategy: ``"best_vs_random"`` or ``"best_vs_worst"``.
        min_margin: drop the pair unless ``score(chosen) - score(rejected)``
            is strictly greater than this. ``0.0`` drops exact ties only.
        rng: random source for ``best_vs_random`` (pass a seeded one).

    Returns:
        ``{"prompt", "chosen", "rejected", "score_chosen", "score_rejected"}``
        or ``None`` when no valid pair exists (fewer than 2 completions, all
        tied, or margin too small).
    """
    if len(completions) != len(scores):
        raise ValueError("completions and scores must have the same length")
    if len(completions) < 2:
        return None
    rng = rng or random.Random(0)
    best = max(range(len(scores)), key=lambda i: scores[i])
    lower = [i for i in range(len(scores)) if scores[i] < scores[best]]
    if not lower:
        return None
    if strategy == "best_vs_worst":
        rej = min(lower, key=lambda i: scores[i])
    elif strategy == "best_vs_random":
        rej = rng.choice(lower)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    if scores[best] - scores[rej] <= min_margin:
        return None
    return {
        "prompt": prompt,
        "chosen": completions[best],
        "rejected": completions[rej],
        "score_chosen": float(scores[best]),
        "score_rejected": float(scores[rej]),
    }


def pair_from_judgment(prompt: str, a: str, b: str, verdict: str) -> dict | None:
    """Turn a pairwise verdict into a pair row.

    Args:
        prompt: the user prompt.
        a: response shown first.
        b: response shown second.
        verdict: ``"A"``, ``"B"`` or ``"tie"`` (case-insensitive).

    Returns:
        A ``{"prompt", "chosen", "rejected"}`` dict, or ``None`` for a tie.
    """
    v = verdict.strip().upper()
    if v == "A":
        return {"prompt": prompt, "chosen": a, "rejected": b}
    if v == "B":
        return {"prompt": prompt, "chosen": b, "rejected": a}
    if v == "TIE":
        return None
    raise ValueError(f"verdict must be A, B or tie, got {verdict!r}")


def debiased_verdict(judge: Callable[[str, str, str], str], prompt: str, a: str, b: str) -> str:
    """Ask a pairwise judge twice with the order swapped; keep only consistent answers.

    LLM judges prefer whichever answer is shown first (or second) more often
    than the content justifies: *position bias*. The standard fix is to ask in
    both orders. If the judge picks the same underlying response both times,
    that is the verdict; if it flips with the order, the pair is a tie and is
    dropped.

    Args:
        judge: ``judge(prompt, first, second) -> "A" | "B" | "tie"`` where
            ``"A"`` means *first*.
        prompt: the user prompt.
        a: response a.
        b: response b.

    Returns:
        ``"A"`` (a wins), ``"B"`` (b wins) or ``"tie"``.
    """
    v1 = judge(prompt, a, b).strip().upper()          # a shown first
    v2 = judge(prompt, b, a).strip().upper()          # b shown first
    v2_in_ab = {"A": "B", "B": "A"}.get(v2, "TIE")    # map back to a/b order
    if v1 == v2_in_ab and v1 in ("A", "B"):
        return v1
    return "tie"


def percent_agreement(labels_1: list, labels_2: list) -> float:
    """Fraction of items on which two annotators gave the same label."""
    if len(labels_1) != len(labels_2) or not labels_1:
        raise ValueError("need two non-empty label lists of equal length")
    return sum(x == y for x, y in zip(labels_1, labels_2)) / len(labels_1)


def cohens_kappa(labels_1: list, labels_2: list) -> float:
    """Cohen's kappa: agreement corrected for the agreement expected by chance.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_o`` is the observed agreement
    and ``p_e = sum_c p1(c) * p2(c)`` is the agreement two annotators would
    reach by labelling independently with their own label frequencies.
    ``1`` = perfect, ``0`` = chance level, negative = worse than chance.

    Returns:
        A ``float``. ``1.0`` when both annotators use a single identical label
        (``p_e = 1``, agreement is trivially perfect).
    """
    p_o = percent_agreement(labels_1, labels_2)
    n = len(labels_1)
    c1, c2 = Counter(labels_1), Counter(labels_2)
    p_e = sum((c1[c] / n) * (c2[c] / n) for c in set(c1) | set(c2))
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def length_bias(pairs: Iterable[dict], count: Callable[[str], int] = len) -> dict:
    """How often is the chosen response the longer one?

    A reward model trained on data where "chosen" is usually longer learns
    "longer is better" as a shortcut, and RL then inflates response length.
    Measure this before training.

    Args:
        pairs: rows with ``chosen`` and ``rejected``.
        count: length function (characters by default; pass a tokenizer's
            ``lambda s: len(tok.encode(s))`` for tokens).

    Returns:
        ``{"n", "chosen_longer", "rejected_longer", "equal",
        "frac_chosen_longer", "mean_len_chosen", "mean_len_rejected"}``.
    """
    n = longer = shorter = equal = 0
    sum_c = sum_r = 0
    for p in pairs:
        lc, lr = count(p["chosen"]), count(p["rejected"])
        n += 1
        sum_c += lc
        sum_r += lr
        if lc > lr:
            longer += 1
        elif lc < lr:
            shorter += 1
        else:
            equal += 1
    if n == 0:
        raise ValueError("no pairs")
    return {
        "n": n,
        "chosen_longer": longer,
        "rejected_longer": shorter,
        "equal": equal,
        "frac_chosen_longer": longer / n,
        "mean_len_chosen": sum_c / n,
        "mean_len_rejected": sum_r / n,
    }


def length_balanced_subset(
    pairs: list[dict], *, seed: int = 0, count: Callable[[str], int] = len
) -> list[dict]:
    """Subsample so "chosen is longer" and "chosen is shorter" are equally common.

    Keeps every pair from the minority side, an equal number sampled from the
    majority side, and every equal-length pair. After this, response length
    alone no longer predicts the label, so a reward model has to use content.

    Returns:
        A new shuffled ``list`` of pairs.
    """
    rng = random.Random(seed)
    longer = [p for p in pairs if count(p["chosen"]) > count(p["rejected"])]
    shorter = [p for p in pairs if count(p["chosen"]) < count(p["rejected"])]
    equal = [p for p in pairs if count(p["chosen"]) == count(p["rejected"])]
    k = min(len(longer), len(shorter))
    out = rng.sample(longer, k) + rng.sample(shorter, k) + equal
    rng.shuffle(out)
    return out


def validate_pair(row: dict) -> list[str]:
    """Problems that make a pair useless for training (empty list = keep).

    Checks the three fields 15.1's ``encode_pairs`` requires are non-empty
    strings, and that chosen and rejected actually differ (identical responses
    give a zero gradient in both Bradley-Terry and DPO, and are a common bug
    after deduplicating completions).
    """
    problems = []
    for k in ("prompt", "chosen", "rejected"):
        if not isinstance(row.get(k), str) or not row[k].strip():
            problems.append(f"{k} missing or empty")
    if not problems and row["chosen"].strip() == row["rejected"].strip():
        problems.append("chosen == rejected")
    return problems
