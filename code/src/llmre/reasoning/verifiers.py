"""Verifiers: automatic, rule-based reward functions for RLVR.

This is Module 16, lesson 16.2. **RLVR** — RL with Verifiable Rewards — replaces
the *learned* reward model of RLHF (lesson 15.1) with an **automatic checker**.
Instead of a neural network guessing how good a completion is (a proxy that the
policy can learn to *hack*), the reward comes from a deterministic rule that knows
the ground truth: exact-match on a math answer, unit tests for code, a symbolic
equality check. The reward is cheap, exact, and cannot be gamed the way a learned
proxy can — which is why verifiable domains (math, code) are where RL for
reasoning took off (DeepSeekMath, DeepSeek-R1).

A verifier here is any function ``(problem, answer) -> reward`` returning a float
in ``{0.0, 1.0}``. These two are deliberately tiny and transparent; a production
math verifier normalizes LaTeX, parses fractions, and calls a CAS, but the
*shape* of the signal is exactly this.
"""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    """Lowercase, strip surrounding whitespace, and drop a leading ``+``.

    Used so that ``" 42 "``, ``"42"`` and ``"+42"`` all compare equal.
    """
    return text.strip().lower().lstrip("+")


def _as_number(text: str):
    """Return ``float(text)`` if ``text`` parses as a number, else ``None``."""
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


def exact_match(pred: str, gold: str) -> float:
    """Reward 1.0 if ``pred`` equals ``gold``, else 0.0.

    The comparison is done two ways and either match counts:

    * **numeric** — if both parse as numbers, compare their float values (so
      ``"3"`` matches ``"3.0"`` and ``"+3"``);
    * **string** — otherwise compare the normalized strings (trimmed, lowercased).

    This is the outcome-level check behind most math RLVR rewards: extract the
    model's final answer, compare it to the gold answer, hand back 1 or 0.

    Args:
        pred: the model's proposed answer (a string).
        gold: the ground-truth answer (a string).

    Returns:
        ``1.0`` on a match, ``0.0`` otherwise. A Python float (not a tensor):
        rewards are cheap scalars assembled into a tensor by the caller.
    """
    pred = str(pred)
    gold = str(gold)
    p_num, g_num = _as_number(pred), _as_number(gold)
    if p_num is not None and g_num is not None:
        return 1.0 if p_num == g_num else 0.0
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


# Map an operator symbol to the function that computes its result.
_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "x": lambda a, b: a * b,   # allow "3 x 4" as multiplication
}

# One integer, an operator, one integer — with optional surrounding whitespace
# and an optional trailing "=". e.g. "3+4", " 12 * 11 ", "5 - 2 =".
_ARITH_RE = re.compile(r"^\s*(-?\d+)\s*([+\-*x])\s*(-?\d+)\s*=?\s*$")


def arithmetic_verifier(problem: str, answer) -> float:
    """Grade a proposed answer to a two-operand integer arithmetic problem.

    Parses ``problem`` of the form ``"a <op> b"`` where ``<op>`` is one of
    ``+ - * x`` (``x`` is multiplication), computes the true result, and compares
    it to ``answer`` via :func:`exact_match`. This is a self-contained example of
    a verifiable reward: given only the problem string and the model's answer, the
    checker recomputes the ground truth and returns a binary reward — no labels,
    no learned reward model.

    Args:
        problem: the problem statement, e.g. ``"3 + 4"`` or ``"12*11"``.
        answer: the proposed answer, an int/float or a string like ``"7"``.

    Returns:
        ``1.0`` if the answer is correct, ``0.0`` if it is wrong.

    Raises:
        ValueError: if ``problem`` is not a recognized ``a <op> b`` expression.
    """
    m = _ARITH_RE.match(str(problem))
    if m is None:
        raise ValueError(f"cannot parse arithmetic problem: {problem!r}")
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    truth = _OPS[op](a, b)
    return exact_match(str(answer), str(truth))
