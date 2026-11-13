"""Building an RL prompt set with verifiable answers (Module 16, lesson 16.5).

RLVR (lesson 16.2) needs prompts whose answers a program can check. Public math
and code sets are large but noisy: some answers are proofs, some are several
values, some are wrong, and many problems are too easy or too hard for the
model you are training. GRPO (lesson 16.1) learns nothing from a group where
every sample gets the same reward, so the data step that matters most is
**difficulty filtering with the current policy**.

The pipeline, in order::

    raw problems
      -> extract_boxed / gold answer      one final answer string
      -> answer_is_verifiable             drop proofs, multi-part, empty
      -> dedupe + decontaminate           reuse llmre.sft.data (Module 14.4)
      -> pass_rate with k samples         run the current policy
      -> difficulty_filter                keep 0 < pass rate < 1 (a band)

Plus :func:`check_constraints`, a tiny verifier for instruction-following
prompts with checkable constraints (IFEval-style).

Plain Python on the CPU. Rewards are floats; the policy is any callable.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

from llmre.reasoning.verifiers import exact_match


def extract_boxed(text: str) -> str | None:
    """Return the contents of the last ``\\boxed{...}`` in ``text``.

    Math datasets and reasoning models put the final answer in ``\\boxed{}``.
    The contents can contain nested braces (``\\boxed{\\frac{1}{2}}``), so a
    regex is not enough; we scan and count braces.

    Args:
        text: a solution or model completion.

    Returns:
        The boxed string (stripped), or ``None`` when there is no complete
        ``\\boxed{...}``.
    """
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        i += 1
    return None


_FRAC_RE = re.compile(r"^\\[dt]?frac\{(-?\d+)\}\{(-?\d+)\}$")


def normalize_answer(ans: str) -> str:
    """Light normalization of a final math answer for string comparison.

    Removes ``$``, ``\\left``/``\\right``, ``\\!``, spaces, a trailing period,
    ``\\text{...}`` wrappers and a leading ``x=``; rewrites ``\\frac{a}{b}`` with
    integer a, b as ``a/b``; strips thousands separators from plain numbers.
    This is a teaching subset. Production code uses a symbolic checker such as
    Hugging Face's ``math-verify``, which parses LaTeX and compares values.
    """
    s = ans.strip().strip("$").strip()
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = s.replace("\\$", "").replace("\\%", "").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace(" ", "").rstrip(".")
    s = re.sub(r"^[a-zA-Z]=", "", s)
    m = _FRAC_RE.match(s)
    if m:
        s = f"{m.group(1)}/{m.group(2)}"
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def answers_match(pred: str | None, gold: str) -> float:
    """Reward 1.0 if two final answers are equal after normalization, else 0.0.

    Fractions ``a/b`` are compared by value, so ``1/2`` matches ``0.5``.
    """
    if pred is None:
        return 0.0
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == g:
        return 1.0
    pv, gv = _value(p), _value(g)
    if pv is not None and gv is not None:
        return 1.0 if abs(pv - gv) <= 1e-9 * max(1.0, abs(gv)) else 0.0
    return exact_match(p, g)


def _value(s: str) -> float | None:
    """Numeric value of ``"3"``, ``"-2.5"`` or ``"a/b"``; ``None`` otherwise."""
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


_PROOF_RE = re.compile(r"\b(prove|show that|demonstrate that|justify)\b", re.IGNORECASE)


def answer_is_verifiable(problem: str, answer: str | None, *, max_len: int = 40) -> list[str]:
    """Reasons a problem/answer pair cannot give a reliable 0/1 reward.

    Empty list = keep. Drops: no answer; proof problems (the "answer" is an
    argument, not a value); answers with several parts (``"3, 5"``, ``"a) .."``)
    because one string match cannot grade them; very long answers, which are
    usually explanations rather than values; and multiple-choice letters
    (a policy can guess them 25% of the time, which inflates reward).

    Args:
        problem: the problem statement.
        answer: the gold final answer (e.g. from :func:`extract_boxed`).
        max_len: maximum normalized answer length in characters.

    Returns:
        A ``list[str]`` of reasons.
    """
    reasons: list[str] = []
    if answer is None or not answer.strip():
        return ["no final answer"]
    if _PROOF_RE.search(problem):
        reasons.append("proof problem")
    norm = normalize_answer(answer)
    if len(norm) > max_len:
        reasons.append("answer too long")
    if _has_top_level_separator(norm):
        reasons.append("multi-part answer")
    if re.fullmatch(r"\(?[A-Ea-e]\)?", norm):
        reasons.append("multiple-choice letter")
    return reasons


def _has_top_level_separator(s: str) -> bool:
    """True if ``s`` has a ``,`` or ``;`` outside every bracket.

    ``"3,5"`` is two answers; ``"(3,5)"`` (a point) and ``"[0,1)"`` (an
    interval) are one answer, because their commas sit inside brackets.
    """
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch in ",;" and depth == 0:
            return True
    return False


def pass_rate(
    problem: str,
    gold: str,
    policy: Callable[[str], str],
    k: int = 8,
    reward: Callable[[str | None, str], float] = answers_match,
) -> float:
    """Fraction of ``k`` sampled policy completions that the verifier accepts.

    Args:
        problem: the prompt.
        gold: the gold final answer.
        policy: ``policy(problem) -> completion`` (sampling, temperature > 0).
        k: samples per problem. ``k = 8`` or ``16`` is typical; it should match
            the GRPO group size you will train with.
        reward: ``reward(pred_answer, gold) -> 0.0 | 1.0``.

    Returns:
        A ``float`` in ``[0, 1]``.
    """
    wins = 0.0
    for _ in range(k):
        wins += reward(extract_boxed(policy(problem)), gold)
    return wins / k


def difficulty_filter(
    rows: Iterable[dict], *, low: float = 0.0, high: float = 1.0, key: str = "pass_rate"
) -> tuple[list[dict], dict[str, int]]:
    """Keep rows whose pass rate is strictly between ``low`` and ``high``.

    With the defaults this drops exactly the problems where every sample was
    right (``p = 1``) or every sample was wrong (``p = 0``). For those, every
    reward in a GRPO group is identical, every advantage is zero, and the
    problem contributes no gradient. Narrower bands such as ``(0, 0.8)`` are
    also used to push toward harder problems.

    Returns:
        ``(kept, counts)`` where ``counts`` has ``too_easy``, ``too_hard``,
        ``kept``.
    """
    kept: list[dict] = []
    counts = {"too_easy": 0, "too_hard": 0, "kept": 0}
    for r in rows:
        p = r[key]
        if p >= high:
            counts["too_easy"] += 1
        elif p <= low:
            counts["too_hard"] += 1
        else:
            counts["kept"] += 1
            kept.append(r)
    return kept, counts


def group_has_signal(rewards: list[float]) -> bool:
    """True if a GRPO group's rewards are not all equal (some advantage is non-zero)."""
    return len(set(rewards)) > 1


# --------------------------------------------------------------------------- #
# Instruction following with checkable constraints (IFEval-style)
# --------------------------------------------------------------------------- #

def check_constraints(response: str, constraints: list[dict]) -> float:
    """Reward 1.0 only if ``response`` satisfies every constraint.

    Each constraint is a dict with a ``type`` and its arguments. Supported
    types (a small subset of the IFEval families):

    * ``{"type": "max_words", "n": 50}``
    * ``{"type": "min_words", "n": 100}``
    * ``{"type": "num_bullets", "n": 3}``   exactly n lines starting with ``* `` or ``- ``
    * ``{"type": "include_keyword", "word": "budget"}``
    * ``{"type": "exclude_keyword", "word": "very"}``
    * ``{"type": "all_lowercase"}``
    * ``{"type": "end_with", "phrase": "Any other questions?"}``

    The score is all-or-nothing ("strict" accuracy). A partial-credit version
    (fraction satisfied) gives denser reward but lets the policy ignore the
    hardest constraint.

    Raises:
        ValueError: on an unknown constraint type.
    """
    words = re.findall(r"\b\w+\b", response)
    lines = response.splitlines()
    for c in constraints:
        t = c["type"]
        if t == "max_words":
            ok = len(words) <= c["n"]
        elif t == "min_words":
            ok = len(words) >= c["n"]
        elif t == "num_bullets":
            ok = sum(1 for ln in lines if re.match(r"^\s*[*-]\s+", ln)) == c["n"]
        elif t == "include_keyword":
            ok = re.search(rf"\b{re.escape(c['word'])}\b", response, re.IGNORECASE) is not None
        elif t == "exclude_keyword":
            ok = re.search(rf"\b{re.escape(c['word'])}\b", response, re.IGNORECASE) is None
        elif t == "all_lowercase":
            ok = response == response.lower()
        elif t == "end_with":
            ok = response.rstrip().endswith(c["phrase"])
        else:
            raise ValueError(f"unknown constraint type {t!r}")
        if not ok:
            return 0.0
    return 1.0
