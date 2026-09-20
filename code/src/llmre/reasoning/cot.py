"""Chain-of-thought utilities: prompt building, answer extraction, self-consistency.

This is Module 17, lesson 17.1. Three small, dependency-free helpers that turn a
language model's *text* into an *answer* and combine several sampled answers:

* :func:`cot_prompt` builds a few-shot **chain-of-thought** prompt — a handful of
  worked exemplars (question, reasoning, answer) followed by the new question, so
  the model continues in the same "show your steps, then state the answer" style
  (Wei et al. 2022).
* :func:`extract_answer` pulls the *final* answer out of a completion. Models are
  prompted to end with a marker (GSM8K's ``#### 42`` or ``The answer is 42``); we
  parse the text after the last such marker.
* :func:`self_consistency` takes several sampled answers for one question and
  returns the **majority vote** (Wang et al. 2022): sample many reasoning paths,
  keep the answer most of them agree on.

None of these touch a GPU or a model — they operate on plain Python strings and
lists, so they are trivially unit-testable and CPU-instant.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

# Default markers a CoT completion ends with. GSM8K uses "#### <answer>"; many
# few-shot prompts use "The answer is <answer>". We match either, case-insensitively.
_HASH_RE = re.compile(r"####\s*(.+)")
_PHRASE_RE = re.compile(r"[Tt]he answer is\s*:?\s*(.+)")


def cot_prompt(
    question: str,
    exemplars: Sequence[tuple[str, str, str] | dict] | None = None,
    answer_marker: str = "####",
) -> str:
    """Build a few-shot chain-of-thought prompt string.

    Each exemplar demonstrates the *format* we want the model to imitate: a
    question, an intermediate reasoning chain, then the answer after
    ``answer_marker``. The new ``question`` is appended with an open ``A:`` so the
    model continues by producing its own reasoning and answer.

    Args:
        question: the new question to ask (plain ``str``).
        exemplars: zero or more worked examples. Each is either a
            ``(question, reasoning, answer)`` tuple or a dict with keys
            ``"question"``, ``"reasoning"``, ``"answer"``. Pass ``None`` or an
            empty list for a zero-shot CoT prompt.
        answer_marker: the token that precedes the final answer in each exemplar
            (default ``"####"``, matching :func:`extract_answer`'s default).

    Returns:
        A single ``str`` (no tensors involved): the exemplars and the new
        question, separated by blank lines, ending in ``"A:"``.
    """
    blocks: list[str] = []
    for ex in exemplars or []:
        if isinstance(ex, dict):
            q, reasoning, ans = ex["question"], ex["reasoning"], ex["answer"]
        else:
            q, reasoning, ans = ex
        blocks.append(f"Q: {q}\nA: {reasoning} {answer_marker} {ans}")
    blocks.append(f"Q: {question}\nA:")
    return "\n\n".join(blocks)


def extract_answer(text: str, pattern: str | None = None) -> str | None:
    """Pull the final answer out of a completion string.

    We take the *last* match, because a chain of thought may mention numbers along
    the way and the real answer is stated at the end.

    Args:
        text: the model's completion (plain ``str``).
        pattern: optional regex. If given, its **last** match is used; group 1 if
            the pattern has a capture group, else the whole match. If ``None``, we
            try ``"#### <answer>"`` first, then ``"The answer is <answer>"``.

    Returns:
        The extracted answer as a stripped ``str`` (trailing ``.`` removed), or
        ``None`` if nothing matched.
    """
    if pattern is not None:
        matches = list(re.finditer(pattern, text))
        if not matches:
            return None
        m = matches[-1]
        got = m.group(1) if m.groups() else m.group(0)
        return got.strip().rstrip(".")

    for regex in (_HASH_RE, _PHRASE_RE):
        matches = list(regex.finditer(text))
        if matches:
            return matches[-1].group(1).strip().rstrip(".")
    return None


def self_consistency(answers: Iterable) -> tuple[object, int]:
    """Majority-vote over several sampled answers (self-consistency).

    Sampling multiple chains of thought and taking the answer most of them reach
    is more accurate than trusting a single greedy chain, because independent
    wrong paths tend to disagree while correct paths converge (Wang et al. 2022).

    Ties are broken **deterministically**: among the answers sharing the top
    count, the one that appeared *earliest* in ``answers`` wins. This makes the
    function reproducible regardless of dict/hash ordering.

    Args:
        answers: an iterable of hashable answers (e.g. the strings returned by
            :func:`extract_answer` for each sampled path). Not tensors.

    Returns:
        ``(answer, count)`` — the winning answer and how many votes it received.
        Raises ``ValueError`` if ``answers`` is empty.
    """
    answers = list(answers)
    if not answers:
        raise ValueError("self_consistency needs at least one answer")

    counts = Counter(answers)
    top = max(counts.values())
    # Deterministic tie-break: first answer (in input order) that hits `top`.
    for a in answers:
        if counts[a] == top:
            return a, top
    raise AssertionError("unreachable")  # pragma: no cover
