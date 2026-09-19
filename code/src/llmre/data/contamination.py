"""Benchmark contamination detection via n-gram overlap (Module 11, lesson 11.3).

If a benchmark's test text leaks into the training corpus, evaluation scores are
inflated: the model may have memorized the answer rather than generalized. A
standard, cheap contamination check represents each document as its set of word
n-grams and measures what fraction of a benchmark item's n-grams also appear in
a training document. A high fraction flags likely leakage.

Plain Python / CPU: inputs are ``str``, output is a ``float`` in ``[0, 1]``. No
tensors.
"""

from __future__ import annotations

import re

# Tokenize on word characters so punctuation ("paris," vs "paris") and casing do
# not create spurious mismatches — this normalization is standard in the n-gram
# contamination checks used by GPT-3 and later reports.
_WORD_RE = re.compile(r"\w+")


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Set of word n-grams of ``text``.

    Text is normalized to lower-cased ``\\w+`` word tokens first, so trailing
    punctuation and capitalization do not block a match.

    Args:
        text: one document as a Python ``str``.
        n: n-gram size in words (e.g. ``8`` or ``13`` for contamination checks).

    Returns:
        A ``set`` of ``n``-tuples of lower-cased words. Empty if the text has
        fewer than ``n`` words.
    """
    toks = _WORD_RE.findall(text.lower())
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def ngram_overlap(train_text: str, test_text: str, n: int = 8) -> float:
    """Fraction of ``test_text``'s n-grams that also occur in ``train_text``.

    This is a directional (asymmetric) measure: it asks how much of the
    *benchmark* item is present in the *training* document, which is exactly the
    leakage question. ``1.0`` means every test n-gram appears in train (e.g. the
    test item is a substring of a train doc); ``~0.0`` means the two share
    almost no length-``n`` word runs.

    Args:
        train_text: a training-corpus document.
        test_text: a benchmark / test-set item.
        n: n-gram size in words.

    Returns:
        A ``float`` in ``[0, 1]``: ``|test_ngrams ∩ train_ngrams| /
        |test_ngrams|``. Returns ``0.0`` when ``test_text`` has fewer than ``n``
        words (no n-gram to match).
    """
    test_ngrams = ngrams(test_text, n)
    if not test_ngrams:
        return 0.0
    train_ngrams = ngrams(train_text, n)
    hits = len(test_ngrams & train_ngrams)
    return hits / len(test_ngrams)
