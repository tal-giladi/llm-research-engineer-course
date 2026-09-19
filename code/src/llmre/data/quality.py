"""Quality filtering for a pretraining corpus (Module 11, lesson 11.1).

A real web-scale pretraining pipeline does not call ``load_dataset(...)`` and
train. Between raw Common Crawl and the token stream that Module 4's
``loader.py`` packs, every document passes a chain of cheap heuristic filters
that throw away boilerplate, machine-generated junk, and text too short or too
repetitive to teach the model anything. This module implements a small,
representative subset of the Gopher / C4-style rules as a single predicate,
``quality_filter(text) -> bool``, plus the per-document metrics it is built from.

None of these operate on tensors: the input is a raw Python ``str`` (one
document) and the outputs are plain Python floats / bools on the CPU. Quality
filtering happens on CPU, before tokenization, so there is no dtype or device to
track here — that starts only once the surviving text is encoded to ids.

The thresholds below are illustrative teaching values in the spirit of the
Gopher (Rae et al., 2021) and C4 (Raffel et al., 2020) filters; a production
pipeline tunes them per corpus.
"""

from __future__ import annotations

import re

# A word is a maximal run of "word" characters (letters/digits/underscore).
# Good enough for English-ish heuristics; real pipelines use a proper tokenizer
# or unicode segmentation, but the metric shapes are identical.
_WORD_RE = re.compile(r"\w+")


def words(text: str) -> list[str]:
    """Split ``text`` into word tokens (maximal ``\\w+`` runs).

    Args:
        text: one raw document as a Python ``str``.

    Returns:
        A ``list[str]`` of word tokens, possibly empty.
    """
    return _WORD_RE.findall(text)


def mean_word_length(text: str) -> float:
    """Mean number of characters per word.

    Real prose sits around 4-6 characters per word. A value far below that is a
    symptom of tokenized/space-spammed junk; far above it, of long hash-like or
    URL-like garbage.

    Args:
        text: one raw document.

    Returns:
        Mean word length as a ``float``; ``0.0`` if there are no words.
    """
    ws = words(text)
    if not ws:
        return 0.0
    return sum(len(w) for w in ws) / len(ws)


def symbol_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are neither letters nor digits.

    Counts characters such as ``#``, ``@``, ``*``, ``<``, ``>`` against all
    non-whitespace characters. Boilerplate, ASCII art, and markup-spam push this
    high; clean prose keeps it low (punctuation alone is a few percent).

    Args:
        text: one raw document.

    Returns:
        A ``float`` in ``[0, 1]``; ``0.0`` for text with no non-whitespace
        characters.
    """
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    symbols = sum(1 for c in non_ws if not c.isalnum())
    return symbols / len(non_ws)


def repetition_ratio(text: str) -> float:
    """Fraction of word tokens that are duplicates (``1 - unique/total``).

    A document of ``n`` words containing ``u`` distinct words scores
    ``1 - u/n``. Natural text repeats function words and so is rarely zero, but
    machine-generated spam ("buy buy buy buy ...") drives this near ``1.0``.

    Args:
        text: one raw document.

    Returns:
        A ``float`` in ``[0, 1)``; ``0.0`` if there is at most one word.
    """
    ws = [w.lower() for w in words(text)]
    if len(ws) <= 1:
        return 0.0
    return 1.0 - len(set(ws)) / len(ws)


def quality_filter(
    text: str,
    *,
    min_words: int = 20,
    max_words: int = 100_000,
    min_mean_word_length: float = 3.0,
    max_mean_word_length: float = 10.0,
    max_symbol_ratio: float = 0.10,
    max_repetition_ratio: float = 0.40,
) -> bool:
    """Return ``True`` if ``text`` should be KEPT for pretraining, else ``False``.

    A single document passes only if it clears every heuristic at once. The
    rules, in order, mirror the cheap Gopher/C4 filters:

    * length: at least ``min_words`` and at most ``max_words`` word tokens
      (too short teaches nothing; absurdly long is usually a dump/log);
    * mean word length within ``[min_mean_word_length, max_mean_word_length]``
      (catches space-spam and hash-garbage);
    * symbol ratio at most ``max_symbol_ratio`` (rejects markup/ASCII-art spam);
    * repetition ratio at most ``max_repetition_ratio`` (rejects "buy buy buy"
      style machine text).

    This is deliberately a small, readable subset. Production pipelines add
    language-id confidence, stop-word presence, fraction of lines ending in
    punctuation, a toxicity classifier, and PII redaction on top.

    Args:
        text: one raw document as a Python ``str``.
        min_words: minimum word-token count to keep.
        max_words: maximum word-token count to keep.
        min_mean_word_length: lower bound on mean characters per word.
        max_mean_word_length: upper bound on mean characters per word.
        max_symbol_ratio: upper bound on :func:`symbol_ratio`.
        max_repetition_ratio: upper bound on :func:`repetition_ratio`.

    Returns:
        ``True`` to keep the document, ``False`` to drop it.
    """
    ws = words(text)
    n = len(ws)
    if n < min_words or n > max_words:
        return False
    mwl = mean_word_length(text)
    if mwl < min_mean_word_length or mwl > max_mean_word_length:
        return False
    if symbol_ratio(text) > max_symbol_ratio:
        return False
    if repetition_ratio(text) > max_repetition_ratio:
        return False
    return True
