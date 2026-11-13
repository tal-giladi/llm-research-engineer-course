"""Building an SFT dataset: normalize, validate, filter, dedupe, decontaminate, mix.

This is Module 14, lesson 14.4. Lessons 14.1-14.3 assumed a clean list of chats
already existed. This module is the step before that: turning downloaded
datasets (each in its own format) into one validated ``messages`` list that
:func:`llmre.sft.chat_template.render_with_response_mask` can consume.

The pipeline, in order::

    raw rows (Alpaca triples, ShareGPT turns, OpenAI-style messages)
      -> to_messages          one schema: [{"role", "content"}, ...]
      -> validate_messages    structural problems (roles, empty turns, no answer)
      -> response_issues      content problems (refusal boilerplate, truncation)
      -> dedupe_by_prompt     exact duplicates of the first user turn
      -> near-dedupe          MinHash from Module 11 (llmre.data.minhash)
      -> decontaminate        n-gram overlap with benchmark items (Module 11)
      -> mix_to_budget        sample sources to a token budget

Everything is plain Python on the CPU: a row is a ``dict``, a chat is a
``list[dict[str, str]]``. No tensors until the training loop.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from typing import Callable, Iterable

from llmre.data.contamination import ngrams
from llmre.data.minhash import MinHasher, estimate_jaccard, shingles

ROLES = ("system", "user", "assistant")

# ShareGPT-style datasets use "from"/"value" with these speaker names.
_SHAREGPT_ROLES = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
}


def to_messages(row: dict) -> list[dict[str, str]]:
    """Convert one raw dataset row to the course's ``messages`` schema.

    Three input shapes are recognized, because those are the ones the public
    SFT datasets ship in:

    * **messages** (Tulu 3, SmolTalk, Nemotron, TRL "conversational"):
      ``{"messages": [{"role": ..., "content": ...}, ...]}``
    * **ShareGPT** (OpenHermes, many community sets):
      ``{"conversations": [{"from": "human", "value": ...}, ...]}``
    * **Alpaca / Dolly triple**: ``{"instruction", "input"?, "output"}``
      (Dolly calls them ``instruction``, ``context``, ``response``).

    Args:
        row: one dataset record as a Python ``dict``.

    Returns:
        A new ``list`` of ``{"role", "content"}`` dicts with stripped content.

    Raises:
        ValueError: if the row matches none of the shapes or uses an unknown
            speaker name.
    """
    if "messages" in row:
        out = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    elif "conversations" in row:
        out = []
        for turn in row["conversations"]:
            speaker = str(turn["from"]).lower()
            if speaker not in _SHAREGPT_ROLES:
                raise ValueError(f"unknown ShareGPT speaker {turn['from']!r}")
            out.append({"role": _SHAREGPT_ROLES[speaker], "content": turn["value"]})
    elif "instruction" in row:
        context = row.get("input") or row.get("context") or ""
        answer = row.get("output", row.get("response"))
        if answer is None:
            raise ValueError("instruction row has no output/response field")
        prompt = row["instruction"].strip()
        if context.strip():
            prompt = f"{prompt}\n\n{context.strip()}"
        out = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    else:
        raise ValueError(f"unrecognized row format with keys {sorted(row)}")
    for m in out:
        m["content"] = (m["content"] or "").strip()
    return out


def validate_messages(messages: list[dict]) -> list[str]:
    """Return the structural problems of one chat (empty list = valid).

    Rules: roles are known; an optional system message comes first and only
    once; after it, turns alternate user, assistant, user, ...; no turn is
    empty; the chat ends with an assistant turn (otherwise there is nothing to
    train on, because 14.2 masks every non-assistant token).

    Args:
        messages: ``list`` of ``{"role", "content"}`` dicts.

    Returns:
        A ``list[str]`` of human-readable problems, in the order found.
    """
    problems: list[str] = []
    if not messages:
        return ["empty conversation"]
    turns = messages
    if messages[0]["role"] == "system":
        turns = messages[1:]
    for i, m in enumerate(messages):
        if m["role"] not in ROLES:
            problems.append(f"turn {i}: unknown role {m['role']!r}")
        elif m["role"] == "system" and i != 0:
            problems.append(f"turn {i}: system message not first")
        if not m["content"].strip():
            problems.append(f"turn {i}: empty {m['role']} content")
    expected = "user"
    for j, m in enumerate(turns):
        if m["role"] == "system":
            continue
        if m["role"] != expected:
            problems.append(f"turn {j}: expected {expected}, got {m['role']}")
            break
        expected = "assistant" if expected == "user" else "user"
    if not turns or turns[-1]["role"] != "assistant":
        problems.append("does not end with an assistant turn")
    return problems


# Phrases that mark a response copied from a chatbot's boilerplate rather than an
# answer. Tulu/OpenHermes-style cleaning removes rows containing them. The list
# is a teaching subset; production lists are longer and language-specific.
REFUSAL_PATTERNS = [
    r"\bas an ai (language )?model\b",
    r"\bi('m| am) (just )?an ai\b",
    r"\bi cannot (fulfill|comply with) (this|that|your) request\b",
    r"\bmy knowledge cutoff\b",
    r"\bi('m| am) sorry, but i (can't|cannot)\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
_SENTENCE_END = (".", "!", "?", "`", ")", '"', "'", "*", "]", "}", "|", ":")


def response_issues(
    messages: list[dict],
    *,
    min_chars: int = 1,
    max_chars: int = 20_000,
    check_refusals: bool = True,
    check_truncation: bool = True,
) -> list[str]:
    """Content-level problems of the assistant turns (empty list = keep).

    Checks each assistant turn for: being too short or too long, chatbot
    boilerplate/refusal phrases (:data:`REFUSAL_PATTERNS`), and an ending that
    looks cut off (last character is a letter or digit with no closing
    punctuation, the classic sign of a generation that hit ``max_tokens``).

    Refusals are not always wrong; a safety set needs good refusals. This
    filter is for general-purpose sets where boilerplate refusals of harmless
    requests teach the model to over-refuse. Apply it per source, not blindly.

    Args:
        messages: a valid chat (see :func:`validate_messages`).
        min_chars: minimum characters in an assistant turn.
        max_chars: maximum characters in an assistant turn.
        check_refusals: run the boilerplate/refusal check. Turn it off for
            safety sources, whose refusals are the point of the data.
        check_truncation: run the truncation heuristic. Turn it off for
            human-written sources, where people often skip the final period
            and the heuristic's precision is low (measured in lesson 14.4).

    Returns:
        A ``list[str]`` of problems.
    """
    issues: list[str] = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        text = m["content"].strip()
        if len(text) < min_chars:
            issues.append(f"turn {i}: response too short")
            continue
        if len(text) > max_chars:
            issues.append(f"turn {i}: response too long")
        if check_refusals and _REFUSAL_RE.search(text):
            issues.append(f"turn {i}: boilerplate/refusal phrase")
        if check_truncation and len(text) > 200 and _looks_truncated(text):
            issues.append(f"turn {i}: looks truncated")
    return issues


_LIST_OR_URL = re.compile(r"^\s*([*\-+]|\d+[.)]|#+)\s|https?://|=\s*-?\d")


def _looks_truncated(text: str) -> bool:
    """Heuristic: the last line is a prose sentence that stops without punctuation.

    A bare "ends in a letter" test flags many good answers (list items, URLs,
    equations like ``3 + 3 = 6``, sign-offs). We only flag a last line that is
    plain prose of at least 8 words and has no closing punctuation. It still has
    false positives (a poem, a heading), which is why the lab prints flagged
    rows for you to read before trusting the filter.
    """
    last = text.rstrip().splitlines()[-1].strip()
    if not last or last.endswith(_SENTENCE_END) or not last[-1].isalnum():
        return False
    if _LIST_OR_URL.search(last):
        return False
    return len(last.split()) >= 8


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection.

    NFKC unicode normalization (so full-width and composed characters compare
    equal), lower-casing, and collapsing every whitespace run to one space.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(text.split())


def first_user_turn(messages: list[dict]) -> str:
    """Return the content of the first user message ("" if there is none)."""
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def prompt_key(messages: list[dict]) -> str:
    """Stable hash of the normalized first user turn (hex ``str``).

    Uses blake2b, not Python's ``hash``, which is salted per process and would
    give different keys on every run.
    """
    norm = normalize_text(first_user_turn(messages))
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def dedupe_by_prompt(chats: Iterable[list[dict]]) -> tuple[list[list[dict]], int]:
    """Keep the first chat for each distinct normalized first user turn.

    Public mixtures overlap: the same Alpaca or ShareGPT prompt appears in many
    community sets. Two chats with the same opening request are the most common
    duplicate in SFT data, so this is the first dedupe to run.

    Args:
        chats: an iterable of chats.

    Returns:
        ``(kept, n_dropped)``: the de-duplicated ``list`` (first occurrence
        wins, order preserved) and how many were removed.
    """
    seen: set[str] = set()
    kept: list[list[dict]] = []
    dropped = 0
    for chat in chats:
        key = prompt_key(chat)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(chat)
    return kept, dropped


def near_dedupe_prompts(
    chats: list[list[dict]], *, threshold: float = 0.8, k: int = 3, m: int = 128, seed: int = 0
) -> tuple[list[list[dict]], int]:
    """Drop chats whose first user turn is a near-duplicate of an earlier one.

    Reuses Module 11's MinHash. Prompts are short, so the shingle size ``k`` is
    small (3 words, not the 5-10 used for web pages). This version compares each
    new signature with every kept one, which is ``O(n^2)``: fine for the few
    thousand rows of the lab. At scale, use the LSH banding of lesson 11.2.

    Args:
        chats: chats in priority order (earlier ones win).
        threshold: estimated Jaccard at or above which two prompts are duplicates.
        k: shingle size in words.
        m: MinHash signature length.
        seed: hash seed (reproducible).

    Returns:
        ``(kept, n_dropped)``.
    """
    hasher = MinHasher(m=m, seed=seed)
    kept: list[list[dict]] = []
    sigs: list[list[int]] = []
    dropped = 0
    for chat in chats:
        sig = hasher.signature(shingles(normalize_text(first_user_turn(chat)), k=k))
        if any(estimate_jaccard(sig, s) >= threshold for s in sigs):
            dropped += 1
            continue
        sigs.append(sig)
        kept.append(chat)
    return kept, dropped


def build_benchmark_index(benchmark_texts: Iterable[str], n: int = 8) -> set[tuple[str, ...]]:
    """Union of word n-grams over all benchmark items (the "do not train" set).

    Args:
        benchmark_texts: test questions (and, ideally, their reference answers).
        n: n-gram size. Tulu 3 used 8-gram matching for its SFT decontamination.

    Returns:
        A ``set`` of n-gram tuples.
    """
    index: set[tuple[str, ...]] = set()
    for text in benchmark_texts:
        index |= ngrams(text, n)
    return index


def contamination_score(text: str, index: set[tuple[str, ...]], n: int = 8) -> float:
    """Fraction of ``text``'s n-grams that appear in the benchmark index.

    Note the direction is the reverse of :func:`llmre.data.contamination.ngram_overlap`:
    here we ask how much of the *training prompt* is benchmark text, which is
    cheap with one big index and works when the prompt quotes only part of a
    benchmark item.

    Returns:
        A ``float`` in ``[0, 1]``; ``0.0`` if ``text`` has fewer than ``n`` words.
    """
    grams = ngrams(text, n)
    if not grams:
        return 0.0
    return len(grams & index) / len(grams)


def decontaminate(
    chats: list[list[dict]], index: set[tuple[str, ...]], *, n: int = 8, threshold: float = 0.5
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Split chats into ``(clean, removed)`` by benchmark overlap of their user turns.

    A chat is removed when any user turn has ``contamination_score >= threshold``.

    Args:
        chats: candidate training chats.
        index: from :func:`build_benchmark_index` (same ``n``).
        n: n-gram size.
        threshold: overlap fraction that counts as contaminated.

    Returns:
        ``(clean, removed)``, both ``list``s of chats; keep ``removed`` for the
        data card so the decision can be audited.
    """
    clean: list[list[dict]] = []
    removed: list[list[dict]] = []
    for chat in chats:
        hit = any(
            contamination_score(m["content"], index, n) >= threshold
            for m in chat
            if m["role"] == "user"
        )
        (removed if hit else clean).append(chat)
    return clean, removed


def approx_tokens(messages: list[dict], count: Callable[[str], int] | None = None) -> int:
    """Token count of a chat's contents.

    Args:
        messages: a chat.
        count: a tokenizer length function. Default: ``len(text) // 4``, the usual
            rough rule for English with a GPT-style BPE (about 4 characters per
            token). Pass a real tokenizer for a real budget.

    Returns:
        An ``int``.
    """
    count = count or (lambda s: max(1, len(s) // 4))
    return sum(count(m["content"]) for m in messages)


def mix_to_budget(
    sources: dict[str, list[list[dict]]],
    weights: dict[str, float],
    total_tokens: int,
    *,
    seed: int = 0,
    count: Callable[[str], int] | None = None,
) -> tuple[list[list[dict]], dict[str, int]]:
    """Sample chats from several sources so each gets ``weight`` of the token budget.

    Each source receives ``total_tokens * w / sum(weights)`` tokens. Its chats
    are shuffled (seeded) and taken until the share is reached. When a source
    runs out before its share, it contributes everything it has: there is no
    silent upsampling (repeat on purpose, never by accident).

    Args:
        sources: name -> list of chats.
        weights: name -> non-negative weight (need not sum to 1).
        total_tokens: target size of the mixture in tokens.
        seed: shuffle seed.
        count: token counter for :func:`approx_tokens`.

    Returns:
        ``(mixture, tokens_per_source)``: the shuffled mixture and the actual
        tokens taken from each source.
    """
    rng = random.Random(seed)
    wsum = sum(weights.values())
    if wsum <= 0:
        raise ValueError("weights must sum to a positive number")
    mixture: list[list[dict]] = []
    taken: dict[str, int] = {}
    for name, chats in sources.items():
        share = total_tokens * weights.get(name, 0.0) / wsum
        order = list(range(len(chats)))
        rng.shuffle(order)
        used = 0
        for i in order:
            if used >= share:
                break
            mixture.append(chats[i])
            used += approx_tokens(chats[i], count)
        taken[name] = used
    rng.shuffle(mixture)
    return mixture, taken
