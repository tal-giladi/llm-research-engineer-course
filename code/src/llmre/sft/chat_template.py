"""Chat templates: turning role-tagged messages into one token string.

A pretrained LM (Modules 6-7) only knows how to *continue text*. To make it
"answer the user" we fine-tune it on conversations. But a conversation is a list
of role-tagged messages::

    [{"role": "system",    "content": "You are terse."},
     {"role": "user",      "content": "2+2?"},
     {"role": "assistant", "content": "4"}]

and the model only eats a flat sequence of token ids. A **chat template** is the
fixed, deterministic rule that flattens that list into one string with special
delimiter tokens marking where each turn starts and ends. Using the *same*
template at train time and at inference time is what teaches the model where the
assistant turn begins and — via the end-of-turn token — when to stop.

This module writes the mechanism from scratch (Module 14, lesson 14.1). The
delimiters here are plain text markers (``<|user|>`` etc.). A production
tokenizer registers them as *atomic* special tokens (Module 4, lesson 04.3); we
keep them as text so the from-scratch :class:`llmre.tokenizer.bpe.BPETokenizer`
can encode them without extra vocabulary surgery.

Nothing here is a tensor: inputs and outputs are Python ``str``/``list[int]``.
Ids only become a ``torch.long`` tensor later, in the training loop.
"""

from __future__ import annotations

from typing import Callable

# Special delimiter tokens. A role header opens a turn; END_TOKEN closes it.
ROLE_TOKENS: dict[str, str] = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
END_TOKEN = "<|end|>"

#: Every delimiter this template emits. A production tokenizer would add these to
#: the vocabulary as atomic, unsplittable ids.
SPECIAL_TOKENS: list[str] = [*ROLE_TOKENS.values(), END_TOKEN]


def _segments(
    messages: list[dict], add_generation_prompt: bool = False
) -> list[tuple[str, bool]]:
    """Break the rendered chat into ``(text, is_response)`` segments, in order.

    Each message becomes three segments: a role header ``<|role|>\\n``, the
    message ``content``, and the closer ``<|end|>\\n``. ``is_response`` is
    ``True`` exactly for the *assistant's* content and its closer — those are the
    tokens we will train on (lesson 14.2). The header is never a response
    (the model does not generate ``<|assistant|>`` itself; the template does),
    and the assistant's ``<|end|>\\n`` *is* a response so the model learns to
    stop.

    Args:
        messages: list of ``{"role": str, "content": str}`` dicts. ``role`` must
            be one of ``system``/``user``/``assistant``.
        add_generation_prompt: if ``True``, append a trailing ``<|assistant|>\\n``
            header with no content, i.e. "your turn" — used at inference to prompt
            the model to generate the assistant reply.

    Returns:
        A list of ``(segment_text, is_response)`` pairs whose concatenation of
        the first elements is exactly :func:`render_chat`'s output.
    """
    segs: list[tuple[str, bool]] = []
    for m in messages:
        role = m["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"unknown role {role!r}; expected one of {list(ROLE_TOKENS)}")
        is_resp = role == "assistant"
        segs.append((f"{ROLE_TOKENS[role]}\n", False))  # header: never trained on
        segs.append((m["content"], is_resp))            # content: trained iff assistant
        segs.append((f"{END_TOKEN}\n", is_resp))        # closer: trained iff assistant
    if add_generation_prompt:
        segs.append((f"{ROLE_TOKENS['assistant']}\n", False))
    return segs


def render_chat(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Render a list of role-tagged messages to one delimited string.

    Args:
        messages: list of ``{"role", "content"}`` dicts (see :func:`_segments`).
        add_generation_prompt: append a trailing ``<|assistant|>\\n`` prompt.

    Returns:
        A single ``str``. Example (newlines shown as ``\\n``)::

            <|system|>\\nYou are terse.\\n<|end|>\\n<|user|>\\n2+2?\\n<|end|>\\n
            <|assistant|>\\n4\\n<|end|>\\n
    """
    return "".join(text for text, _ in _segments(messages, add_generation_prompt))


def assistant_char_spans(
    messages: list[dict], add_generation_prompt: bool = False
) -> list[tuple[int, int]]:
    """Character spans of each assistant response inside :func:`render_chat`.

    Each span ``(start, end)`` covers one assistant turn's content **plus** its
    ``<|end|>\\n`` closer, so ``render_chat(...)[start:end]`` is exactly the text
    the model should learn to produce for that turn.

    Args:
        messages: the same list passed to :func:`render_chat`.
        add_generation_prompt: must match what you pass to :func:`render_chat`.

    Returns:
        A list of ``(start_char, end_char)`` tuples, one per assistant turn, in
        order. Half-open indices into the rendered string.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    open_start: int | None = None
    for text, is_resp in _segments(messages, add_generation_prompt):
        if is_resp and open_start is None:
            open_start = pos
        if not is_resp and open_start is not None:
            spans.append((open_start, pos))
            open_start = None
        pos += len(text)
    if open_start is not None:
        spans.append((open_start, pos))
    return spans


def render_with_response_mask(
    messages: list[dict],
    encode: Callable[[str], list[int]],
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[int]]:
    """Encode a chat and mark which token ids are assistant-response tokens.

    We encode **segment by segment** (header / content / closer) and concatenate
    the ids, tagging every id with whether its segment was a response. Encoding
    per segment — rather than encoding the whole string and re-aligning — makes
    the mask exact: there is no ambiguity about which token straddles a boundary.

    Args:
        messages: the ``{"role", "content"}`` list.
        encode: a tokenizer's ``encode(str) -> list[int]``, e.g.
            :meth:`llmre.tokenizer.bpe.BPETokenizer.encode`.
        add_generation_prompt: append the trailing ``<|assistant|>\\n`` prompt.

    Returns:
        ``(input_ids, response_mask)``: two equal-length Python ``list[int]``.
        ``response_mask[i]`` is ``1`` when ``input_ids[i]`` is an assistant
        response token (content or its ``<|end|>`` closer), else ``0``. Feed both
        to :func:`llmre.sft.masking.build_labels`.
    """
    input_ids: list[int] = []
    response_mask: list[int] = []
    for text, is_resp in _segments(messages, add_generation_prompt):
        seg_ids = encode(text)
        input_ids.extend(seg_ids)
        response_mask.extend([1 if is_resp else 0] * len(seg_ids))
    return input_ids, response_mask
