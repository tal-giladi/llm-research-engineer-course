"""Sequence packing for SFT: fit several short examples into one fixed-length row.

SFT examples are short and uneven. Padding each one to the block size ``T``
wastes most of the compute on pad tokens. Packing puts several examples
end-to-end in one row of length ``T`` and keeps three things correct per example:

1. **Loss mask** - each example keeps its own prompt/response split, and no
   position is ever trained to predict a token from a *different* example.
2. **Attention** - a token may only attend to earlier tokens of its *own*
   example (block-diagonal causal mask built from ``doc_ids``).
3. **Positions** - position ids restart at 0 for every example, so the second
   example in a row looks exactly like it would on its own.

This module (Module 14, lesson 14.2) implements packing from scratch:

* :func:`first_fit_decreasing` - the bin-packing step (which examples share a row).
* :func:`pack_examples` - build the packed tensors for those rows.
* :func:`packed_attention_mask` - the additive block-diagonal causal mask.
* :func:`packed_labels` - shifted next-token labels that respect boundaries.
* :func:`packing_efficiency` - fraction of row positions that hold real tokens.
"""

from __future__ import annotations

import torch

from llmre.sft.masking import IGNORE_INDEX

PAD_DOC = -1  # doc id written into padding positions


def first_fit_decreasing(lengths: list[int], block_size: int) -> list[list[int]]:
    """Group example indices into rows of total length ``<= block_size``.

    Sort examples longest first, then put each one into the first row that still
    has room, opening a new row when none does. Examples longer than
    ``block_size`` raise: truncate or drop them before packing.

    Args:
        lengths: token count of each example.
        block_size: row length ``T``.

    Returns:
        A list of rows; each row is the list of example indices it holds, in the
        order they were placed.
    """
    for i, n in enumerate(lengths):
        if n > block_size:
            raise ValueError(f"example {i} has {n} tokens > block_size {block_size}")
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    rows: list[list[int]] = []
    free: list[int] = []  # free[r] = tokens still available in row r
    for i in order:
        for r, room in enumerate(free):
            if lengths[i] <= room:
                rows[r].append(i)
                free[r] -= lengths[i]
                break
        else:
            rows.append([i])
            free.append(block_size - lengths[i])
    return rows


def pack_examples(
    examples: list[tuple[list[int], list[int]]],
    block_size: int,
    pad_id: int = 0,
    rows: list[list[int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Build packed ``(R, T)`` tensors from ``(input_ids, response_mask)`` examples.

    Args:
        examples: one ``(ids, response_mask)`` pair per example; both lists of
            ints with the same length.
        block_size: row length ``T``.
        pad_id: token id used to fill the unused tail of each row.
        rows: which examples go in which row. Defaults to
            :func:`first_fit_decreasing` on the example lengths.

    Returns:
        Dict of ``torch.long`` CPU tensors, each shape ``(R, T)``:

        * ``input_ids`` - the packed tokens, ``pad_id`` in the tail.
        * ``response_mask`` - 1 on response tokens, 0 on prompt and padding.
        * ``doc_ids`` - which example (0, 1, 2 ... within the row) owns each
          position, ``-1`` on padding.
        * ``position_ids`` - restarts at 0 for each example, 0 on padding.
    """
    if rows is None:
        rows = first_fit_decreasing([len(ids) for ids, _ in examples], block_size)
    R = len(rows)
    out = {
        "input_ids": torch.full((R, block_size), pad_id, dtype=torch.long),
        "response_mask": torch.zeros((R, block_size), dtype=torch.long),
        "doc_ids": torch.full((R, block_size), PAD_DOC, dtype=torch.long),
        "position_ids": torch.zeros((R, block_size), dtype=torch.long),
    }
    for r, members in enumerate(rows):
        start = 0
        for d, i in enumerate(members):
            ids, mask = examples[i]
            if len(ids) != len(mask):
                raise ValueError(f"example {i}: ids and mask lengths differ")
            end = start + len(ids)
            if end > block_size:
                raise ValueError(f"row {r} overflows block_size {block_size}")
            out["input_ids"][r, start:end] = torch.tensor(ids, dtype=torch.long)
            out["response_mask"][r, start:end] = torch.tensor(mask, dtype=torch.long)
            out["doc_ids"][r, start:end] = d
            out["position_ids"][r, start:end] = torch.arange(len(ids))
            start = end
    return out


def packed_attention_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Additive block-diagonal causal mask from ``doc_ids``.

    Query ``i`` may attend to key ``j`` iff ``j <= i`` and both belong to the
    same (non-padding) example. A padding query attends only to itself, so its
    softmax row is never all ``-inf`` (which would give NaN).

    Args:
        doc_ids: shape ``(..., T)`` long, ``-1`` on padding.

    Returns:
        Float tensor of shape ``(..., 1, T, T)`` on ``doc_ids``' device: ``0.0``
        where attention is allowed, ``-inf`` where it is forbidden. The singleton
        axis broadcasts over attention heads.
    """
    T = doc_ids.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
    same = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)          # (..., T, T)
    real = (doc_ids != PAD_DOC).unsqueeze(-1)                      # (..., T, 1)
    allowed = causal & same & real
    eye = torch.eye(T, dtype=torch.bool, device=doc_ids.device)
    allowed = allowed | (eye & ~real)                              # pad -> self only
    mask = torch.zeros(allowed.shape, dtype=torch.float32, device=doc_ids.device)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask.unsqueeze(-3)


def packed_labels(
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    doc_ids: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Next-token labels for a packed row, aligned with the logits.

    ``labels[..., t]`` is the target for ``logits[..., t]``: the token at
    ``t + 1`` if it is a response token of the *same* example, else
    ``ignore_index``. The last position has no next token and is ignored.

    Args:
        input_ids, response_mask, doc_ids: shape ``(..., T)`` long.

    Returns:
        Long tensor of shape ``(..., T)`` on the same device.
    """
    labels = torch.full_like(input_ids, ignore_index)
    nxt_ids = input_ids[..., 1:]
    keep = (response_mask[..., 1:] != 0) & (doc_ids[..., 1:] == doc_ids[..., :-1])
    keep = keep & (doc_ids[..., 1:] != PAD_DOC)
    labels[..., :-1] = torch.where(keep, nxt_ids, torch.full_like(nxt_ids, ignore_index))
    return labels


def packing_efficiency(doc_ids: torch.Tensor) -> float:
    """Fraction of row positions that hold a real (non-padding) token."""
    return float((doc_ids != PAD_DOC).float().mean())
