"""Loss masking for SFT: compute the loss on the response tokens only.

During pretraining (Module 7) we compute cross-entropy on *every* next-token
position — the model should predict all of the text. During supervised
fine-tuning we do **not** want to reward the model for reproducing the user's
prompt; we only want it to learn the *response given the prompt*. So we compute
the loss on the assistant-response positions only and ignore the prompt
positions.

The mechanism is the standard "ignore index" trick: build a ``labels`` tensor
that equals the target token id on response positions and a sentinel
``ignore_index`` (conventionally ``-100``, HuggingFace's default) on prompt
positions. Cross-entropy then skips every ignored position.

This module (Module 14, lesson 14.2) implements the masked loss from scratch on
top of a numerically stable log-softmax, exactly as
:func:`llmre.evaluation.metrics.cross_entropy` does for the unmasked case.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def build_labels(
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Turn a response mask into a ``labels`` tensor with prompt positions ignored.

    Args:
        input_ids: token ids, shape ``(...,)`` (typically ``(T,)`` or ``(B, T)``)
            dtype ``torch.long``, any device. The target for a response position
            is the id sitting there.
        response_mask: same shape as ``input_ids``, dtype bool or int. A truthy
            entry marks a position the model should be trained to produce (the
            assistant response); a falsy entry marks a prompt position to ignore.
        ignore_index: sentinel written into ignored positions. Must match the
            ``ignore_index`` later passed to :func:`masked_cross_entropy`.

    Returns:
        ``labels``, an ``int64`` tensor the same shape/device as ``input_ids``:
        ``labels[i] = input_ids[i]`` where ``response_mask[i]`` is truthy, else
        ``ignore_index``.
    """
    mask = response_mask.to(torch.bool)
    labels = torch.full_like(input_ids, ignore_index)
    labels[mask] = input_ids[mask]
    return labels


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Mean cross-entropy (nats) over positions whose label is not ``ignore_index``.

    Positions are aligned one-to-one: ``logits[..., i, :]`` is scored against
    ``labels[..., i]``. (In a full training loop you shift the sequence by one so
    that position ``i`` predicts token ``i+1``; do that shift before calling this
    and pass the already-aligned ``labels``.)

    Args:
        logits: shape ``(..., V)`` — e.g. ``(T, V)`` or ``(B, T, V)`` — dtype
            float, any device. ``V`` is the vocabulary size.
        labels: shape ``(...)`` matching ``logits`` without its last axis, dtype
            ``torch.long``. Entries equal to ``ignore_index`` are dropped.
        ignore_index: the sentinel marking positions to skip.

    Returns:
        Scalar tensor (shape ``()``), same dtype/device as ``logits``: the mean
        cross-entropy in nats over the kept positions. Equals
        :func:`llmre.evaluation.metrics.cross_entropy` computed on just the kept
        ``(logits, labels)`` rows. If no position survives, returns ``0.0``.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])   # (N, V)
    flat_labels = labels.reshape(-1)                     # (N,)
    keep = flat_labels != ignore_index                   # (N,) bool
    if keep.sum() == 0:
        return logits.new_zeros(())
    kept_logits = flat_logits[keep]                      # (M, V)
    kept_labels = flat_labels[keep]                      # (M,)

    # Numerically stable log-softmax, then gather the true-token log-prob.
    logp = kept_logits - torch.logsumexp(kept_logits, dim=-1, keepdim=True)  # (M, V)
    rows = torch.arange(kept_labels.shape[0], device=kept_labels.device)
    true_logp = logp[rows, kept_labels]                  # (M,)
    return -true_logp.mean()                             # scalar, nats
