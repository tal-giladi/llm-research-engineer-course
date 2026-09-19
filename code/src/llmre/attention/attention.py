"""Scaled dot-product attention and causal multi-head self-attention.

This is the attention core of Module 5 (`lessons/module-05/lesson-02.md` and
`lesson-03.md`). We implement the attention formula from scratch first, then
wrap it into the ``CausalSelfAttention`` module that a GPT ``Block`` uses.

The one formula everything below unpacks is

    Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_h) + mask ) V

with an additive causal mask that sets every "future" score to -inf so a
position can only attend to itself and earlier positions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmre.model.config import GPTConfig


def scaled_dot_product_attention(q, k, v, mask=None):
    """Scaled dot-product attention, implemented from scratch.

    Computes ``softmax(q @ k^T / sqrt(d_h) + mask) @ v`` over the last two axes,
    treating every leading axis as a batch axis.

    Args:
        q: query tensor, shape ``(..., T, d_h)``, any float dtype/device.
        k: key tensor, shape ``(..., S, d_h)`` (``S`` keys; ``S == T`` for
            self-attention). Same dtype/device as ``q``.
        v: value tensor, shape ``(..., S, d_v)``. Same dtype/device as ``q``.
        mask: optional additive mask broadcastable to the score shape
            ``(..., T, S)``. Entries are ``0`` where attention is allowed and
            ``-inf`` (or a large negative number) where it is forbidden. ``None``
            means full (bidirectional) attention.

    Returns:
        Tensor of shape ``(..., T, d_v)`` — one attended value vector per query
        position — with the same dtype/device as the inputs.

    Note:
        The scale is ``1 / sqrt(d_h)`` where ``d_h = q.size(-1)``. Dividing keeps
        the pre-softmax scores at a moderate magnitude regardless of head size,
        so softmax does not saturate.
    """
    d_h = q.size(-1)
    # (..., T, d_h) @ (..., d_h, S) -> (..., T, S): every query dotted with every key.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_h)
    if mask is not None:
        # Additive mask: allowed positions get +0, forbidden positions get -inf,
        # which become exactly-zero weight after softmax (exp(-inf) = 0).
        scores = scores + mask
    # Softmax over the key axis (last dim): each query row becomes weights summing to 1.
    weights = F.softmax(scores, dim=-1)
    # Weighted sum of value vectors: (..., T, S) @ (..., S, d_v) -> (..., T, d_v).
    return weights @ v


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention over a ``(B, T, C)`` sequence.

    Projects the input into queries, keys and values, splits the channel axis
    into ``n_head`` heads, runs scaled dot-product attention per head under a
    causal mask, merges the heads and applies an output projection.

    Args:
        cfg: a ``GPTConfig``. Uses ``n_embd`` (``C``), ``n_head`` (``nh``),
            ``block_size`` (max ``T`` for the mask buffer), ``dropout`` and
            ``bias``. Requires ``n_embd % n_head == 0``.

    Shapes (forward):
        input  x: ``(B, T, C)`` float.
        output   : ``(B, T, C)`` float, same dtype/device as ``x``.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.d_h = cfg.n_embd // cfg.n_head
        # One fused Linear producing Q, K and V stacked along the channel axis:
        # (B, T, C) -> (B, T, 3C). Cheaper than three separate matmuls.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Output projection W_O: mixes the concatenated head outputs, (C -> C).
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout
        # Causal mask as a (1, 1, block_size, block_size) lower-triangular
        # buffer of 0 / -inf. Registered (not a Parameter) so it moves with
        # .to(device) but carries no gradient and is not saved as a weight.
        mask = torch.triu(
            torch.full((cfg.block_size, cfg.block_size), float("-inf")), diagonal=1
        )
        self.register_buffer("attn_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        """Apply causal multi-head self-attention.

        Args:
            x: input activations, shape ``(B, T, C)`` float, ``T <= block_size``.

        Returns:
            Tensor of shape ``(B, T, C)`` — the attention output, ready to be
            added back into the residual stream.
        """
        B, T, C = x.shape
        # Project to Q, K, V in one matmul, then split the 3C axis into three (B, T, C).
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Split channels into heads and move the head axis next to the batch axis:
        # (B, T, C) -> (B, T, nh, d_h) -> (B, nh, T, d_h).
        q = q.view(B, T, self.n_head, self.d_h).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_h).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_h).transpose(1, 2)
        # Slice the causal mask to the current T; it broadcasts over (B, nh).
        mask = self.attn_mask[:, :, :T, :T]
        y = scaled_dot_product_attention(q, k, v, mask=mask)  # (B, nh, T, d_h)
        y = self.attn_dropout(y)
        # Merge heads back: (B, nh, T, d_h) -> (B, T, nh, d_h) -> (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Output projection + residual dropout.
        y = self.resid_dropout(self.c_proj(y))
        return y
