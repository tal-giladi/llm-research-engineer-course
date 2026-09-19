"""Grouped-Query Attention (GQA) — MHA / MQA / GQA in one module (Module 12, lesson 03).

Standard multi-head attention (Module 5, ``CausalSelfAttention``) gives every
query head its *own* key and value head: ``n_head`` queries, ``n_head`` keys,
``n_head`` values. At inference time the keys and values of every past token are
cached (the **KV cache**) so they are not recomputed each step, and that cache
holds ``2 · n_layer · n_head · T · hd`` numbers — it grows with the number of KV
heads and quickly dominates memory for long contexts.

Two variants shrink it:

* **MQA** (Multi-Query Attention, Shazeer 2019): all query heads share a *single*
  key/value head. The KV cache shrinks by a factor of ``n_head``.
* **GQA** (Grouped-Query Attention, Ainslie et al. 2023; used by Llama 3,
  ``papers/index.md`` #10): use ``g`` key/value heads, each shared by a group of
  ``n_head / g`` query heads. It interpolates between the two extremes:
  ``g == n_head`` is plain MHA, ``g == 1`` is MQA.

The query heads still all compute independently; the smaller KV set is just
*repeated* (broadcast) across each group so the per-head attention math is
unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llmre.attention.attention import scaled_dot_product_attention
from llmre.attention.rope import apply_rope
from llmre.model.config import GPTConfig


class GroupedQueryAttention(nn.Module):
    """Causal grouped-query self-attention over a ``(B, T, C)`` sequence.

    Projects the input to ``n_head`` query heads but only ``n_kv_head`` key/value
    heads, runs causal scaled-dot-product attention per query head (sharing each
    KV head across its group of query heads), then an output projection.

    Args:
        cfg: a ``GPTConfig``. Uses ``n_embd`` (``C``), ``n_head`` (``nh``),
            ``dropout`` and ``bias``. The per-head dim is ``hd = n_embd // n_head``.
        n_kv_head: number ``g`` of key/value heads. Must divide ``n_head``.
            ``None`` (default) means ``g = n_head`` (plain MHA). ``g = 1`` is MQA.

    Shapes (forward):
        input  x: ``(B, T, C)`` float.
        output   : ``(B, T, C)`` float, same dtype/device as ``x``.
    """

    def __init__(self, cfg: GPTConfig, n_kv_head: int | None = None):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_head if n_kv_head is None else n_kv_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_rep = self.n_head // self.n_kv_head  # query heads per KV head
        self.d_h = cfg.n_embd // cfg.n_head
        self.n_embd = cfg.n_embd

        # Separate projections: Q is full width, K and V are the smaller KV width.
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.d_h, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.d_h, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.d_h, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None = None) -> torch.Tensor:
        """Apply causal grouped-query attention.

        Args:
            x: input activations, shape ``(B, T, C)`` float.
            freqs_cis: optional RoPE factors (see
                :func:`llmre.attention.rope.precompute_freqs_cis`), complex, shape
                ``(>=T, hd//2)``. If given, RoPE is applied to Q and K before
                attention. ``None`` means no positional rotation.

        Returns:
            Tensor of shape ``(B, T, C)`` — the attention output.
        """
        B, T, C = x.shape
        # Project. Q -> (B, nh, T, hd); K, V -> (B, n_kv, T, hd).
        q = self.q_proj(x).view(B, T, self.n_head, self.d_h).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_h).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_h).transpose(1, 2)

        if freqs_cis is not None:
            # RoPE rotates Q and K; apply per their own head counts.
            q, _ = apply_rope(q, q, freqs_cis)
            k, _ = apply_rope(k, k, freqs_cis)

        # Share each KV head across its group of n_rep query heads. repeat_interleave
        # along the head axis turns (B, n_kv, T, hd) into (B, nh, T, hd): KV head j
        # is copied to query heads j*n_rep .. j*n_rep + n_rep - 1.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Causal mask: 0 on/below the diagonal, -inf above. Broadcasts over (B, nh).
        mask = torch.triu(
            torch.full((T, T), float("-inf"), dtype=x.dtype, device=x.device), diagonal=1
        )
        y = scaled_dot_product_attention(q, k, v, mask=mask)  # (B, nh, T, hd)
        y = self.attn_dropout(y)
        # Merge heads: (B, nh, T, hd) -> (B, T, nh, hd) -> (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
