"""DeepSeek Sparse Attention (DSA): lightning indexer + top-k selection (Module 12, lesson 06).

Extends the dense causal attention of :mod:`llmre.attention.gqa` (lesson 12.3), which
itself wraps the from-scratch :func:`llmre.attention.attention.scaled_dot_product_attention`
of lesson 05.2. Nothing there is changed: with ``topk >= T`` the layer below computes
exactly the same function as :class:`~llmre.attention.gqa.GroupedQueryAttention` with the
same weights (the test suite checks this).

The idea (DeepSeek-V3.2, arXiv 2512.02556, eqs. 1-4):

1. A small **lightning indexer** scores every past token ``s`` for query ``t``::

       I[t, s] = sum_j  w[t, j] * ReLU(q_idx[t, j] . k_idx[s])

   with ``H_I`` indexer heads of width ``d_I`` (few, small, cheap; FP8 in production).
2. Each query keeps only the ``topk`` past tokens with the highest ``I[t, :]``.
3. The main attention runs over that subset only: ``O(T * topk)`` score pairs instead of
   ``O(T^2)``. The indexer itself is still ``O(T^2)`` but each pair costs far less.
4. The indexer is trained to imitate the main attention: KL(p_t || softmax(I_t)), where
   ``p_t`` is the main attention distribution summed over heads and L1-normalized. In the
   *dense warm-up* stage only the indexer trains; in the *sparse* stage the KL is taken over
   the selected set and the indexer's input is detached, so the LM loss never reaches the
   indexer and the KL never reaches the main model.

Everything here is a readable reference on toy sizes (it builds the full ``(T, T)`` score
matrix and masks it). Production kernels (FlashMLA sparse, DeepGEMM indexer logits) gather
only the selected keys instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmre.attention.attention import scaled_dot_product_attention
from llmre.model.config import GPTConfig


def lightning_indexer_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Index scores ``I[t, s] = sum_j w[t, j] * ReLU(q_idx[t, j] . k_idx[s])``, causally masked.

    Args:
        q_idx: indexer queries, shape ``(B, T, H_I, d_I)``.
        k_idx: indexer keys, shape ``(B, T, d_I)`` (one key per token, shared by all
            indexer heads).
        w: per-query head weights, shape ``(B, T, H_I)``.

    Returns:
        ``(B, T, T)`` scores, same dtype/device as the inputs; entries with ``s > t``
        (the future) are ``-inf``.
    """
    T = q_idx.shape[1]
    dots = torch.einsum("bthd,bsd->bths", q_idx, k_idx)  # (B, T, H_I, T)
    scores = (w[..., None] * F.relu(dots)).sum(dim=2)  # (B, T, T)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q_idx.device), diagonal=1)
    return scores.masked_fill(future, float("-inf"))


def topk_selection_mask(index_scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Boolean ``(B, T, T)`` mask: True where key ``s`` is among query ``t``'s top-``topk``.

    Only causal positions (``s <= t``) can be selected, so row ``t`` has exactly
    ``min(t + 1, topk)`` True entries. Ties are broken by ``torch.topk`` (lower index
    first in practice); the lesson's examples avoid ties at the boundary.
    """
    B, T, _ = index_scores.shape
    k = min(topk, T)
    idx = index_scores.topk(k, dim=-1).indices  # (B, T, k)
    mask = torch.zeros(B, T, T, dtype=torch.bool, device=index_scores.device)
    mask.scatter_(-1, idx, True)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=index_scores.device))
    return mask & causal  # rows with t + 1 < k picked some -inf (future) entries: drop them


def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, select: torch.Tensor) -> torch.Tensor:
    """Attention restricted to the selected keys (reference: mask, then dense SDPA).

    Args:
        q: ``(B, H, T, d)``; k: ``(B, H, T, d)``; v: ``(B, H, T, d_v)``.
        select: ``(B, T, T)`` bool from :func:`topk_selection_mask` (shared by all heads,
            as in DSA, where the selection is made once per query token).

    Returns:
        ``(B, H, T, d_v)``.
    """
    mask = torch.zeros(select.shape, dtype=q.dtype, device=q.device)
    mask = mask.masked_fill(~select, float("-inf"))[:, None]  # (B, 1, T, T)
    return scaled_dot_product_attention(q, k, v, mask=mask)


def sparse_attention_gather(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, index_scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Same function as :func:`sparse_attention`, computed the way a kernel does it.

    For each query, gather its ``topk`` selected keys/values into a small ``(topk, d)``
    block and attend over that block only, so the score matrix is ``(T, topk)``, never
    ``(T, T)``. Slots that would point into the future (early rows, ``t + 1 < topk``)
    are masked out.
    """
    B, H, T, d = q.shape
    k_sel = min(topk, T)
    top = index_scores.topk(k_sel, dim=-1)  # (B, T, k)
    idx = top.indices
    valid = torch.isfinite(top.values)  # False where a future (-inf) slot was picked
    # Gather: (B, H, T, k, d) — row t holds the keys of its own selected positions.
    gidx = idx[:, None, :, :, None].expand(B, H, T, k_sel, d)
    k_g = k[:, :, None].expand(B, H, T, T, d).gather(3, gidx)
    v_g = v[:, :, None].expand(B, H, T, T, v.shape[-1]).gather(
        3, idx[:, None, :, :, None].expand(B, H, T, k_sel, v.shape[-1])
    )
    scores = (q[:, :, :, None, :] * k_g).sum(-1) / d**0.5  # (B, H, T, k)
    scores = scores.masked_fill(~valid[:, None], float("-inf"))
    p = scores.softmax(dim=-1)
    return (p[..., None] * v_g).sum(dim=-2)


def attention_target_distribution(attn_probs: torch.Tensor) -> torch.Tensor:
    """DSA's indexer target: main attention probabilities summed over heads, L1-normalized.

    Args:
        attn_probs: ``(B, H, T, T)`` softmax weights of the main attention.

    Returns:
        ``(B, T, T)``; every row sums to 1.
    """
    p = attn_probs.sum(dim=1)
    return p / p.sum(dim=-1, keepdim=True)


def indexer_kl_loss(target: torch.Tensor, index_scores: torch.Tensor, select: torch.Tensor | None = None) -> torch.Tensor:
    """``sum_t KL(p_t || softmax(I_t))``, averaged over the batch (DSA eqs. 3 and 4).

    Args:
        target: ``(B, T, T)`` rows of ``p`` from :func:`attention_target_distribution`.
        index_scores: ``(B, T, T)`` indexer scores (``-inf`` on the future).
        select: optional ``(B, T, T)`` bool. If given (sparse stage, eq. 4), both ``p`` and
            ``softmax(I)`` are restricted to the selected set and renormalized there.

    Returns:
        Scalar tensor. Zero exactly when ``softmax(I_t) == p_t`` on every row.
    """
    allowed = torch.isfinite(index_scores)
    if select is not None:
        allowed = allowed & select
    scores = index_scores.masked_fill(~allowed, float("-inf"))
    log_q = scores.log_softmax(dim=-1)
    p = target.masked_fill(~allowed, 0.0)
    p = p / p.sum(dim=-1, keepdim=True)
    terms = torch.where(p > 0, p * (torch.log(p.clamp_min(1e-30)) - log_q), torch.zeros_like(p))
    return terms.sum(dim=(-1, -2)).mean()


class DSAttention(nn.Module):
    """Causal self-attention with a lightning indexer and top-k key selection.

    The main attention is :class:`~llmre.attention.gqa.GroupedQueryAttention`'s
    (same projection names, same KV-head sharing); DSA in DeepSeek-V3.2 uses MLA in its
    MQA mode, i.e. one shared KV head, which is ``n_kv_head=1`` here.

    Args:
        cfg: ``GPTConfig`` (``n_embd``, ``n_head``, ``bias``).
        n_kv_head: KV heads ``g`` for the main attention (default 1 = MQA, as in DSA).
        n_index_head: indexer heads ``H_I``.
        d_index: indexer head width ``d_I``.
        topk: keys kept per query.

    Shapes: ``x`` ``(B, T, C)`` -> ``(B, T, C)``.
    """

    def __init__(self, cfg: GPTConfig, n_kv_head: int = 1, n_index_head: int = 2, d_index: int = 8, topk: int = 16):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % n_kv_head == 0
        self.n_head, self.n_kv_head = cfg.n_head, n_kv_head
        self.n_rep = cfg.n_head // n_kv_head
        self.d_h = cfg.n_embd // cfg.n_head
        self.topk, self.n_index_head, self.d_index = topk, n_index_head, d_index
        # Main attention: same layout and names as GroupedQueryAttention.
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.d_h, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, n_kv_head * self.d_h, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, n_kv_head * self.d_h, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        # Lightning indexer: H_I small query heads, one shared key, one weight per head.
        self.idx_q = nn.Linear(cfg.n_embd, n_index_head * d_index, bias=False)
        self.idx_k = nn.Linear(cfg.n_embd, d_index, bias=False)
        self.idx_w = nn.Linear(cfg.n_embd, n_index_head, bias=False)

    def index_scores(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, C)`` -> ``(B, T, T)`` lightning-indexer scores (future = ``-inf``)."""
        B, T, _ = x.shape
        q_idx = self.idx_q(x).view(B, T, self.n_index_head, self.d_index)
        return lightning_indexer_scores(q_idx, self.idx_k(x), self.idx_w(x))

    def _qkv(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_h).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_h).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_h).transpose(1, 2)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        return q, k, v

    def forward(self, x: torch.Tensor, dense: bool = False, return_aux: bool = False):
        """Run the layer.

        Args:
            x: ``(B, T, C)``.
            dense: True = dense warm-up stage (attend to every past token; the indexer is
                scored and can be trained against the dense attention). False = sparse
                stage (attend only to the indexer's top-k). In both stages the indexer
                reads ``x.detach()``: the indexer loss trains only the indexer, and
                (because top-k selection has no gradient) the LM loss trains only the
                main attention.
            return_aux: also return a dict with ``index_scores``, ``select``, ``target``
                (head-summed attention distribution, detached) and ``indexer_loss``.

        Returns:
            ``y`` ``(B, T, C)``, or ``(y, aux)`` when ``return_aux``.
        """
        B, T, C = x.shape
        q, k, v = self._qkv(x)
        scores = self.index_scores(x.detach())  # the KL never reaches the main model
        select = topk_selection_mask(scores, T if dense else self.topk)

        mask = torch.zeros(select.shape, dtype=x.dtype, device=x.device).masked_fill(~select, float("-inf"))
        att = (q @ k.transpose(-2, -1)) / self.d_h**0.5 + mask[:, None]
        probs = att.softmax(dim=-1)  # (B, H, T, T); zero outside the selected set
        y = (probs @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        if not return_aux:
            return y
        target = attention_target_distribution(probs.detach())
        loss = indexer_kl_loss(target, scores, None if dense else select)
        return y, {"index_scores": scores, "select": select, "target": target, "indexer_loss": loss}


def selected_pairs(T: int, topk: int) -> tuple[int, int]:
    """(dense causal score pairs, DSA main-attention score pairs) for one sequence of ``T``."""
    dense = T * (T + 1) // 2
    sparse = sum(min(t + 1, topk) for t in range(T))
    return dense, sparse


def per_pair_macs(n_head: int, d_qk: int, d_v: int) -> int:
    """Multiply-adds per (query, key) pair for one layer: scores ``n_head*d_qk`` + values ``n_head*d_v``.

    For the indexer pass ``d_v=0`` (it produces scores only) and ``n_head=H_I``, ``d_qk=d_I``.
    """
    return n_head * (d_qk + d_v)


def recall_at_k(index_scores: torch.Tensor, attn_target: torch.Tensor, topk: int) -> float:
    """Fraction of the main attention's own top-``topk`` keys that the indexer also selects.

    A diagnostic for how well a (warm-up-trained) indexer imitates the attention it replaces.
    Rows with ``t + 1 <= topk`` are trivially perfect and are skipped.
    """
    T = index_scores.shape[-1]
    if T <= topk:
        return 1.0
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=attn_target.device), diagonal=1)
    sel = topk_selection_mask(index_scores, topk)[:, topk:]
    true = topk_selection_mask(attn_target.masked_fill(future, float("-inf")), topk)[:, topk:]
    return ((sel & true).sum() / true.sum()).item()
