"""Mixture of Experts (MoE) — DeepSeek-style, from scratch (Module 12, lesson 04).

A dense transformer runs *every* token through the *same* MLP, so parameters and
per-token compute grow together. A **Mixture of Experts** breaks that link:
replace the one MLP with ``n_experts`` separate expert FFNs and a small
**router** that sends each token to only its ``top_k`` experts. Total parameters
scale with ``n_experts``, but each token still touches only ``top_k`` of them, so
FLOPs per token stay ~constant (sparse activation).

The catch is **load balancing**. The router is learned, and left alone it tends
to collapse onto a few favourite experts (they get better because they get more
tokens, so they get even more tokens). Two remedies appear here:

* an **auxiliary load-balance loss** (Switch Transformers, ``papers/index.md``
  #12) that pushes the token distribution toward uniform across experts;
* **shared experts** (DeepSeekMoE, #13) that are always on, capturing common
  knowledge so the routed experts can specialize.

DeepSeek-V3 (#14) additionally replaces the aux loss with a per-expert bias added
to the router logits, nudged up/down each step to equalize load without adding a
loss-gradient term (**aux-loss-free balancing** — PUBLICLY DOCUMENTED in the
DeepSeek-V3 technical report). This file implements the classic aux-loss form and
optional shared experts; the bias trick is discussed in the lesson.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmre.model.swiglu import SwiGLU


class MoE(nn.Module):
    """A sparse Mixture-of-Experts feed-forward layer with top-k routing.

    Each token is scored by a linear router over ``n_experts`` experts, routed to
    its ``top_k`` highest-scoring experts, and its output is the gate-weighted sum
    of those experts' outputs (plus any always-on shared experts). Returns the
    output and a scalar auxiliary load-balance loss.

    Args:
        dim: model / residual width ``C``.
        n_experts: number ``E`` of routed experts.
        top_k: number ``k`` of experts each token is routed to (``1`` = Switch-style).
        hidden_dim: inner width of each expert FFN (defaults to SwiGLU's ``(8/3)C``).
        n_shared: number of always-on shared experts (DeepSeekMoE); ``0`` disables.
        bias: whether the expert/router Linears carry a bias (router never does).

    Shapes (forward):
        input  x: ``(B, T, C)`` float.
        output   : ``(B, T, C)`` float, plus a scalar ``aux_loss`` tensor.
    """

    def __init__(
        self,
        dim: int,
        n_experts: int,
        top_k: int,
        hidden_dim: int | None = None,
        n_shared: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        assert 1 <= top_k <= n_experts, "need 1 <= top_k <= n_experts"
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        # The router: a bias-free Linear producing one score per expert.
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(dim, hidden_dim=hidden_dim, bias=bias) for _ in range(n_experts)]
        )
        self.shared = (
            nn.ModuleList([SwiGLU(dim, hidden_dim=hidden_dim, bias=bias) for _ in range(n_shared)])
            if n_shared > 0
            else None
        )

    def route(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score and select experts for a batch of tokens.

        Args:
            x_flat: tokens, shape ``(N, C)`` where ``N = B*T``.

        Returns:
            ``(topk_idx, topk_gate, probs)`` where ``topk_idx`` is ``(N, top_k)``
            long (the chosen expert ids, distinct per token), ``topk_gate`` is
            ``(N, top_k)`` float gate weights that sum to 1 per token, and
            ``probs`` is the full ``(N, E)`` router softmax (used by the aux loss).
        """
        probs = F.softmax(self.router(x_flat), dim=-1)          # (N, E)
        topk_gate, topk_idx = probs.topk(self.top_k, dim=-1)    # (N, k) each
        # Renormalize the kept gates so each token's weights sum to 1.
        topk_gate = topk_gate / topk_gate.sum(dim=-1, keepdim=True)
        return topk_idx, topk_gate, probs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route ``x`` (shape ``(B, T, C)``) and return ``(output, aux_loss)``."""
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)                               # (N, C)
        N = x_flat.size(0)
        topk_idx, topk_gate, probs = self.route(x_flat)

        out = torch.zeros_like(x_flat)
        # Loop over experts: for each, gather the tokens that chose it (in any of
        # their top_k slots), run the expert once on that sub-batch, and scatter
        # the gate-weighted result back. This is the sparse dispatch.
        for e, expert in enumerate(self.experts):
            sel = topk_idx == e                                 # (N, k) bool
            token_mask = sel.any(dim=-1)                        # (N,)
            if not bool(token_mask.any()):
                continue
            gate = (topk_gate * sel).sum(dim=-1)                # (N,) gate for expert e
            xe = x_flat[token_mask]                             # (n_e, C)
            ye = expert(xe)                                     # (n_e, C)
            out[token_mask] += gate[token_mask].unsqueeze(-1) * ye

        # Shared experts (if any) process every token, always on.
        if self.shared is not None:
            for sh in self.shared:
                out = out + sh(x_flat)

        out = out.view(B, T, C)

        # --- auxiliary load-balance loss (Switch Transformers) ---
        # f_i: fraction of routing slots that went to expert i (a count -> detached).
        # P_i: mean router probability mass on expert i (carries the gradient).
        # aux = E * sum_i f_i * P_i is minimized when both are uniform (1/E each).
        assign = F.one_hot(topk_idx, self.n_experts).float().sum(dim=1)  # (N, E)
        f = assign.mean(dim=0)                                  # (E,) tokens-per-expert fraction
        P = probs.mean(dim=0)                                   # (E,) mean prob per expert
        aux_loss = self.n_experts * torch.sum(f.detach() * P)
        return out, aux_loss
