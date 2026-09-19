"""The transformer building blocks: LayerNorm, MLP, and the pre-norm Block.

This is the assembly stage of Module 5 (`lessons/module-05/lesson-04.md`). A
GPT is just ``n_layer`` copies of ``Block`` stacked on top of the embeddings
(Module 6). Each block is two residual sub-layers:

    x = x + attn(ln1(x))     # communication across positions
    x = x + mlp(ln2(x))      # per-position computation

That "normalize, transform, add back" shape is the pre-norm transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llmre.attention.attention import CausalSelfAttention
from llmre.model.config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm over the last dimension, from scratch, with optional bias.

    Normalizes each length-``C`` feature vector to zero mean and unit variance,
    then applies a learned per-channel scale ``gamma`` and shift ``beta``:

        y = (x - mean) / sqrt(var + eps) * gamma + beta

    where ``mean`` and ``var`` are taken over the last axis only (per position,
    per sequence), using the biased (population) variance to match
    ``torch.nn.LayerNorm``.

    Args:
        ndim: size ``C`` of the normalized last dimension.
        bias: if ``True`` include the learned shift ``beta``; else it is 0.
        eps: constant added inside the sqrt for numerical stability.

    Shapes:
        input / output ``(..., C)``, same dtype/device; ``gamma``/``beta`` are
        ``(C,)`` parameters.
    """

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        # Mean and (biased) variance over the last axis, keeping dims for broadcast.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = x_hat * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class MLP(nn.Module):
    """Position-wise feed-forward network: Linear C->4C, GELU, Linear 4C->C.

    Applied independently to every position. The 4x hidden expansion gives the
    block capacity to compute nonlinear per-token features; GELU is the
    nonlinearity (without it the two Linears would collapse into one).

    Args:
        cfg: a ``GPTConfig``; uses ``n_embd`` (``C``), ``bias`` and ``dropout``.

    Shapes:
        input / output ``(B, T, C)``; hidden activation ``(B, T, 4C)``.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)       # (B, T, C) -> (B, T, 4C)
        x = self.gelu(x)       # elementwise nonlinearity
        x = self.c_proj(x)     # (B, T, 4C) -> (B, T, C)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One pre-norm transformer block.

    Two residual sub-layers, each ``x = x + sublayer(norm(x))``:
        1. causal multi-head self-attention  (mixes information across positions)
        2. position-wise MLP                  (transforms each position)

    Args:
        cfg: a ``GPTConfig`` passed straight through to the sub-modules.

    Shapes:
        input / output ``(B, T, C)`` float, unchanged shape/dtype/device.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # Pre-norm: normalize BEFORE each sub-layer; the residual add keeps the
        # raw x on a clean gradient highway straight through the network.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
