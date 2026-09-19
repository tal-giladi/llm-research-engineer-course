"""Tests for llmre.model.block — LayerNorm parity and Block shape preservation.

These back `lessons/module-05/lesson-04.md`: the from-scratch LayerNorm matches
``torch.nn.LayerNorm`` numerically, and a pre-norm ``Block`` maps ``(B, T, C)``
to ``(B, T, C)`` unchanged.
"""

from __future__ import annotations

import torch

from llmre.model.block import Block, LayerNorm, MLP
from llmre.model.config import GPTConfig


def _cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=100, block_size=16, n_layer=1, n_head=4, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_layernorm_matches_torch():
    torch.manual_seed(0)
    C = 32
    x = torch.randn(4, 7, C, dtype=torch.float64)

    ours = LayerNorm(C, bias=True, eps=1e-5).double()
    ref = torch.nn.LayerNorm(C, eps=1e-5).double()
    # Give both the same (non-trivial) affine parameters.
    with torch.no_grad():
        gamma = torch.randn(C, dtype=torch.float64)
        beta = torch.randn(C, dtype=torch.float64)
        ours.weight.copy_(gamma)
        ours.bias.copy_(beta)
        ref.weight.copy_(gamma)
        ref.bias.copy_(beta)

    assert torch.allclose(ours(x), ref(x), atol=1e-6, rtol=0)


def test_layernorm_no_bias_matches_torch():
    torch.manual_seed(1)
    C = 16
    x = torch.randn(3, 5, C, dtype=torch.float64)
    ours = LayerNorm(C, bias=False).double()
    ref = torch.nn.LayerNorm(C, elementwise_affine=True, bias=False).double()
    with torch.no_grad():
        gamma = torch.randn(C, dtype=torch.float64)
        ours.weight.copy_(gamma)
        ref.weight.copy_(gamma)
    assert torch.allclose(ours(x), ref(x), atol=1e-6, rtol=0)


def test_block_preserves_shape():
    torch.manual_seed(2)
    cfg = _cfg()
    block = Block(cfg).eval()
    B, T, C = 2, 6, cfg.n_embd
    x = torch.randn(B, T, C)
    y = block(x)
    assert y.shape == (B, T, C)
    assert y.dtype == x.dtype


def test_mlp_preserves_shape():
    torch.manual_seed(3)
    cfg = _cfg()
    mlp = MLP(cfg)
    B, T, C = 2, 4, cfg.n_embd
    x = torch.randn(B, T, C)
    assert mlp(x).shape == (B, T, C)
