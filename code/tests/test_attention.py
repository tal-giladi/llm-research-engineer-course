"""Tests for llmre.attention — shapes, causality, and parity with torch.

These back the claims of `lessons/module-05/lesson-02.md` and `lesson-03.md`:
the from-scratch attention has the right shape, respects the causal mask
(no position sees its future), and reproduces PyTorch's fused kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llmre.attention.attention import CausalSelfAttention, scaled_dot_product_attention
from llmre.model.config import GPTConfig


def _cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=100, block_size=16, n_layer=1, n_head=4, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_causal_self_attention_forward_shape():
    torch.manual_seed(0)
    cfg = _cfg()
    attn = CausalSelfAttention(cfg)
    B, T, C = 2, 5, cfg.n_embd
    x = torch.randn(B, T, C)
    y = attn(x)
    assert y.shape == (B, T, C)
    assert y.dtype == x.dtype


def test_causality_future_does_not_leak_into_past():
    """Perturbing the token at position j must not change any output at pos < j."""
    torch.manual_seed(1)
    cfg = _cfg()
    attn = CausalSelfAttention(cfg).eval()  # eval() so dropout is a no-op
    B, T, C = 1, 6, cfg.n_embd
    x = torch.randn(B, T, C)

    j = 3  # perturb this position
    x2 = x.clone()
    x2[:, j, :] += 10.0 * torch.randn(C)

    with torch.no_grad():
        y1 = attn(x)
        y2 = attn(x2)

    # Outputs at positions strictly before j must be identical (up to fp error);
    # a causal model cannot let position j influence earlier positions.
    assert torch.allclose(y1[:, :j, :], y2[:, :j, :], atol=1e-6, rtol=0)
    # Sanity: the perturbation DID change the output at position j (else the test
    # would pass trivially for a broken all-zeros module).
    assert not torch.allclose(y1[:, j, :], y2[:, j, :], atol=1e-6, rtol=0)


def test_scaled_dot_product_attention_matches_torch_causal():
    torch.manual_seed(2)
    B, nh, T, d_h = 2, 4, 7, 8
    q = torch.randn(B, nh, T, d_h, dtype=torch.float64)
    k = torch.randn(B, nh, T, d_h, dtype=torch.float64)
    v = torch.randn(B, nh, T, d_h, dtype=torch.float64)

    # Our additive causal mask: 0 on/below the diagonal, -inf above it.
    mask = torch.triu(torch.full((T, T), float("-inf"), dtype=torch.float64), diagonal=1)
    ours = scaled_dot_product_attention(q, k, v, mask=mask)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5, rtol=0)


def test_scaled_dot_product_attention_matches_torch_unmasked():
    torch.manual_seed(3)
    B, T, d_h = 3, 5, 6
    q = torch.randn(B, T, d_h, dtype=torch.float64)
    k = torch.randn(B, T, d_h, dtype=torch.float64)
    v = torch.randn(B, T, d_h, dtype=torch.float64)
    ours = scaled_dot_product_attention(q, k, v, mask=None)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(ours, ref, atol=1e-5, rtol=0)
