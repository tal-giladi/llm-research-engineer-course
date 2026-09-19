"""Tests for llmre.attention.flash — the FlashAttention algorithm on CPU.

These back the claims of `lessons/module-08/lesson-04.md`: the block-tiled,
online-softmax implementation computes *exactly* the same thing as a plain
``softmax(QKᵀ/√d + mask) V`` and as PyTorch's fused kernel — both non-causal and
causal — to 1e-4. The whole point of FlashAttention is that it changes the memory
traffic, not the math, so parity is the property that must hold.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from llmre.attention.flash import flash_attention_reference


def _naive_attention(q, k, v, causal=False):
    """The straightforward reference: build the full T×T matrix and softmax it."""
    B, nh, T, hd = q.shape
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, nh, T, T)
    if causal:
        mask = torch.triu(
            torch.full((T, T), float("-inf"), dtype=q.dtype, device=q.device),
            diagonal=1,
        )
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    return weights @ v


def _qkv(B, nh, T, hd, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, nh, T, hd, generator=g, dtype=torch.float64)
    k = torch.randn(B, nh, T, hd, generator=g, dtype=torch.float64)
    v = torch.randn(B, nh, T, hd, generator=g, dtype=torch.float64)
    return q, k, v


def test_flash_matches_naive_noncausal():
    q, k, v = _qkv(2, 3, 20, 8, seed=0)
    out = flash_attention_reference(q, k, v, causal=False, block_size=8)
    ref = _naive_attention(q, k, v, causal=False)
    assert torch.allclose(out, ref, atol=1e-4, rtol=0)


def test_flash_matches_naive_causal():
    q, k, v = _qkv(2, 3, 20, 8, seed=1)
    out = flash_attention_reference(q, k, v, causal=True, block_size=8)
    ref = _naive_attention(q, k, v, causal=True)
    assert torch.allclose(out, ref, atol=1e-4, rtol=0)


def test_flash_matches_torch_sdpa_noncausal():
    q, k, v = _qkv(2, 4, 17, 16, seed=2)
    out = flash_attention_reference(q, k, v, causal=False, block_size=6)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref, atol=1e-4, rtol=0)


def test_flash_matches_torch_sdpa_causal():
    q, k, v = _qkv(2, 4, 17, 16, seed=3)
    out = flash_attention_reference(q, k, v, causal=True, block_size=6)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-4, rtol=0)


def test_flash_result_independent_of_block_size():
    """Tiling is an implementation detail: the answer must not depend on Bc."""
    q, k, v = _qkv(1, 2, 32, 8, seed=4)
    ref = _naive_attention(q, k, v, causal=True)
    for bs in (1, 3, 8, 16, 32, 64):
        out = flash_attention_reference(q, k, v, causal=True, block_size=bs)
        assert torch.allclose(out, ref, atol=1e-4, rtol=0), f"block_size={bs}"


def test_flash_handles_t_not_multiple_of_block():
    """T=17 with block_size=5 exercises ragged final blocks on both axes."""
    q, k, v = _qkv(1, 1, 17, 4, seed=5)
    ref = _naive_attention(q, k, v, causal=True)
    out = flash_attention_reference(q, k, v, causal=True, block_size=5)
    assert torch.allclose(out, ref, atol=1e-4, rtol=0)
