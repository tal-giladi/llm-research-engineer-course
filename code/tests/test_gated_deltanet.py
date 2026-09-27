"""Tests for lesson 12.5: linear attention, the gated delta rule and hybrid-stack memory."""

import pytest
import torch
import torch.nn.functional as F

from llmre.attention.gated_deltanet import (
    GatedDeltaNet,
    causal_linear_attention_parallel,
    gated_delta_rule_recurrent,
    hybrid_decode_cache_bytes,
    hybrid_layer_pattern,
    kv_cache_bytes,
    linear_attention_recurrent,
    linear_state_bytes,
)
from llmre.model.config import GPTConfig


def _qkv(B=2, H=3, T=7, d_k=4, d_v=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, d_k, generator=g, dtype=torch.float64)
    k = F.normalize(torch.randn(B, H, T, d_k, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, d_v, generator=g, dtype=torch.float64)
    alpha = torch.rand(B, H, T, generator=g, dtype=torch.float64) * 0.5 + 0.5
    beta = torch.rand(B, H, T, generator=g, dtype=torch.float64)
    return q, k, v, alpha, beta


def test_linear_attention_recurrent_matches_parallel():
    q, k, v, _, _ = _qkv()
    o_rec, _ = linear_attention_recurrent(q, k, v)
    assert torch.allclose(o_rec, causal_linear_attention_parallel(q, k, v), atol=1e-10)


def test_decayed_linear_attention_recurrent_matches_parallel():
    q, k, v, alpha, _ = _qkv()
    o_rec, _ = linear_attention_recurrent(q, k, v, alpha=alpha)
    assert torch.allclose(o_rec, causal_linear_attention_parallel(q, k, v, alpha=alpha), atol=1e-10)


def test_hand_worked_example_from_lesson():
    # d_k = 2, d_v = 1. Write v=3 under key e1, then v=5 under the same key, then v=2 under e2.
    k = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64).view(1, 1, 3, 2)
    v = torch.tensor([3.0, 5.0, 2.0], dtype=torch.float64).view(1, 1, 3, 1)
    q = torch.tensor([[1.0, 0.0]] * 3, dtype=torch.float64).view(1, 1, 3, 2)
    ones = torch.ones(1, 1, 3, dtype=torch.float64)

    o_lin, _ = linear_attention_recurrent(q, k, v)
    assert o_lin.flatten().tolist() == [3.0, 8.0, 8.0]  # 3 + 5 collide

    o_delta, S = gated_delta_rule_recurrent(q, k, v, alpha=ones, beta=ones)
    assert o_delta.flatten().tolist() == [3.0, 5.0, 5.0]  # overwritten, not summed
    assert S.flatten().tolist() == [5.0, 2.0]

    alpha = torch.tensor([1.0, 0.5, 1.0], dtype=torch.float64).view(1, 1, 3)
    beta = torch.tensor([1.0, 0.5, 1.0], dtype=torch.float64).view(1, 1, 3)
    o_gated, _ = gated_delta_rule_recurrent(q, k, v, alpha=alpha, beta=beta)
    assert o_gated.flatten()[1].item() == pytest.approx(3.25)


def test_delta_rule_write_then_read_returns_value():
    # With a unit key, beta = 1 and alpha = 1, reading back the key just written gives v exactly.
    q, k, v, _, _ = _qkv(T=5)
    ones = torch.ones(q.shape[:-1], dtype=torch.float64)
    _, S = gated_delta_rule_recurrent(q, k, v, alpha=ones, beta=ones)
    read = (S @ k[..., -1, :, None]).squeeze(-1)
    assert torch.allclose(read, v[..., -1, :], atol=1e-10)


def test_beta_zero_never_writes():
    q, k, v, alpha, _ = _qkv()
    zeros = torch.zeros_like(alpha)
    o, S = gated_delta_rule_recurrent(q, k, v, alpha=alpha, beta=zeros)
    assert torch.count_nonzero(o) == 0 and torch.count_nonzero(S) == 0


def test_state_carry_equals_full_sequence():
    q, k, v, alpha, beta = _qkv(T=9)
    o_full, S_full = gated_delta_rule_recurrent(q, k, v, alpha, beta)
    o1, S1 = gated_delta_rule_recurrent(q[..., :4, :], k[..., :4, :], v[..., :4, :], alpha[..., :4], beta[..., :4])
    o2, S2 = gated_delta_rule_recurrent(
        q[..., 4:, :], k[..., 4:, :], v[..., 4:, :], alpha[..., 4:], beta[..., 4:], initial_state=S1
    )
    assert torch.allclose(torch.cat([o1, o2], dim=-2), o_full, atol=1e-10)
    assert torch.allclose(S2, S_full, atol=1e-10)


def _layer():
    torch.manual_seed(0)
    cfg = GPTConfig(n_embd=16, n_head=4, bias=False)
    return GatedDeltaNet(cfg).double()


def test_module_shape_and_causality():
    layer = _layer()
    x = torch.randn(2, 6, 16, dtype=torch.float64)
    y = layer(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    x2 = x.clone()
    x2[:, 4:] += 1.0  # change the future
    assert torch.allclose(layer(x2)[:, :4], y[:, :4], atol=1e-12)


def test_module_token_by_token_decode_matches_full_forward():
    layer = _layer()
    x = torch.randn(1, 6, 16, dtype=torch.float64)
    y_full, S_full = layer.forward_with_state(x)
    state, ys = None, []
    for t in range(6):
        y_t, state = layer.forward_with_state(x[:, t : t + 1], state)
        ys.append(y_t)
    assert state.shape == (1, 4, 4, 4)  # (B, H, d, d): does not grow with T
    assert torch.allclose(torch.cat(ys, dim=1), y_full, atol=1e-10)
    assert torch.allclose(state, S_full, atol=1e-10)


def test_hybrid_pattern_and_memory():
    assert hybrid_layer_pattern(8, 3) == ["linear", "linear", "linear", "full"] * 2
    # State bytes do not depend on context length; KV bytes grow linearly.
    assert linear_state_bytes(36, 32, 128, 128) == 36 * 32 * 128 * 128 * 4
    assert kv_cache_bytes(12, 2, 256, 2000) == 2 * kv_cache_bytes(12, 2, 256, 1000)
    # Qwen3-Next-shaped stack at 256K context: the hybrid needs ~1/4 of the all-full KV cache.
    pattern = hybrid_layer_pattern(48, 3)
    T = 262_144
    hybrid = hybrid_decode_cache_bytes(pattern, T, 2, 256, 32, 128, 128)
    all_full = kv_cache_bytes(48, 2, 256, T)
    assert 0.25 < hybrid / all_full < 0.26
