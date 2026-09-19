"""Tests for Module 12 modern-architecture pieces: RoPE, RMSNorm, SwiGLU, GQA, MoE.

These back the claims of ``lessons/module-12/lesson-01.md`` .. ``lesson-04.md``:

* RoPE is a rotation (preserves norm) and makes the q·k score depend only on the
  relative offset ``m - n``.
* RMSNorm matches a manual root-mean-square computation.
* SwiGLU keeps the ``(B, T, C)`` shape.
* GQA reduces exactly to standard MHA when ``n_kv_head == n_head`` (shared weights).
* MoE returns the right shape, routes each token to exactly ``top_k`` experts, and
  its auxiliary load-balance loss is a finite scalar.
"""

from __future__ import annotations

import torch

from llmre.attention.attention import CausalSelfAttention
from llmre.attention.gqa import GroupedQueryAttention
from llmre.attention.rope import apply_rope, precompute_freqs_cis
from llmre.model.block import LayerNorm
from llmre.model.config import GPTConfig
from llmre.model.moe import MoE
from llmre.model.rmsnorm import RMSNorm
from llmre.model.swiglu import SwiGLU


def _cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=100, block_size=32, n_layer=1, n_head=4, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


# --------------------------------------------------------------------------- RoPE


def test_rope_preserves_norm():
    torch.manual_seed(0)
    B, nh, T, hd = 2, 3, 8, 16
    q = torch.randn(B, nh, T, hd, dtype=torch.float64)
    k = torch.randn(B, nh, T, hd, dtype=torch.float64)
    freqs = precompute_freqs_cis(hd, T).to(torch.complex128)
    q_rot, k_rot = apply_rope(q, k, freqs)
    # Rotation is length-preserving: each vector's L2 norm is unchanged.
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-10)
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-10)


def test_rope_dot_product_depends_only_on_relative_offset():
    """<RoPE(q, m), RoPE(k, n)> must equal <RoPE(q, m+d), RoPE(k, n+d)>."""
    torch.manual_seed(1)
    hd = 8
    T_max = 64
    freqs = precompute_freqs_cis(hd, T_max).to(torch.complex128)

    # A single query vector and key vector, shaped (B=1, nh=1, T=1, hd).
    qv = torch.randn(1, 1, 1, hd, dtype=torch.float64)
    kv = torch.randn(1, 1, 1, hd, dtype=torch.float64)

    def score(m: int, n: int) -> float:
        q_rot, _ = apply_rope(qv, qv, freqs[m : m + 1])
        k_rot, _ = apply_rope(kv, kv, freqs[n : n + 1])
        return float((q_rot * k_rot).sum())

    m, n, d = 5, 2, 7  # offset m - n = 3
    base = score(m, n)
    shifted = score(m + d, n + d)  # same offset 3
    # The freqs table is built in float32, so equality holds to ~float32 precision.
    assert abs(base - shifted) < 1e-6
    # Sanity: a DIFFERENT offset gives a different score (RoPE is doing something).
    assert abs(base - score(m + 1, n)) > 1e-4


# ------------------------------------------------------------------------ RMSNorm


def test_rmsnorm_matches_manual():
    torch.manual_seed(2)
    C = 16
    norm = RMSNorm(C, eps=1e-5).double()
    # Randomize the gain so we exercise the weight multiply too.
    with torch.no_grad():
        norm.weight.copy_(torch.randn(C, dtype=torch.float64))
    x = torch.randn(4, 7, C, dtype=torch.float64)

    y = norm(x)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    manual = x / rms * norm.weight
    assert torch.allclose(y, manual, atol=1e-6)


def test_rmsnorm_differs_from_layernorm_on_nonzero_mean():
    # With a nonzero-mean input, RMSNorm (no centering) must differ from LayerNorm.
    torch.manual_seed(3)
    C = 16
    x = torch.randn(2, 5, C) + 3.0  # shift the mean well away from 0
    rms = RMSNorm(C)
    ln = LayerNorm(C, bias=True)
    assert not torch.allclose(rms(x), ln(x), atol=1e-3)


# ------------------------------------------------------------------------- SwiGLU


def test_swiglu_output_shape():
    torch.manual_seed(4)
    B, T, C = 2, 6, 32
    ff = SwiGLU(C)
    x = torch.randn(B, T, C)
    y = ff(x)
    assert y.shape == (B, T, C)
    assert y.dtype == x.dtype


# ---------------------------------------------------------------------------- GQA


def test_gqa_output_shape():
    torch.manual_seed(5)
    cfg = _cfg(n_head=8, n_embd=64)
    for n_kv in (8, 4, 2, 1):  # MHA, GQA (two group sizes), MQA
        attn = GroupedQueryAttention(cfg, n_kv_head=n_kv)
        B, T, C = 2, 7, cfg.n_embd
        y = attn(torch.randn(B, T, C))
        assert y.shape == (B, T, C)


def test_gqa_equals_mha_when_kv_heads_full():
    """With n_kv_head == n_head and shared weights, GQA == CausalSelfAttention."""
    torch.manual_seed(6)
    cfg = _cfg(n_head=4, n_embd=32, bias=True)
    mha = CausalSelfAttention(cfg).eval().double()
    gqa = GroupedQueryAttention(cfg, n_kv_head=cfg.n_head).eval().double()

    C = cfg.n_embd
    with torch.no_grad():
        # CausalSelfAttention fuses Q,K,V into c_attn (3C, C); split it into GQA's
        # separate q/k/v projections so both modules compute identical Q,K,V.
        w = mha.c_attn.weight       # (3C, C)
        b = mha.c_attn.bias         # (3C,)
        gqa.q_proj.weight.copy_(w[0:C])
        gqa.k_proj.weight.copy_(w[C : 2 * C])
        gqa.v_proj.weight.copy_(w[2 * C : 3 * C])
        gqa.q_proj.bias.copy_(b[0:C])
        gqa.k_proj.bias.copy_(b[C : 2 * C])
        gqa.v_proj.bias.copy_(b[2 * C : 3 * C])
        gqa.c_proj.weight.copy_(mha.c_proj.weight)
        gqa.c_proj.bias.copy_(mha.c_proj.bias)

    x = torch.randn(2, 6, C, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(mha(x), gqa(x), atol=1e-10)


# ---------------------------------------------------------------------------- MoE


def test_moe_output_shape_and_aux_scalar():
    torch.manual_seed(7)
    B, T, C = 2, 5, 32
    moe = MoE(dim=C, n_experts=8, top_k=2, n_shared=1)
    x = torch.randn(B, T, C)
    out, aux = moe(x)
    assert out.shape == (B, T, C)
    # aux must be a finite 0-dim scalar tensor.
    assert aux.dim() == 0
    assert torch.isfinite(aux)


def test_moe_routes_each_token_to_exactly_top_k():
    torch.manual_seed(8)
    B, T, C = 3, 4, 32
    top_k = 3
    moe = MoE(dim=C, n_experts=8, top_k=top_k)
    x = torch.randn(B, T, C).reshape(-1, C)
    topk_idx, topk_gate, _ = moe.route(x)
    N = x.size(0)
    assert topk_idx.shape == (N, top_k)
    # Each token's chosen experts are distinct -> exactly top_k experts per token.
    for row in topk_idx:
        assert len(set(row.tolist())) == top_k
    # Gates form a distribution per token (sum to 1).
    assert torch.allclose(topk_gate.sum(dim=-1), torch.ones(N), atol=1e-6)
