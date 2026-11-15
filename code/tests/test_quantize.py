"""Tests for lesson 08.7: post-training weight quantization (llmre.model.quantize).

Foundational references: the unquantized ``GPT`` of Module 6 (a quantized copy must stay
close to it and leave it untouched), and round-to-nearest, which GPTQ must reduce to when the
Hessian is diagonal and AWQ must reduce to at ``alpha = 0``.
"""

from __future__ import annotations

import pytest
import torch

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.model.quantize import (
    E2M1_VALUES,
    awq_quantize,
    awq_scales,
    dequantize,
    fake_quantize,
    gptq_quantize,
    layer_output_error,
    mxfp4_quantize,
    quantize_asymmetric,
    quantize_gpt_weights,
    quantize_groupwise,
    quantize_symmetric,
    quantized_weight_bytes,
    rtn_quantize,
)


def _correlated_inputs(n=256, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(d, d, generator=g) / d**0.5 + torch.eye(d)
    x = torch.randn(n, d, generator=g) @ mix
    x[:, 0] *= 10.0
    return x.double()


def test_lesson_hand_example_symmetric():
    w = torch.tensor([0.8, -0.3, 0.05, -1.2])
    q, s = quantize_symmetric(w, 4)
    assert torch.equal(q, torch.tensor([5.0, -2.0, 0.0, -7.0]))
    assert s.item() == pytest.approx(1.2 / 7)
    assert torch.allclose(dequantize(q, s), torch.tensor([6 / 7 * 1.0, -2.4 / 7, 0.0, -1.2]), atol=1e-6)


def test_lesson_hand_example_asymmetric():
    w = torch.tensor([0.8, -0.3, 0.05, -1.2])
    q, s, z = quantize_asymmetric(w, 4)
    assert torch.equal(q, torch.tensor([15.0, 7.0, 9.0, 0.0]))
    assert z.item() == 9.0
    assert torch.allclose(dequantize(q, s, z), torch.tensor([0.8, -0.8 / 3, 0.0, -1.2]), atol=1e-6)


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_rounding_error_at_most_half_a_step(bits):
    w = torch.randn(16, 64, dtype=torch.float64)
    q, s = quantize_symmetric(w, bits, dim=1)
    assert ((dequantize(q, s) - w).abs() <= s / 2 + 1e-12).all()
    q, s, z = quantize_asymmetric(w, bits, dim=1)
    assert ((dequantize(q, s, z) - w).abs() <= s / 2 + 1e-12).all()
    assert q.min() >= 0 and q.max() <= 2**bits - 1


def test_zero_is_exact_and_fake_quant_is_idempotent():
    w = torch.tensor([[0.0, 0.3, 0.9, 1.7]])
    for sym in (True, False):
        once = fake_quantize(w, 4, dim=1, symmetric=sym)
        assert once[0, 0].item() == 0.0
        assert torch.allclose(fake_quantize(once, 4, dim=1, symmetric=sym), once, atol=1e-6)


def test_groupwise_with_one_group_is_per_channel():
    w = torch.randn(8, 32)
    assert torch.allclose(quantize_groupwise(w, 4, 32), rtn_quantize(w, 4), atol=0)


def test_outlier_hurts_per_tensor_more_than_per_channel():
    torch.manual_seed(0)
    w = torch.randn(16, 64) * 0.02
    w[3, :] *= 100.0  # one row with large weights
    err_tensor = (fake_quantize(w, 4, dim=None) - w)[torch.arange(16) != 3].abs().mean()
    err_channel = (rtn_quantize(w, 4) - w)[torch.arange(16) != 3].abs().mean()
    assert err_channel < err_tensor / 5


def test_layer_output_error_equals_hessian_quadratic_form():
    x = _correlated_inputs()
    w = torch.randn(4, 32, dtype=torch.float64)
    wq = rtn_quantize(w, 3)
    d = wq - w
    H = x.T @ x
    assert layer_output_error(w, wq, x) == pytest.approx(float(torch.einsum("ri,ij,rj->", d, H, d)), rel=1e-9)


def test_gptq_lesson_hand_example():
    w = torch.tensor([[0.8, 0.3]], dtype=torch.float64)
    H = torch.tensor([[1.0, 0.9], [0.9, 1.0]], dtype=torch.float64)
    q = gptq_quantize(w, hessian=H, damp=0.0, scale=torch.tensor([0.5]))
    assert torch.allclose(q, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    d = q - w
    assert float(d @ H @ d.T) == pytest.approx(0.022)
    r = torch.tensor([[1.0, 0.5]], dtype=torch.float64) - w  # RTN
    assert float(r @ H @ r.T) == pytest.approx(0.152)


@pytest.mark.parametrize("group_size", [None, 8])
def test_gptq_with_diagonal_hessian_is_rtn(group_size):
    w = torch.randn(6, 16, dtype=torch.float64)
    H = torch.diag(torch.rand(16, dtype=torch.float64) + 0.5)
    q = gptq_quantize(w, hessian=H, bits=4, group_size=group_size, damp=0.0)
    assert torch.allclose(q, rtn_quantize(w, 4, group_size), atol=1e-12)


@pytest.mark.parametrize("bits,group_size", [(4, None), (3, None), (4, 8)])
def test_gptq_beats_rtn_on_correlated_inputs(bits, group_size):
    torch.manual_seed(0)
    x = _correlated_inputs()
    w = torch.randn(16, 32, dtype=torch.float64)
    e_rtn = layer_output_error(w, rtn_quantize(w, bits, group_size), x)
    e_gptq = layer_output_error(w, gptq_quantize(w, x, bits, group_size), x)
    assert e_gptq < 0.8 * e_rtn


def test_gptq_output_is_on_the_grid():
    torch.manual_seed(1)
    x = _correlated_inputs()
    w = torch.randn(4, 32, dtype=torch.float64)
    q = gptq_quantize(w, x, bits=3)
    # every row takes at most 2**3 distinct values (one scale per row)
    for row in q:
        assert torch.unique(row).numel() <= 8


def test_awq_alpha_zero_is_rtn_and_search_never_worse():
    torch.manual_seed(0)
    x = _correlated_inputs()
    w = torch.randn(16, 32, dtype=torch.float64)
    assert torch.allclose(awq_scales(x, 0.0), torch.ones(32, dtype=torch.float64))
    w_eff, alpha = awq_quantize(w, x, bits=4, group_size=8)
    assert 0.0 <= alpha < 1.0
    assert layer_output_error(w, w_eff, x) <= layer_output_error(w, rtn_quantize(w, 4, 8), x) + 1e-9


def test_mxfp4_lesson_example_and_grid():
    deq, scale = mxfp4_quantize(torch.tensor([[0.1, -0.7, 2.9, 0.26]]), block=4)
    assert scale.item() == 0.5
    assert torch.allclose(deq, torch.tensor([[0.0, -0.75, 3.0, 0.25]]))
    x = torch.randn(3, 64) * 5
    deq, scale = mxfp4_quantize(x)
    assert scale.shape == (3, 2)
    elems = (deq.double().reshape(3, 2, 32) / scale.unsqueeze(-1)).abs()
    assert torch.isin(elems, E2M1_VALUES.double()).all()
    assert torch.equal(torch.log2(scale), torch.round(torch.log2(scale)))  # powers of two


def test_quantized_weight_bytes():
    assert quantized_weight_bytes(4096, 4096, 4, 128) * 8 / (4096 * 4096) == pytest.approx(4.125)
    assert quantized_weight_bytes(4096, 4096, 4, 128, asymmetric=True) * 8 / (4096 * 4096) == pytest.approx(4.25)
    assert quantized_weight_bytes(10, 20, 8) == 10 * 20 + 10 * 2


@pytest.mark.parametrize("method", ["rtn", "gptq", "awq"])
def test_quantize_gpt_weights_close_and_original_untouched(method):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    idx = torch.randint(0, 64, (4, 16))
    q8 = quantize_gpt_weights(model, idx, method, bits=8)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
    assert torch.equal(q8.transformer.wte.weight, model.transformer.wte.weight)
    assert q8.lm_head.weight is q8.transformer.wte.weight  # tying survives the copy
    with torch.no_grad():
        ref = model(idx)[0]
        out8 = q8(idx)[0]
        out3 = quantize_gpt_weights(model, idx, method, bits=3)(idx)[0]
    assert (out8 - ref).abs().max() < 0.05
    assert (out3 - ref).abs().max() > (out8 - ref).abs().max()


def test_bad_arguments():
    with pytest.raises(ValueError):
        quantize_symmetric(torch.ones(3), 1)
    with pytest.raises(ValueError):
        quantize_groupwise(torch.ones(2, 10), 4, 3)
    with pytest.raises(ValueError):
        gptq_quantize(torch.ones(2, 4))
    with pytest.raises(ValueError):
        quantize_gpt_weights(GPT(GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=1, n_embd=8)),
                             torch.zeros(1, 4, dtype=torch.long), "nf4")
