"""Tests for llmre.optim.muon (lesson 03.4) — Muon from scratch vs references.

Checks, in order: the cubic Newton-Schulz iteration converges to the exact
orthogonal factor U V^T; the tuned quintic lands every singular value in a
band around 1 after five steps; our Muon matches ``torch.optim.Muon`` step for
step; with an exact orthogonalizer a 1-row matrix reduces to normalized
momentum-SGD; the parameter split and the hybrid optimizer train the course GPT.
"""

from __future__ import annotations

import pytest
import torch

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.optim.muon import (
    CUBIC_COEFFS,
    Muon,
    MuonWithAuxAdamW,
    lr_scale,
    newton_schulz,
    newton_schulz_flops,
    orthogonalize_svd,
    split_params_for_muon,
)


def test_cubic_newton_schulz_converges_to_svd_polar_factor():
    torch.manual_seed(0)
    G = torch.randn(6, 9, dtype=torch.float64)
    O = newton_schulz(G, steps=40, coeffs=CUBIC_COEFFS, dtype=None)
    assert torch.allclose(O, orthogonalize_svd(G), atol=1e-10)
    # rows of a wide orthogonal factor are orthonormal: O O^T = I
    assert torch.allclose(O @ O.T, torch.eye(6, dtype=torch.float64), atol=1e-10)


def test_quintic_singular_values_land_near_one_and_keep_singular_vectors():
    torch.manual_seed(0)
    for shape in [(32, 64), (64, 32)]:  # wide and tall go through the same code
        G = torch.randn(*shape, dtype=torch.float64)
        O = newton_schulz(G, steps=5, dtype=None)
        s = torch.linalg.svdvals(O)
        assert s.min() > 0.6 and s.max() < 1.25
        # Same singular vectors: O = U diag(s') V^T, so U^T O V is diagonal.
        U, _, Vh = torch.linalg.svd(G, full_matrices=False)
        D = U.T @ O @ Vh.T
        off = D - torch.diag(torch.diagonal(D))
        assert off.abs().max() < 1e-8


def test_diagonal_worked_example_from_lesson():
    # Lesson 03.4 section 3.2: G = diag(3, 1), one cubic step from X0 = G / ||G||_F.
    G = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    X1 = newton_schulz(G, steps=1, coeffs=CUBIC_COEFFS, dtype=None)
    assert torch.allclose(torch.diagonal(X1), torch.tensor([0.99605, 0.45850], dtype=torch.float64), atol=1e-4)
    # Rank-1 example (section 3.4): the orthogonalized [[1, 2], [0, 0]] is u v^T
    # = R / ||R||. Its one singular value is already 1 after normalization and
    # x = 1 is a fixed point of the cubic map; the zero direction stays zero.
    R = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64)
    expected = R / 5 ** 0.5
    assert torch.allclose(newton_schulz(R, steps=3, coeffs=CUBIC_COEFFS, dtype=None), expected, atol=1e-12)


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon needs torch >= 2.9")
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("adjust", ["original", "match_rms_adamw"])
def test_muon_matches_torch_over_5_steps(nesterov, adjust):
    torch.manual_seed(1)
    W0 = torch.randn(6, 10)
    X, Y = torch.randn(32, 10), torch.randn(32, 6)
    w_ours = W0.clone().requires_grad_(True)
    w_ref = W0.clone().requires_grad_(True)
    ours = Muon([w_ours], lr=0.02, momentum=0.95, nesterov=nesterov, weight_decay=0.1, adjust_lr=adjust)
    ref = torch.optim.Muon([w_ref], lr=0.02, momentum=0.95, nesterov=nesterov, weight_decay=0.1,
                           adjust_lr_fn=adjust)
    for _ in range(5):
        for w, opt in ((w_ours, ours), (w_ref, ref)):
            opt.zero_grad()
            ((X @ w.T - Y) ** 2).mean().backward()
            opt.step()
        assert torch.allclose(w_ours, w_ref, atol=1e-6, rtol=0)


def test_one_row_matrix_reduces_to_normalized_momentum_sgd():
    # For a (1, n) matrix the polar factor of m is m / ||m||, so exact-orthogonalized
    # Muon is momentum-SGD (03.1, EMA form) with the step normalized to length lr.
    torch.manual_seed(2)
    w0 = torch.randn(1, 5, dtype=torch.float64)
    x, y = torch.randn(20, 5, dtype=torch.float64), torch.randn(20, 1, dtype=torch.float64)
    w = w0.clone().requires_grad_(True)
    opt = Muon([w], lr=0.1, momentum=0.9, nesterov=False, orthogonalizer="svd", adjust_lr=None)
    w_ref, buf = w0.clone(), torch.zeros_like(w0)
    for _ in range(4):
        opt.zero_grad()
        ((x @ w.T - y) ** 2).mean().backward()
        opt.step()
        wr = w_ref.clone().requires_grad_(True)
        ((x @ wr.T - y) ** 2).mean().backward()
        buf = 0.9 * buf + 0.1 * wr.grad
        w_ref = w_ref - 0.1 * buf / buf.norm()
        assert torch.allclose(w.detach(), w_ref, atol=1e-12)


def test_lr_scale_and_flops():
    assert lr_scale((4096, 1024), "original") == 2.0
    assert lr_scale((1024, 4096), "original") == 1.0
    assert abs(lr_scale((1024, 4096), "match_rms_adamw") - 0.2 * 64) < 1e-12
    # An orthogonal (A, B) matrix has RMS 1/sqrt(max(A, B)); match_rms_adamw makes it 0.2.
    O = orthogonalize_svd(torch.randn(64, 256, dtype=torch.float64))
    rms = O.pow(2).mean().sqrt().item() * lr_scale((64, 256), "match_rms_adamw")
    assert abs(rms - 0.2) < 1e-9
    assert newton_schulz_flops(4096, 4096, steps=5) == 5 * 6 * 4096 ** 3


def test_rejects_non_matrix_params():
    with pytest.raises(ValueError):
        Muon([torch.zeros(3, requires_grad=True)])


def _tiny_gpt():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0, bias=True))


def test_split_params_covers_every_parameter_once():
    model = _tiny_gpt()
    muon, adamw = split_params_for_muon(model)
    ids = [id(p) for p in muon + adamw]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert all(p.ndim == 2 for p in muon)
    assert id(model.lm_head.weight) in {id(p) for p in adamw}  # tied with wte: AdamW, once
    assert id(model.transformer.wpe.weight) in {id(p) for p in adamw}
    assert len(muon) == 4 * 2  # c_attn, attn c_proj, c_fc, mlp c_proj per block


def test_hybrid_trains_tiny_gpt():
    model = _tiny_gpt()
    opt = MuonWithAuxAdamW(model, muon_kwargs={"lr": 0.02}, adamw_kwargs={"lr": 3e-3})
    torch.manual_seed(3)
    x = torch.randint(0, 32, (4, 8))
    y = torch.roll(x, -1, dims=1)
    first = None
    for _ in range(60):
        _, loss = model(x, y)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first
