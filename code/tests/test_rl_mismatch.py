"""Tests for lesson 16.4: training-inference mismatch (llmre.rl.mismatch)."""

import math

import pytest
import torch

from llmre.rl.grpo import grpo_objective
from llmre.rl.mismatch import (
    decoupled_grpo_objective,
    engine_num_splits,
    kl_regularized_optimum,
    kl_regularized_value,
    mismatch_metrics,
    ratio_rejection_mask,
    softmax_bandit_gradient,
    split_k_matvec,
    toy_logprobs,
    train_kl_bandit,
    truncated_is_weights,
)


def _data(seed=0, B=64, d=64, V=50):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d, V, generator=g) * 0.5
    x = torch.randn(B, d, generator=g)
    tok = torch.randint(0, V, (B,), generator=g)
    return x, W, tok


def test_bf16_addition_is_not_associative():
    one = torch.tensor(1.0, dtype=torch.bfloat16)
    s = torch.tensor(2.0 ** -8, dtype=torch.bfloat16)
    assert ((one + s) + s).item() == 1.0
    assert (one + (s + s)).item() == 1.0078125


def test_split_k_matches_reference():
    x, W, _ = _data()
    ref = x.double() @ W.double()
    for splits in (1, 2, 4, 64):
        out = split_k_matvec(x, W, splits)
        assert out.shape == (64, 50) and out.dtype == torch.float32
        assert torch.allclose(out.double(), ref, atol=1e-4)


def test_split_changes_bits():
    x, W, _ = _data()
    a32, b32 = split_k_matvec(x, W, 1), split_k_matvec(x, W, 4)
    assert not torch.equal(a32, b32)
    assert (a32 - b32).abs().max() < 1e-4
    a16 = split_k_matvec(x, W, 1, acc_dtype=torch.bfloat16)
    b16 = split_k_matvec(x, W, 4, acc_dtype=torch.bfloat16)
    assert (a16 - b16).abs().max() > 1e-2  # bf16 accumulation: much larger


def test_split_k_validates():
    x, W, _ = _data()
    with pytest.raises(ValueError):
        split_k_matvec(x, W, 0)
    with pytest.raises(ValueError):
        split_k_matvec(x, W.T, 1)


def test_engine_heuristic_depends_on_batch():
    assert engine_num_splits(1) != engine_num_splits(64)


def test_batch_variance_and_invariance():
    x, W, tok = _data()
    for acc in (torch.float32, torch.bfloat16):
        alone = toy_logprobs(x[:4], W, tok[:4], acc_dtype=acc)
        in_batch = toy_logprobs(x, W, tok, acc_dtype=acc)[:4]
        assert not torch.equal(alone, in_batch)
        alone_inv = toy_logprobs(x[:4], W, tok[:4], batch_invariant=True, acc_dtype=acc)
        in_batch_inv = toy_logprobs(x, W, tok, batch_invariant=True, acc_dtype=acc)[:4]
        assert torch.equal(alone_inv, in_batch_inv)


def test_mismatch_metrics():
    lp = torch.log(torch.tensor([[0.5, 0.25, 0.1]]))
    zero = mismatch_metrics(lp, lp)
    assert zero == {"mean_abs_diff": 0.0, "max_abs_diff": 0.0, "k3_kl": 0.0, "max_seq_log_ratio": 0.0}
    mu = torch.log(torch.tensor([[0.4, 0.25, 0.2]]))
    m = mismatch_metrics(lp, mu)
    d = [math.log(0.5 / 0.4), 0.0, math.log(0.1 / 0.2)]
    assert m["max_abs_diff"] == pytest.approx(abs(d[2]), rel=1e-5)
    assert m["mean_abs_diff"] == pytest.approx(sum(abs(v) for v in d) / 3, rel=1e-5)
    k3 = sum(math.exp(v) - v - 1 for v in d) / 3
    assert m["k3_kl"] == pytest.approx(k3, rel=1e-4) and m["k3_kl"] > 0
    masked = mismatch_metrics(lp, mu, mask=torch.tensor([[True, True, False]]))
    assert masked["max_abs_diff"] == pytest.approx(abs(d[0]), rel=1e-5)


def test_token_tis_by_hand():
    train = torch.log(torch.tensor([[0.6, 0.02, 0.3]]))
    roll = torch.log(torch.tensor([[0.5, 0.001, 0.3]]))
    w = truncated_is_weights(train, roll, cap=2.0)
    assert torch.allclose(w, torch.tensor([[1.2, 2.0, 1.0]]), atol=1e-5)
    w_mask = truncated_is_weights(train, roll, cap=2.0, mask=torch.tensor([[True, False, True]]))
    assert torch.allclose(w_mask, torch.tensor([[1.2, 0.0, 1.0]]), atol=1e-5)


def test_sequence_tis_by_hand():
    train = torch.log(torch.tensor([[0.6, 0.5], [0.5, 0.5]]))
    roll = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.1]]))
    w = truncated_is_weights(train, roll, cap=2.0, level="sequence")
    assert torch.allclose(w, torch.tensor([[1.2, 1.2], [2.0, 2.0]]), atol=1e-5)
    with pytest.raises(ValueError):
        truncated_is_weights(train, roll, level="batch")


def test_weights_are_detached():
    train = torch.zeros(1, 3, requires_grad=True)
    w = truncated_is_weights(train, torch.zeros(1, 3))
    assert not w.requires_grad


def test_rejection_mask():
    train = torch.log(torch.tensor([[0.5, 0.5], [0.9, 0.9], [0.5, 0.5]]))
    roll = torch.log(torch.tensor([[0.5, 0.5], [0.3, 0.3], [0.5, 0.4]]))
    keep = ratio_rejection_mask(train, roll, low=0.5, high=2.0)
    assert keep.tolist() == [True, False, True]  # ratios 1, 9, 1.25
    keep_geo = ratio_rejection_mask(train, roll, low=0.9, high=1.1, geometric=True)
    assert keep_geo.tolist() == [True, False, False]  # geometric 1, 3, ~1.118


def test_decoupled_reduces_to_grpo():
    torch.manual_seed(0)
    lp, old, adv = torch.randn(8), torch.randn(8), torch.randn(8)
    a = decoupled_grpo_objective(lp, old, old.clone(), adv)
    b = grpo_objective(lp, old, adv)
    assert torch.allclose(a, b)


def test_decoupled_weight_and_gradient():
    lp = torch.tensor([0.0, 0.0], requires_grad=True)
    old = torch.zeros(2)
    roll = torch.log(torch.tensor([0.8, 0.1]))  # pi_old / mu = 1.25, 10 -> capped 2
    adv = torch.tensor([1.0, 1.0])
    obj = decoupled_grpo_objective(lp, old, roll, adv, cap=2.0, reduction="none")
    assert torch.allclose(obj, torch.tensor([1.25, 2.0]), atol=1e-5)
    obj.sum().backward()
    assert torch.allclose(lp.grad, torch.tensor([1.25, 2.0]), atol=1e-5)


def _bandit():
    K = 4
    r = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    ref = torch.log_softmax(torch.zeros(K, dtype=torch.float64), -1)
    return r, ref, 0.5


def test_is_gradient_equals_true_gradient():
    r, ref, beta = _bandit()
    theta = torch.tensor([0.3, -0.2, 0.1, 0.0], dtype=torch.float64, requires_grad=True)
    kl_regularized_value(theta, r, ref, beta).backward()
    true = theta.grad
    th = theta.detach()
    mu = torch.log_softmax(th + torch.tensor([-1.0, 1.0, 0.0, 0.0], dtype=torch.float64), -1)
    g_is = softmax_bandit_gradient(th, r, ref, beta, mu, "is")
    g_none = softmax_bandit_gradient(th, r, ref, beta, mu, "none")
    g_tis_big = softmax_bandit_gradient(th, r, ref, beta, mu, "tis", cap=1e9)
    assert torch.allclose(g_is, true, atol=1e-12)
    assert torch.allclose(g_tis_big, true, atol=1e-12)
    assert not torch.allclose(g_none, true, atol=1e-3)
    # with no mismatch the uncorrected estimator is already exact
    g_same = softmax_bandit_gradient(th, r, ref, beta, torch.log_softmax(th, -1), "none")
    assert torch.allclose(g_same, true, atol=1e-12)


def test_bandit_fixed_points():
    r, ref, beta = _bandit()
    opt = kl_regularized_optimum(r, ref, beta)
    bias = torch.tensor([-1.0, 1.0, 0.0, 0.0])

    def kl(lp):
        return float((lp.exp() * (lp - opt)).sum())

    assert kl(train_kl_bandit(r, ref, beta, torch.zeros(4), "none")) < 1e-5
    assert kl(train_kl_bandit(r, ref, beta, bias, "is")) < 1e-5
    assert kl(train_kl_bandit(r, ref, beta, bias, "none")) > 0.05
    assert kl(train_kl_bandit(r, ref, beta, bias, "tis", cap=1.0)) > 0.01


def test_support_mismatch_is_not_fixable_by_is():
    r = torch.tensor([1.0, 0.0, 0.0, 0.5])
    ref = torch.log_softmax(torch.zeros(4), -1)
    beta = 0.5
    cut = torch.tensor([0.0, 0.0, 0.0, -float("inf")])
    lp = train_kl_bandit(r, ref, beta, cut, "is")
    assert torch.isfinite(lp).all()
    opt = kl_regularized_optimum(r, ref, beta)
    assert lp[3].exp() < 0.05 < opt[3].exp()
