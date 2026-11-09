"""Tests for the score-centering extension of lesson 16.4 (llmre.rl.score_centering)."""

import pytest
import torch

from llmre.rl.grpo import grpo_objective
from llmre.rl.mismatch import kl_regularized_optimum, train_kl_bandit
from llmre.rl.score_centering import (
    centered_bandit_gradient,
    drift_signal_decomposition,
    score_centered_objective,
    tis_weight,
    top_k_score_centered_objective,
    train_centered_bandit,
)


def _grad(fn, theta):
    th = theta.clone().requires_grad_(True)
    fn(th).backward()
    return th.grad


def _expected_update(theta, q, R, objective):
    """Sum over every token y of q(y) * grad(objective for y)."""
    total = torch.zeros_like(theta)
    for y in range(theta.shape[-1]):
        tok = torch.tensor([[y]])

        def f(th):
            return objective(torch.log_softmax(th, -1)[None, None], tok, torch.tensor([float(R[y])]))

        total += q[y] * _grad(f, theta)
    return total


def _full(q):
    return lambda lp, tok, adv: score_centered_objective(lp, q.log()[None, None], tok, adv)


def test_hand_example_decomposition():
    p, q = torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0.6, 0.2, 0.2])
    drift, signal = drift_signal_decomposition(p, q, torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(drift, torch.tensor([0.06, -0.06, 0.0]), atol=1e-6)
    assert torch.allclose(signal, torch.tensor([0.24, -0.12, -0.12]), atol=1e-6)


def test_constant_reward_expected_update_is_zero():
    theta = torch.tensor([0.4, -0.1, 0.2, 0.7], dtype=torch.float64)
    q = torch.softmax(theta + torch.tensor([1.0, -1.0, 0.5, 0.0], dtype=torch.float64), -1)
    g = _expected_update(theta, q, torch.ones(4, dtype=torch.float64), _full(q))
    assert torch.allclose(g, torch.zeros(4, dtype=torch.float64), atol=1e-12)


def test_expected_update_equals_signal():
    theta = torch.tensor([0.4, -0.1, 0.2], dtype=torch.float64)
    q = torch.softmax(theta + torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64), -1)
    R = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)
    g = _expected_update(theta, q, R, _full(q))
    _, signal = drift_signal_decomposition(torch.softmax(theta, -1), q, R)
    # softmax scores are defined up to a constant vector; signal sums to zero as well
    assert torch.allclose(g, signal, atol=1e-12)


def test_on_policy_gradient_matches_grpo_first_step():
    torch.manual_seed(0)
    theta = torch.randn(4, 3, 7, dtype=torch.float64, requires_grad=True)
    tokens = torch.randint(0, 7, (4, 3))
    adv = torch.tensor([1.0, -1.0, 0.5, 0.0], dtype=torch.float64)
    logp = torch.log_softmax(theta, -1)
    score_centered_objective(logp, logp.detach(), tokens, adv, reduction="sum").backward()
    g_centered = theta.grad.clone()
    theta.grad = None
    logp = torch.log_softmax(theta, -1)
    seq = logp.gather(-1, tokens[..., None]).squeeze(-1).sum(-1)
    grpo_objective(seq, seq.detach(), adv, reduction="none").sum().backward()
    assert torch.allclose(g_centered, theta.grad, atol=1e-10)


def test_group_centering_leaves_prefix_drift():
    """Second token cannot affect reward, so its true gradient is zero."""
    root_q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    p2 = torch.tensor([0.5, 0.5], dtype=torch.float64)
    q2 = torch.tensor([0.8, 0.2], dtype=torch.float64)
    adv_after_first_token_0 = torch.tensor([1.0, 1.0], dtype=torch.float64)  # group mean is 0 over both prefixes
    assert float((root_q * torch.tensor([1.0, -1.0], dtype=torch.float64)).sum()) == 0.0
    theta = p2.log()

    def plain(lp, tok, adv):
        return (adv[None] * lp.gather(-1, tok[..., None]).squeeze(-1)).mean()

    assert torch.allclose(_expected_update(theta, q2, adv_after_first_token_0, plain), q2 - p2, atol=1e-12)
    g = _expected_update(theta, q2, adv_after_first_token_0, _full(q2))
    assert torch.allclose(g, torch.zeros(2, dtype=torch.float64), atol=1e-12)


def test_full_is_makes_centering_vanish():
    theta = torch.tensor([[[0.3, 0.1, -0.4, 0.2]]], dtype=torch.float64)
    q = torch.log_softmax(theta + torch.tensor([0.5, -0.5, 0.2, 0.0], dtype=torch.float64), -1)
    tok, adv = torch.tensor([[2]]), torch.tensor([1.0], dtype=torch.float64)

    def centered(th):
        return score_centered_objective(torch.log_softmax(th, -1), q, tok, adv, weight_fn=lambda r: r)

    def is_only(th):
        lp = torch.log_softmax(th, -1)
        return ((lp[0, 0, 2].detach() - q[0, 0, 2]).exp() * lp[0, 0, 2])

    assert torch.allclose(_grad(centered, theta), _grad(is_only, theta), atol=1e-10)


def test_top_k_hand_example():
    p = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)
    q = torch.tensor([0.6, 0.25, 0.1, 0.05], dtype=torch.float64)
    theta = p.log()[None, None]
    tok = torch.tensor([[0]])
    adv = torch.tensor([1.0], dtype=torch.float64)

    def f(th):
        return top_k_score_centered_objective(
            torch.log_softmax(th, -1), torch.tensor([[[0, 1]]]), q[:2].log()[None, None], tok, q[0].log()[None, None], adv
        )

    s0 = torch.tensor([0.5, -0.3, -0.15, -0.05], dtype=torch.float64)
    s_bar_hat = torch.tensor([0.1, -0.05, -0.0375, -0.0125], dtype=torch.float64)
    assert torch.allclose(_grad(f, theta)[0, 0], s0 - s_bar_hat, atol=1e-9)


@pytest.mark.parametrize("weight_fn", [None, tis_weight(2.0)])
def test_full_head_matches_full_centering(weight_fn):
    torch.manual_seed(1)
    theta = torch.randn(2, 3, 6, dtype=torch.float64)
    q = torch.log_softmax(theta + 0.5 * torch.randn(2, 3, 6, dtype=torch.float64), -1)
    tok = torch.randint(0, 6, (2, 3))
    adv = torch.tensor([1.0, -1.0], dtype=torch.float64)
    ids = torch.arange(6).expand(2, 3, 6)
    sampled_q = q.gather(-1, tok[..., None]).squeeze(-1)

    def full(th):
        return score_centered_objective(torch.log_softmax(th, -1), q, tok, adv, weight_fn=weight_fn)

    def topk(th):
        return top_k_score_centered_objective(torch.log_softmax(th, -1), ids, q, tok, sampled_q, adv, weight_fn=weight_fn)

    assert torch.allclose(_grad(full, theta), _grad(topk, theta), atol=1e-8)


def test_mask_and_validation():
    lp = torch.log_softmax(torch.zeros(1, 2, 3), -1)
    tok = torch.zeros(1, 2, dtype=torch.long)
    out = score_centered_objective(
        lp, lp + 0.0, tok, torch.ones(1), mask=torch.tensor([[True, False]]), reduction="none"
    )
    assert out[0, 1] == 0.0
    with pytest.raises(ValueError):
        score_centered_objective(lp, lp, tok, torch.ones(2))
    with pytest.raises(ValueError):
        score_centered_objective(lp, lp, tok, torch.ones(1), reduction="max")
    with pytest.raises(ValueError):
        tis_weight(0.0)


def _bandit():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    ref = torch.log_softmax(torch.zeros(4, dtype=torch.float64), -1)
    return r, ref, 0.5, torch.tensor([-1.0, 1.0, 0.0, 0.0], dtype=torch.float64)


def _kl(lp, opt):
    return float((lp.exp() * (lp - opt)).sum())


def test_bandit_fixed_point():
    r, ref, beta, bias = _bandit()
    opt = kl_regularized_optimum(r, ref, beta)
    assert _kl(train_kl_bandit(r, ref, beta, bias, "none"), opt) > 0.05
    assert _kl(train_centered_bandit(r, ref, beta, bias), opt) < 1e-6
    assert _kl(train_centered_bandit(r, ref, beta, bias, cap=1.0), opt) < 1e-6


def test_centered_gradient_is_not_the_true_gradient():
    r, ref, beta, bias = _bandit()
    theta = torch.tensor([0.3, -0.2, 0.1, 0.0], dtype=torch.float64)
    q = torch.log_softmax(theta + bias, -1)
    centered = centered_bandit_gradient(theta, r, ref, beta, q)
    R = r - beta * (torch.log_softmax(theta, -1) - ref)
    assert torch.allclose(centered, q.exp() * (R - (q.exp() * R).sum()), atol=1e-12)
    on_policy = centered_bandit_gradient(theta, r, ref, beta, torch.log_softmax(theta, -1))
    assert not torch.allclose(centered, on_policy, atol=1e-3)
