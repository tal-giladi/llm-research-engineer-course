"""Tests for the Frontier updates: unbiased pass@k and the GVPO loss."""

import math

import pytest
import torch

from llmre.evaluation.pass_at_k import mean_pass_at_k, pass_at_k
from llmre.rl.gvpo import gvpo_loss


# --------------------------------------------------------------------------- #
# pass@k
# --------------------------------------------------------------------------- #
def test_pass_at_1_is_fraction_correct():
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)


def test_pass_at_k_matches_binomial_formula():
    n, c, k = 8, 2, 4
    expected = 1 - math.comb(n - c, k) / math.comb(n, k)
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_pass_at_k_edges():
    assert pass_at_k(8, 0, 4) == 0.0
    assert pass_at_k(8, 5, 4) == 1.0      # only 3 failures, any 4 contains a pass
    assert pass_at_k(4, 1, 4) == 1.0


def test_pass_at_k_monotone_in_k():
    vals = [pass_at_k(16, 3, k) for k in range(1, 17)]
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


def test_mean_pass_at_k():
    assert mean_pass_at_k([(4, 0), (4, 4)], k=4) == pytest.approx(0.5)


def test_pass_at_k_rejects_bad_args():
    with pytest.raises(ValueError):
        pass_at_k(4, 5, 1)
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 5)


# --------------------------------------------------------------------------- #
# GVPO
# --------------------------------------------------------------------------- #
def test_gvpo_zero_at_analytic_optimum():
    beta = 0.5
    ref_logits = torch.tensor([[0.3, -0.2, 1.0, 0.0]])
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    ref_lp = torch.log_softmax(ref_logits, dim=1)
    opt_lp = torch.log_softmax(ref_logits + rewards / beta, dim=1)   # pi_ref * exp(R/beta) / Z
    assert gvpo_loss(opt_lp, ref_lp, rewards, beta=beta).item() == pytest.approx(0.0, abs=1e-10)


def test_gvpo_positive_away_from_optimum():
    ref_lp = torch.log_softmax(torch.zeros(1, 4), dim=1)
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert gvpo_loss(ref_lp, ref_lp, rewards, beta=0.1).item() > 0


def test_gvpo_gradient_descent_converges_to_optimum():
    torch.manual_seed(0)
    beta = 0.5
    ref_logits = torch.randn(2, 5)
    rewards = torch.randint(0, 2, (2, 5)).float()
    ref_lp = torch.log_softmax(ref_logits, dim=1)
    logits = torch.zeros(2, 5, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=5.0)
    for _ in range(500):
        opt.zero_grad()
        gvpo_loss(torch.log_softmax(logits, dim=1), ref_lp, rewards, beta=beta).backward()
        opt.step()
    target = torch.softmax(ref_logits + rewards / beta, dim=1)
    assert torch.allclose(torch.softmax(logits, dim=1), target, atol=1e-3)


def test_gvpo_invariant_to_per_prompt_reward_shift():
    lp = torch.log_softmax(torch.randn(3, 4), dim=1)
    ref = torch.log_softmax(torch.randn(3, 4), dim=1)
    r = torch.randn(3, 4)
    shift = torch.tensor([[5.0], [-2.0], [100.0]])
    assert torch.allclose(gvpo_loss(lp, ref, r), gvpo_loss(lp, ref, r + shift))
