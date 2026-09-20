"""Tests for Module 16 (RL for reasoning): GRPO advantages/objective, verifiers,
and the toy RLVR loop.

These exercise the *existing* implementations in
:mod:`llmre.rl.grpo`, :mod:`llmre.reasoning.verifiers`, and
:mod:`llmre.reasoning.rlvr`.
"""

import math

import pytest
import torch

from llmre.reasoning.rlvr import run_rlvr
from llmre.reasoning.verifiers import arithmetic_verifier, exact_match
from llmre.rl.grpo import grpo_advantages, grpo_objective


# --------------------------------------------------------------------------- #
# GRPO advantages: group-normalized (row mean ~0, population std 1)
# --------------------------------------------------------------------------- #
def test_grpo_advantages_row_mean_is_zero():
    rewards = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0, 1.0],
            [3.0, 3.0, 0.0, 0.0],
        ]
    )
    adv = grpo_advantages(rewards)
    assert adv.shape == rewards.shape
    assert adv.dtype == rewards.dtype
    # every group's advantages are centered on 0 (the group mean is the baseline)
    row_means = adv.mean(dim=1)
    assert torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-5), row_means


def test_grpo_advantages_unit_population_std():
    rewards = torch.tensor([[2.0, 0.0, 1.0, 1.0], [5.0, 1.0, 3.0, 3.0]])
    adv = grpo_advantages(rewards)
    # population std (unbiased=False) is ~1 within each group (eps makes it just under 1)
    row_std = adv.std(dim=1, unbiased=False)
    assert torch.allclose(row_std, torch.ones_like(row_std), atol=1e-4), row_std


def test_grpo_advantages_matches_hand_computation():
    # Group of 4 rewards, worked by hand:
    #   mean = (1+0+1+0)/4 = 0.5
    #   population std = sqrt(((0.5)^2 * 4)/4) = 0.5
    #   A_i = (r_i - 0.5) / 0.5  ->  [+1, -1, +1, -1]  (eps makes it ~0.99999998)
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    adv = grpo_advantages(rewards)
    expected = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    assert torch.allclose(adv, expected, atol=1e-6), adv

    # A second group with a non-trivial spread:
    #   [2,0,1,1] -> mean 1, pop std sqrt(0.5)=0.70710678
    #   A = [1/std, -1/std, 0, 0] = [+1.41421356, -1.41421356, 0, 0]
    rewards2 = torch.tensor([[2.0, 0.0, 1.0, 1.0]])
    adv2 = grpo_advantages(rewards2)
    s = math.sqrt(0.5)
    expected2 = torch.tensor([[1.0 / s, -1.0 / s, 0.0, 0.0]])
    assert torch.allclose(adv2, expected2, atol=1e-5), adv2


def test_grpo_advantages_unanimous_group_is_all_zero():
    # A group where every completion got the same reward carries no signal:
    # std = 0, so the eps-guarded division returns all zeros (no push either way).
    for val in (0.0, 1.0):
        adv = grpo_advantages(torch.full((1, 6), val))
        assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6), (val, adv)


def test_grpo_advantages_rejects_non_2d():
    with pytest.raises(ValueError):
        grpo_advantages(torch.tensor([1.0, 0.0, 1.0]))


# --------------------------------------------------------------------------- #
# GRPO objective: on-policy reduces to A * logprob; clipping caps the ratio
# --------------------------------------------------------------------------- #
def test_grpo_objective_on_policy_is_group_baselined_pg():
    # On-policy the ratio is exactly 1, so the objective VALUE equals A itself...
    logp = torch.tensor([-0.5, -1.2, -0.1], requires_grad=True)
    adv = torch.tensor([1.0, -1.0, 0.5])
    obj = grpo_objective(logp, logp.detach(), adv, reduction="none")
    assert torch.allclose(obj.detach(), adv, atol=1e-6), obj
    # ...and its GRADIENT w.r.t. the logprobs is A (the group-baselined policy
    # gradient): d/d logp [ratio * A] = A * ratio = A when ratio == 1.
    obj.sum().backward()
    assert torch.allclose(logp.grad, adv, atol=1e-6), logp.grad


def test_grpo_objective_clips_positive_advantage():
    # ratio pushed to 1.5 with clip_eps 0.2 -> clipped ratio 1.2; for A>0 the min
    # of unclipped (1.5*A) and clipped (1.2*A) is the clipped one.
    old = torch.tensor([0.0])
    new = torch.tensor([math.log(1.5)])
    adv = torch.tensor([2.0])
    obj = grpo_objective(new, old, adv, clip_eps=0.2, reduction="none")
    assert torch.allclose(obj, torch.tensor([1.2 * 2.0]), atol=1e-5), obj


# --------------------------------------------------------------------------- #
# Verifiers: exact-match and the arithmetic verifier grade correctly
# --------------------------------------------------------------------------- #
def test_exact_match_numeric_and_string():
    assert exact_match("3", "3.0") == 1.0      # numeric equality
    assert exact_match("+7", "7") == 1.0       # leading plus
    assert exact_match(" 42 ", "42") == 1.0    # whitespace
    assert exact_match("Cat", "cat") == 1.0    # case-insensitive string path
    assert exact_match("8", "9") == 0.0
    assert exact_match("dog", "cat") == 0.0


def test_arithmetic_verifier_grades_correctly():
    assert arithmetic_verifier("3 + 4", 7) == 1.0
    assert arithmetic_verifier("3 + 4", 8) == 0.0
    assert arithmetic_verifier("12*11", 132) == 1.0
    assert arithmetic_verifier("5 - 2 =", 3) == 1.0
    assert arithmetic_verifier("6 x 7", 42) == 1.0
    assert arithmetic_verifier("6 x 7", "42") == 1.0   # string answer also works


def test_arithmetic_verifier_rejects_unparseable_problem():
    with pytest.raises(ValueError):
        arithmetic_verifier("what is love", 0)


# --------------------------------------------------------------------------- #
# RLVR loop: training raises the toy correct-rate by a clear margin
# --------------------------------------------------------------------------- #
def test_rlvr_loop_increases_correct_rate():
    res = run_rlvr(seed=0)
    # starts near chance (~1/10), ends near-certain on these easy sums
    assert res.initial_correct_rate < 0.4
    assert res.final_correct_rate > 0.9
    # and the improvement is a clear margin, not noise
    assert res.final_correct_rate - res.initial_correct_rate > 0.5
    assert len(res.history) == 150


def test_rlvr_loop_is_reproducible_and_robust_across_seeds():
    a = run_rlvr(seed=0)
    b = run_rlvr(seed=0)
    assert a.history == b.history            # same seed -> identical run
    # improvement holds for a different seed too
    c = run_rlvr(seed=1)
    assert c.final_correct_rate - c.initial_correct_rate > 0.5
