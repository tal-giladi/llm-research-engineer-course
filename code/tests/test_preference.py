"""Tests for Module 15 (preference learning): reward model, DPO, PPO math."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmre.preference.dpo import dpo_loss, dpo_reward_margin, sequence_logprob
from llmre.preference.reward_model import RewardModel, bradley_terry_loss
from llmre.rl.ppo import compute_advantages, kl_penalty, ppo_clip_objective


# --------------------------------------------------------------------------- #
# Bradley-Terry reward loss
# --------------------------------------------------------------------------- #
def test_bradley_terry_matches_neg_log_sigmoid_exactly():
    rw = torch.tensor([2.0, 2.0, 3.0, 0.0])
    rl = torch.tensor([1.0, 0.0, -1.0, 0.0])
    got = bradley_terry_loss(rw, rl, reduction="none")
    want = -torch.log(torch.sigmoid(rw - rl))
    assert torch.allclose(got, want, atol=1e-6), (got, want)
    # mean reduction
    assert torch.allclose(bradley_terry_loss(rw, rl), want.mean(), atol=1e-6)


def test_bradley_terry_decreases_as_margin_grows():
    margins = [0.0, 0.5, 1.0, 2.0, 4.0]
    losses = [bradley_terry_loss(torch.tensor([m]), torch.tensor([0.0])).item() for m in margins]
    # strictly decreasing in the margin r_w - r_l
    for a, b in zip(losses, losses[1:]):
        assert b < a, losses
    # margin 0 => sigmoid 0.5 => loss = log 2
    assert abs(losses[0] - math.log(2.0)) < 1e-6


def test_reward_model_reads_last_token_and_trains_a_margin():
    torch.manual_seed(0)
    C = 8
    rm = RewardModel(n_embd=C, backbone=None)
    # Two "sequences" of hidden states; reward is read at the final position.
    hidden = torch.randn(2, 4, C)
    r = rm.reward_from_hidden(hidden)
    assert r.shape == (2,)
    # last-token selection matches taking score at T-1
    scores = rm.score(hidden).squeeze(-1)
    assert torch.allclose(r, scores[:, -1], atol=1e-6)

    # seq_lengths selects the right index
    r2 = rm.reward_from_hidden(hidden, seq_lengths=torch.tensor([2, 4]))
    assert torch.allclose(r2[0], scores[0, 1], atol=1e-6)
    assert torch.allclose(r2[1], scores[1, 3], atol=1e-6)

    # One BT training step lowers the loss (reward model learns the preference).
    opt = torch.optim.SGD(rm.parameters(), lr=0.5)
    hc, hr = torch.randn(3, 4, C), torch.randn(3, 4, C)
    before = bradley_terry_loss(rm.reward_from_hidden(hc), rm.reward_from_hidden(hr)).item()
    for _ in range(20):
        opt.zero_grad()
        loss = bradley_terry_loss(rm.reward_from_hidden(hc), rm.reward_from_hidden(hr))
        loss.backward()
        opt.step()
    after = bradley_terry_loss(rm.reward_from_hidden(hc), rm.reward_from_hidden(hr)).item()
    assert after < before


# --------------------------------------------------------------------------- #
# sequence_logprob + DPO loss
# --------------------------------------------------------------------------- #
def test_sequence_logprob_sums_token_logps_and_skips_ignore_index():
    logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]])  # (1, 2, 3)
    labels = torch.tensor([[0, 2]])
    lp = sequence_logprob(logits, labels)
    logp = F.log_softmax(logits, dim=-1)
    want = logp[0, 0, 0] + logp[0, 1, 2]
    assert lp.shape == (1,)
    assert torch.allclose(lp[0], want, atol=1e-6)

    # ignore_index positions contribute nothing.
    labels_masked = torch.tensor([[0, -100]])
    lp2 = sequence_logprob(logits, labels_masked)
    assert torch.allclose(lp2[0], logp[0, 0, 0], atol=1e-6)


def test_dpo_loss_matches_hand_computed_value():
    # Hand-computed (verified in py):
    # pi_logratios = -2 - (-3) = 1 ; ref_logratios = -2.5 - (-2.5) = 0
    # logits = 1 ; beta=0.5 -> beta*logits=0.5 ; loss = -log sigmoid(0.5) = 0.4740769863
    pcw = torch.tensor([-2.0])
    pcl = torch.tensor([-3.0])
    rcw = torch.tensor([-2.5])
    rcl = torch.tensor([-2.5])
    loss = dpo_loss(pcw, pcl, rcw, rcl, beta=0.5)
    assert abs(loss.item() - 0.4740769863128662) < 1e-6, loss.item()

    # Reproduce it independently.
    want = -F.logsigmoid(0.5 * ((pcw - pcl) - (rcw - rcl)))
    assert torch.allclose(loss, want.mean(), atol=1e-6)


def test_dpo_gradient_increases_the_chosen_minus_rejected_margin():
    # Treat the policy sequence log-probs as the trainable quantities (toy proxy
    # for the policy's parameters). A few DPO steps must widen the policy margin
    # log pi(y_w) - log pi(y_l).
    torch.manual_seed(0)
    policy_chosen = torch.tensor([-2.0, -1.5], requires_grad=True)
    policy_rejected = torch.tensor([-2.0, -1.5], requires_grad=True)
    ref_chosen = torch.tensor([-2.0, -1.5])
    ref_rejected = torch.tensor([-2.0, -1.5])

    def margin():
        return (policy_chosen - policy_rejected).mean().item()

    start = margin()
    opt = torch.optim.SGD([policy_chosen, policy_rejected], lr=0.5)
    for _ in range(10):
        opt.zero_grad()
        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
        loss.backward()
        opt.step()
    assert margin() > start + 1e-3, (start, margin())

    # implicit rewards: chosen reward ends up above rejected reward
    rc, rr = dpo_reward_margin(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    assert (rc > rr).all()


# --------------------------------------------------------------------------- #
# PPO clipped objective, advantages, KL
# --------------------------------------------------------------------------- #
def _obj(r, A, eps=0.2):
    """Reference scalar PPO objective from the raw ratio (not log-space)."""
    unclipped = r * A
    clipped = min(max(r, 1 - eps), 1 + eps) * A
    return min(unclipped, clipped)


def test_ppo_clip_matches_manual_including_clipped_branch():
    eps = 0.2
    cases = [
        (1.5, 2.0),   # A>0, ratio high -> clipped to 1.2*A = 2.4
        (0.7, 2.0),   # A>0, ratio low  -> unclipped 1.4 is the min
        (1.5, -2.0),  # A<0, ratio high -> unclipped -3.0 is the min
        (0.7, -2.0),  # A<0, ratio low  -> clipped -1.6 is the min
        (1.05, 2.0),  # inside band -> no clipping, 2.1
    ]
    for r, A in cases:
        old_lp = torch.tensor(0.0)
        lp = torch.tensor(math.log(r))               # so exp(lp - old_lp) == r
        adv = torch.tensor(A)
        got = ppo_clip_objective(lp, old_lp, adv, clip_eps=eps)
        assert abs(got.item() - _obj(r, A, eps)) < 1e-5, (r, A, got.item())

    # Explicit clipped-branch value.
    got = ppo_clip_objective(torch.tensor(math.log(1.5)), torch.tensor(0.0),
                             torch.tensor(2.0), clip_eps=0.2)
    assert abs(got.item() - 2.4) < 1e-5


def test_ppo_objective_reduces_over_a_batch():
    lp = torch.tensor([math.log(1.5), math.log(0.7)])
    old = torch.zeros(2)
    adv = torch.tensor([2.0, 2.0])
    mean = ppo_clip_objective(lp, old, adv, clip_eps=0.2)
    per = ppo_clip_objective(lp, old, adv, clip_eps=0.2, reduction="none")
    assert per.shape == (2,)
    assert torch.allclose(mean, per.mean(), atol=1e-6)
    # 2.4 (clipped) and 1.4 (unclipped) -> mean 1.9
    assert abs(mean.item() - 1.9) < 1e-5


def test_compute_advantages_and_kl_penalty():
    rewards = torch.tensor([1.0, 0.0, 2.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    adv = compute_advantages(rewards, values)
    assert torch.allclose(adv, torch.tensor([0.5, -0.5, 1.5]), atol=1e-6)
    norm = compute_advantages(rewards, values, normalize=True)
    assert abs(norm.mean().item()) < 1e-6

    # KL estimate is the mean log-ratio; equal policies -> 0.
    lp = torch.tensor([-1.0, -2.0])
    assert abs(kl_penalty(lp, lp).item()) < 1e-6
    ref = torch.tensor([-1.5, -2.5])
    assert torch.allclose(kl_penalty(lp, ref, reduction="none"),
                          torch.tensor([0.5, 0.5]), atol=1e-6)
