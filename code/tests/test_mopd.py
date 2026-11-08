"""Tests for the 17.4 advanced extension: multi-teacher on-policy distillation (MOPD)."""

import math

import pytest
import torch
import torch.nn.functional as F

from llmre.reasoning.distill import expected_sequence_divergence, reverse_kl
from llmre.reasoning.mopd import (
    distill_two_domains,
    domain_rkl,
    make_two_domain_teachers,
    mopd_advantages,
    mopd_loss,
    route_teacher_logprobs,
    teacher_token_share,
    token_share_weights,
    truncated_is_weights,
)

PT = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
PS = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)


def test_hand_example_advantages():
    # Lesson 17.4 extension, worked example: A = ln 2, 0, -ln 4; E_{p_S}[A] = -0.5 ln 2.
    adv = mopd_advantages(PT.log().unsqueeze(0), PS.log().unsqueeze(0))[0]
    assert adv.tolist() == pytest.approx([math.log(2), 0.0, -math.log(4)])
    assert (PS * adv).sum().item() == pytest.approx(-0.5 * math.log(2))
    assert (PS * adv).sum().item() == pytest.approx(-reverse_kl(PT.log(), PS.log()).item())


def test_advantage_is_detached_and_adds_orm():
    s = PS.log().unsqueeze(0).clone().requires_grad_(True)
    adv = mopd_advantages(PT.log().unsqueeze(0), s, torch.tensor([1.0], dtype=torch.float64), alpha=0.5)
    assert not adv.requires_grad
    assert adv[0].tolist() == pytest.approx([math.log(2) + 0.5, 0.5, -math.log(4) + 0.5])


def test_expected_policy_gradient_equals_reverse_kl_gradient():
    # Reduction to the foundational 17.4 objective: weight each possible sampled token
    # by p_S (exact expectation); the MOPD surrogate's gradient is grad KL(p_S || p_T).
    z = PS.log().clone().requires_grad_(True)
    log_ps = F.log_softmax(z, -1)
    adv = mopd_advantages(PT.log(), log_ps)
    surrogate = -(log_ps.exp().detach() * adv * log_ps).sum()
    g_mopd, = torch.autograd.grad(surrogate, z)
    z2 = PS.log().clone().requires_grad_(True)
    g_rkl, = torch.autograd.grad(reverse_kl(PT.log(), z2), z2)
    assert torch.allclose(g_mopd, g_rkl)
    assert g_rkl.tolist() == pytest.approx([-0.3120, -0.1040, 0.4159], abs=1e-4)


def test_routing_picks_each_sequences_teacher():
    all_t = torch.arange(2 * 3 * 4, dtype=torch.float64).view(2, 3, 4)  # (K=2, B=3, T=4)
    ids = torch.tensor([1, 0, 1])
    out = route_teacher_logprobs(all_t, ids)
    assert out.shape == (3, 4)
    for b in range(3):
        assert torch.equal(out[b], all_t[ids[b], b])
    with pytest.raises(ValueError):
        route_teacher_logprobs(all_t, torch.tensor([0, 2, 1]))


def test_truncated_is_weights_drop_outside_band():
    pi = torch.log(torch.tensor([[0.3, 0.3, 0.3]], dtype=torch.float64))
    mu = torch.log(torch.tensor([[0.25, 0.1, 0.7]], dtype=torch.float64))
    w = truncated_is_weights(pi, mu, eps_low=0.5, eps_high=2.0)
    assert w[0].tolist() == pytest.approx([1.2, 0.0, 0.0])


def test_loss_masks_prompt_and_only_student_gets_gradient():
    g = torch.Generator().manual_seed(0)
    s = (-torch.randn(2, 5, generator=g, dtype=torch.float64).abs()).requires_grad_(True)
    t = (-torch.randn(2, 5, generator=g, dtype=torch.float64).abs()).requires_grad_(True)
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    loss = mopd_loss(s, t, mask)
    loss.backward()
    assert t.grad is None or torch.all(t.grad == 0)
    assert torch.all(s.grad[mask == 0] == 0)
    assert torch.all(s.grad[mask == 1] != 0)


def test_sequence_vs_token_normalization():
    # One long and one short sequence with constant per-token terms 1 and 3.
    s = torch.full((2, 4), -1.0, dtype=torch.float64)
    t = torch.tensor([[-1.0] * 4, [-1.0] * 4], dtype=torch.float64)
    t[0] += 1.0  # adv = 1 on seq 0
    t[1] += 3.0  # adv = 3 on seq 1
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])
    # per-token term = -adv * log p = adv (log p = -1)
    assert mopd_loss(s, t, mask, normalize="sequence").item() == pytest.approx((1 + 3) / 2)
    assert mopd_loss(s, t, mask, normalize="token").item() == pytest.approx((4 * 1 + 3) / 5)


def test_token_share_and_balancing_weights():
    # Lesson example: 2 math answers of 300 tokens, 2 code of 250, 1 IF of 20.
    lengths = [300, 300, 250, 250, 20]
    mask = torch.zeros(5, 300)
    for b, n in enumerate(lengths):
        mask[b, :n] = 1
    ids = torch.tensor([0, 0, 1, 1, 2])
    share = teacher_token_share(mask, ids, 3)
    assert share.tolist() == pytest.approx([600 / 1120, 500 / 1120, 20 / 1120])
    w = token_share_weights(mask, ids, 3)
    weighted = torch.zeros(3, dtype=torch.float64).index_add_(0, ids, w * mask.sum(-1).double())
    assert (weighted / weighted.sum()).tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_single_teacher_reduces_to_foundational_on_policy_objective():
    tA, _, starts = make_two_domain_teachers()
    g = torch.Generator().manual_seed(1)
    s = torch.randn(4, 4, generator=g, dtype=torch.float64)
    for start in starts:
        ours = domain_rkl(tA, s, start, 8)
        ref = expected_sequence_divergence(tA, s, "student", "reverse", start, 8)
        assert ours.item() == pytest.approx(ref.item())


def test_routed_beats_single_teacher_on_the_other_domain():
    tA, tB, starts = make_two_domain_teachers()
    routed = distill_two_domains([tA, tB], starts, [8, 8], [0.5, 0.5], "routed", "sequence")
    single = distill_two_domains([tA, tB], starts, [8, 8], [0.5, 0.5], "teacher0", "sequence")
    assert single["rkl"][0] < 1e-2  # teacher A's domain is learned perfectly...
    assert single["rkl"][1] > 3.0  # ...and domain B is lost
    assert routed["rkl"][1] < single["rkl"][1] - 2.0
    assert routed["rkl"][0] == pytest.approx(routed["rkl"][1], abs=1e-3)  # symmetric compromise


def test_token_normalization_hands_shared_capacity_to_long_answers():
    tA, tB, starts = make_two_domain_teachers()
    seq = distill_two_domains([tA, tB], starts, [24, 4], [0.5, 0.5], "routed", "sequence")
    tok = distill_two_domains([tA, tB], starts, [24, 4], [0.5, 0.5], "routed", "token")
    assert tok["domain_weight"] == pytest.approx([24 / 28, 4 / 28])
    assert tok["rkl"][0] < 0.05 and tok["rkl"][1] > 2.0  # short-answer domain nearly lost
    assert seq["rkl"][1] < tok["rkl"][1] - 1.0
    assert tok["p_alternate"] > 0.85 > 0.55 > seq["p_alternate"]
