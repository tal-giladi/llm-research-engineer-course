"""Tests for lesson 17.4: forward/reverse KL distillation and on-policy distillation."""

import math

import pytest
import torch
import torch.nn.functional as F

from llmre.reasoning.distill import (
    bimodal_teacher_probs,
    distill_markov,
    fit_unimodal_student,
    forward_kl,
    generalized_jsd,
    make_teacher_chain,
    reverse_kl,
    slipping_student_logits,
    state_visitation,
)

PT = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
PS = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)


def test_hand_example_values():
    # Lesson 17.4 section 3: forward = 0.6 ln 2 - 0.1 ln 4, reverse = 0.5 ln 2.
    assert forward_kl(PT.log(), PS.log()).item() == pytest.approx(0.6 * math.log(2) - 0.1 * math.log(4))
    assert reverse_kl(PT.log(), PS.log()).item() == pytest.approx(0.5 * math.log(2))


def test_kl_matches_torch_kl_div():
    g = torch.Generator().manual_seed(0)
    t = torch.randn(2, 5, 11, generator=g, dtype=torch.float64)
    s = torch.randn(2, 5, 11, generator=g, dtype=torch.float64)
    lt, ls = F.log_softmax(t, -1), F.log_softmax(s, -1)
    ref_fwd = F.kl_div(ls, lt, log_target=True, reduction="none").sum(-1)
    ref_rev = F.kl_div(lt, ls, log_target=True, reduction="none").sum(-1)
    assert torch.allclose(forward_kl(t, s, reduction="none"), ref_fwd)
    assert torch.allclose(reverse_kl(t, s, reduction="none"), ref_rev)


def test_forward_kl_with_one_hot_teacher_is_sft_cross_entropy():
    # The foundational SFT loss (lesson 14.2 / 17.3) is the special case of forward KL
    # where the teacher puts all its mass on the observed token.
    g = torch.Generator().manual_seed(1)
    s = torch.randn(7, 13, generator=g, dtype=torch.float64)
    y = torch.randint(0, 13, (7,), generator=g)
    one_hot_teacher = torch.full((7, 13), -1e4, dtype=torch.float64)
    one_hot_teacher[torch.arange(7), y] = 0.0
    assert forward_kl(one_hot_teacher, s).item() == pytest.approx(F.cross_entropy(s, y).item(), abs=1e-8)


def test_gradients_match_closed_forms():
    z = PS.log().clone().requires_grad_(True)
    forward_kl(PT.log(), z).backward()
    assert torch.allclose(z.grad, PS - PT)
    z = PS.log().clone().requires_grad_(True)
    rkl = reverse_kl(PT.log(), z)
    rkl.backward()
    expected = PS * (PS.log() - PT.log() - rkl.detach())
    assert torch.allclose(z.grad, expected)
    assert torch.allclose(expected, torch.tensor([-0.3119, -0.1040, 0.4159], dtype=torch.float64), atol=1e-4)


def test_teacher_gets_no_gradient():
    t = PT.log().clone().requires_grad_(True)
    s = PS.log().clone().requires_grad_(True)
    (forward_kl(t, s) + reverse_kl(t, s) + generalized_jsd(t, s, 0.3)).backward()
    assert t.grad is None and s.grad is not None


def test_generalized_jsd_endpoints_and_limits():
    t, s = PT.log(), PS.log()
    assert generalized_jsd(t, s, 0.0).item() == pytest.approx(forward_kl(t, s).item())
    assert generalized_jsd(t, s, 1.0).item() == pytest.approx(reverse_kl(t, s).item())
    eps = 1e-5
    assert (generalized_jsd(t, s, eps) / eps).item() == pytest.approx(forward_kl(t, s).item(), rel=1e-3)
    assert (generalized_jsd(t, s, 1 - eps) / eps).item() == pytest.approx(reverse_kl(t, s).item(), rel=1e-3)
    jsd = generalized_jsd(t, s, 0.5).item()
    assert 0.0 < jsd <= math.log(2)
    with pytest.raises(ValueError):
        generalized_jsd(t, s, 1.5)


def test_mask_ignores_prompt_positions():
    g = torch.Generator().manual_seed(2)
    t = torch.randn(1, 4, 6, generator=g, dtype=torch.float64)
    s = torch.randn(1, 4, 6, generator=g, dtype=torch.float64)
    mask = torch.tensor([[0, 0, 1, 1]])
    per_tok = forward_kl(t, s, reduction="none")
    assert forward_kl(t, s, mask).item() == pytest.approx(per_tok[0, 2:].mean().item())


def test_forward_kl_covers_reverse_kl_seeks():
    tp = bimodal_teacher_probs()
    gap = slice(7, 12)  # bins between the two teacher modes (4 and 15)
    fwd = fit_unimodal_student(tp, "forward")
    rev = fit_unimodal_student(tp, "reverse")
    # Forward KL spreads over both modes, putting real mass where the teacher has ~none.
    assert fwd["probs"][gap].sum() > 0.2 > 10 * tp[gap].sum()
    assert fwd["sigma"] > 5
    # Reverse KL locks onto the nearer mode and matches its shape; loss -> ln 2.
    assert rev["mu"] == pytest.approx(4.0, abs=0.05)
    assert rev["sigma"] == pytest.approx(1.0, abs=0.05)
    assert rev["loss"] == pytest.approx(math.log(2), abs=1e-3)


def test_state_visitation_rows_are_distributions():
    t, start = make_teacher_chain()
    d = state_visitation(F.softmax(t, -1), start, 10)
    assert d.shape == (10, 8)
    assert torch.allclose(d.sum(-1), torch.ones(10, dtype=torch.float64))
    assert d[0, start] == 1.0
    assert d[2, 2] > 0.99  # the teacher walks 0 -> 1 -> 2 almost surely


def test_slip_zero_is_plain_softmax():
    z = torch.randn(3, 5, dtype=torch.float64)
    assert torch.allclose(slipping_student_logits(z, 0.0), F.log_softmax(z, -1))


@pytest.mark.parametrize("kind", ["forward", "reverse"])
def test_on_policy_learns_recovery_off_policy_does_not(kind):
    t, start = make_teacher_chain()
    off = distill_markov(t, start, policy="off", kind=kind)
    on = distill_markov(t, start, policy="on", kind=kind)
    # Off-policy never sees off-road prefixes, so recovery stays at chance (1/V).
    assert off["recovery_prob"] == pytest.approx(1 / 8, abs=0.01)
    assert on["recovery_prob"] > 0.7
    # ...which is what the user experiences: less time lost, lower on-policy KL.
    assert on["off_road_time"] < 0.7 * off["off_road_time"]
    assert on["eval_on_policy_rkl"] < 0.85 * off["eval_on_policy_rkl"]
    # Both runs actually trained.
    assert on["train_loss"][-1] < on["train_loss"][0]
    assert off["train_loss"][-1] < off["train_loss"][0]
