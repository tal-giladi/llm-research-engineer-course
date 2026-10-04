"""Logit distillation: forward vs reverse KL, and on-policy distillation (lesson 17.4).

This extends :mod:`llmre.reasoning.r1_pipeline` (lesson 17.3). There, "distillation"
meant *sequence-level* distillation: sample complete answers from a big teacher and
plain-SFT the student on them with cross-entropy against the sampled tokens. This
module adds the two ideas that modern distillation pipelines layer on top:

1. **Logit (token-level) distillation.** Instead of one sampled token per position,
   the student matches the teacher's *whole* next-token distribution. Which way the
   KL points matters:

   * forward KL  ``KL(p_T || p_S) = sum_v p_T(v) (log p_T(v) - log p_S(v))`` —
     "mode covering": the student is punished wherever the teacher has mass and the
     student has little. With a one-hot teacher it *is* the SFT cross-entropy.
   * reverse KL  ``KL(p_S || p_T) = sum_v p_S(v) (log p_S(v) - log p_T(v))`` —
     "mode seeking": the student is punished wherever *it* puts mass the teacher
     does not, so a student with too little capacity picks one teacher mode.
   * generalized JSD with ``beta`` (GKD, Agarwal et al. 2023,
     https://arxiv.org/abs/2306.13649): ``beta=0`` -> forward KL, ``beta=1`` ->
     reverse KL, in between a bounded mixture. This is the convention used by TRL's
     ``DistillationTrainer``.

2. **On-policy distillation.** *Which states* (prefixes) the divergence is averaged
   over. Off-policy = prefixes the teacher produced (or a fixed dataset). On-policy =
   prefixes the *student* produces, scored token-by-token by the teacher. A student
   that makes a mistake at inference lands in prefixes the teacher never visits;
   only on-policy training ever shows it those prefixes.

To make the on-/off-policy difference exact and seed-free, the toy "language model"
here is a **bigram Markov chain** over ``V`` tokens: the policy is a ``(V, V)`` table
of next-token logits, one row per previous token. For such a chain the expected
loss over all length-``T`` rollouts can be computed exactly from state-visitation
distributions (``d_{t+1} = d_t @ P``), so no sampling noise hides the effect. Real
systems estimate the same expectation from sampled rollouts.

All tensors are small CPU ``float32``/``float64`` tensors; everything runs in well
under a second.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Divergences over the vocabulary (last dim)
# --------------------------------------------------------------------------- #
def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Token-level forward KL ``KL(p_T || p_S)``, summed over the vocabulary.

    Args:
        teacher_logits: ``(..., V)`` logits of the teacher (treated as constants).
        student_logits: ``(..., V)`` logits of the student, same shape.
        mask: optional ``(...)`` tensor, 1 for positions that count (e.g. completion
            tokens), 0 for prompt/padding.
        reduction: ``"mean"`` (masked mean over positions) or ``"none"`` (``(...)``).
    """
    log_pt = F.log_softmax(teacher_logits.detach(), dim=-1)
    log_ps = F.log_softmax(student_logits, dim=-1)
    kl = (log_pt.exp() * (log_pt - log_ps)).sum(-1)  # (...)
    return kl if reduction == "none" else _masked_mean(kl, mask)


def reverse_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Token-level reverse KL ``KL(p_S || p_T)``, summed over the vocabulary.

    Exact over the vocabulary (not the one-sample log-ratio estimator of lesson
    15.2): the expectation is taken under ``p_S`` analytically at each position.
    Same arguments as :func:`forward_kl`.
    """
    log_pt = F.log_softmax(teacher_logits.detach(), dim=-1)
    log_ps = F.log_softmax(student_logits, dim=-1)
    kl = (log_ps.exp() * (log_ps - log_pt)).sum(-1)
    return kl if reduction == "none" else _masked_mean(kl, mask)


def generalized_jsd(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float = 0.5,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Generalized Jensen-Shannon divergence of GKD, in TRL's ``beta`` convention.

    ``M = (1 - beta) p_S + beta p_T`` and
    ``JSD_beta = beta KL(p_T || M) + (1 - beta) KL(p_S || M)``.

    ``beta=0`` returns :func:`forward_kl` and ``beta=1`` returns :func:`reverse_kl`
    (special-cased, as TRL does). The mixture formula itself goes to 0 at both ends;
    it is ``JSD_beta / beta`` that tends to forward KL as ``beta -> 0`` and
    ``JSD_beta / (1 - beta)`` that tends to reverse KL as ``beta -> 1``.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if beta == 0.0:
        return forward_kl(teacher_logits, student_logits, mask, reduction)
    if beta == 1.0:
        return reverse_kl(teacher_logits, student_logits, mask, reduction)
    log_pt = F.log_softmax(teacher_logits.detach(), dim=-1)
    log_ps = F.log_softmax(student_logits, dim=-1)
    # log M = logsumexp(log(1-beta) + log p_S, log(beta) + log p_T), computed stably.
    log_m = torch.logsumexp(
        torch.stack([log_ps + math.log(1.0 - beta), log_pt + math.log(beta)]), dim=0
    )
    kl_t_m = (log_pt.exp() * (log_pt - log_m)).sum(-1)
    kl_s_m = (log_ps.exp() * (log_ps - log_m)).sum(-1)
    jsd = beta * kl_t_m + (1.0 - beta) * kl_s_m
    return jsd if reduction == "none" else _masked_mean(jsd, mask)


def divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    kind: str,
    reduction: str = "none",
) -> torch.Tensor:
    """Dispatch: ``kind`` in ``{"forward", "reverse"}`` or ``"jsd:<beta>"``."""
    if kind == "forward":
        return forward_kl(teacher_logits, student_logits, reduction=reduction)
    if kind == "reverse":
        return reverse_kl(teacher_logits, student_logits, reduction=reduction)
    if kind.startswith("jsd:"):
        return generalized_jsd(
            teacher_logits, student_logits, float(kind[4:]), reduction=reduction
        )
    raise ValueError(f"unknown divergence {kind!r}")


# --------------------------------------------------------------------------- #
# Demo 1: mode covering vs mode seeking with a too-small student
# --------------------------------------------------------------------------- #
def discretized_gaussian_logits(mu: torch.Tensor, log_sigma: torch.Tensor, V: int) -> torch.Tensor:
    """Logits ``-(i - mu)^2 / (2 sigma^2)`` over bins ``i = 0..V-1``: a one-bump student.

    The student has only two parameters, so it *cannot* represent a two-bump teacher.
    Returns a ``(V,)`` tensor with the dtype of ``mu``.
    """
    i = torch.arange(V, dtype=mu.dtype)
    return -((i - mu) ** 2) / (2.0 * torch.exp(2.0 * log_sigma))


def bimodal_teacher_probs(V: int = 20, modes: tuple[int, int] = (4, 15), width: float = 1.0) -> torch.Tensor:
    """A two-bump teacher distribution over ``V`` bins, ``(V,)`` float64, sums to 1."""
    i = torch.arange(V, dtype=torch.float64)
    p = sum(torch.exp(-((i - m) ** 2) / (2 * width**2)) for m in modes)
    return p / p.sum()


def fit_unimodal_student(
    teacher_probs: torch.Tensor,
    kind: str,
    init_mu: float = 6.0,
    init_log_sigma: float = math.log(1.5),
    steps: int = 2000,
    lr: float = 0.05,
) -> dict:
    """Fit the 2-parameter student to ``teacher_probs`` by minimizing ``kind`` divergence.

    The default start (``mu=6``, ``sigma=1.5``) sits near the left teacher mode.
    Reverse KL is non-convex here: started in the empty gap between the modes it can
    stall in a wide, flat solution, so the start matters (see lesson 17.4).

    Returns ``{"mu", "sigma", "probs", "loss"}`` (floats and a ``(V,)`` tensor).
    """
    V = teacher_probs.shape[-1]
    t_logits = teacher_probs.clamp_min(1e-300).log()
    mu = torch.tensor(float(init_mu), dtype=torch.float64, requires_grad=True)
    log_sigma = torch.tensor(float(init_log_sigma), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = divergence(t_logits, discretized_gaussian_logits(mu, log_sigma, V), kind, "mean")
        loss.backward()
        opt.step()
    with torch.no_grad():
        s_logits = discretized_gaussian_logits(mu, log_sigma, V)
        return {
            "mu": mu.item(),
            "sigma": log_sigma.exp().item(),
            "probs": F.softmax(s_logits, -1),
            "loss": divergence(t_logits, s_logits, kind, "mean").item(),
        }


# --------------------------------------------------------------------------- #
# Demo 2: on-policy vs off-policy distillation on a bigram "language model"
# --------------------------------------------------------------------------- #
def make_teacher_chain(V: int = 8, n_core: int = 4, p_stay: float = 0.9999) -> tuple[torch.Tensor, int]:
    """A teacher bigram LM that almost always cycles through tokens ``0..n_core-1``.

    From a core token ``s`` it emits ``(s + 1) % n_core`` with probability ``p_stay``
    and spreads the rest over the "off-road" tokens ``n_core..V-1``. From an off-road
    token it returns to token 0 with probability 0.9: the teacher knows how to
    recover, but it almost never needs to, so its own rollouts almost never show it.

    Returns ``(teacher_logits (V, V) float64, start_token)``.
    """
    P = torch.zeros(V, V, dtype=torch.float64)
    n_off = V - n_core
    for s in range(n_core):
        P[s, (s + 1) % n_core] = p_stay
        P[s, n_core:] = (1.0 - p_stay) / n_off
    for s in range(n_core, V):
        P[s, 0] = 0.9
        P[s, 1:] = 0.1 / (V - 1)
    P = P + 1e-6  # tiny floor so every log is finite
    P = P / P.sum(-1, keepdim=True)
    return P.log(), 0


def slipping_student_logits(logits: torch.Tensor, slip: float) -> torch.Tensor:
    """Log-probs of ``(1 - slip) * softmax(logits) + slip / V``.

    Models a student that can never be perfect (finite capacity, sampling
    temperature): whatever it learns, a fraction ``slip`` of its probability mass is
    spread uniformly, so it sometimes steps "off-road". ``slip=0`` is the plain
    softmax policy.
    """
    V = logits.shape[-1]
    return torch.log((1.0 - slip) * F.softmax(logits, -1) + slip / V)


def state_visitation(trans_probs: torch.Tensor, start: int, T: int) -> torch.Tensor:
    """Exact distribution over the previous token at steps ``0..T-1``: ``(T, V)``.

    ``d_0`` is one-hot on ``start`` and ``d_{t+1} = d_t @ P``. Row ``t`` is the
    probability that the prefix ends in each token when the model picks token ``t+1``.
    """
    V = trans_probs.shape[0]
    d = torch.zeros(V, dtype=trans_probs.dtype)
    d[start] = 1.0
    out = []
    for _ in range(T):
        out.append(d)
        d = d @ trans_probs
    return torch.stack(out)


def expected_sequence_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    weights_from: str,
    kind: str,
    start: int,
    T: int,
) -> torch.Tensor:
    """Expected per-token divergence over length-``T`` rollouts.

    ``weights_from="teacher"`` averages over prefixes the teacher generates
    (off-policy); ``"student"`` over prefixes the student generates (on-policy).
    The visitation weights are *detached*: as in GKD, gradients do not flow through
    the sampling of the prefixes, only through the student's next-token distribution.
    """
    per_state = divergence(teacher_logits, student_logits, kind, "none")  # (V,)
    src = teacher_logits if weights_from == "teacher" else student_logits
    d = state_visitation(F.softmax(src.detach(), -1), start, T)  # (T, V)
    return (d * per_state).sum(-1).mean()


def distill_markov(
    teacher_logits: torch.Tensor,
    start: int,
    policy: str = "on",
    kind: str = "reverse",
    slip: float = 0.05,
    T: int = 32,
    steps: int = 300,
    lr: float = 2.0,
    n_core: int = 4,
) -> dict:
    """Distill the bigram teacher into a uniform-initialised, slipping bigram student.

    ``policy="off"`` weights each state's divergence by the teacher's visitation;
    ``policy="on"`` by the student's own (recomputed every step). Plain gradient
    descent on the same parameters with the same budget, so the *only* difference
    between the two runs is which prefixes the loss is averaged over.

    Returns a dict:
      * ``student_logits`` - learned ``(V, V)`` table (before the slip mixture);
      * ``train_loss`` - list of per-step training losses;
      * ``eval_on_policy_rkl`` - expected reverse KL to the teacher on the student's
        *own* rollouts, i.e. what a user sees at inference;
      * ``off_road_time`` - expected fraction of steps the student spends on tokens
        ``>= n_core``;
      * ``recovery_prob`` - mean probability, from an off-road token, of returning to
        token 0 (teacher: 0.9; untrained: ``1/V``).
    """
    weights_from = "student" if policy == "on" else "teacher"
    V = teacher_logits.shape[0]
    s_logits = torch.zeros(V, V, dtype=teacher_logits.dtype, requires_grad=True)
    opt = torch.optim.SGD([s_logits], lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        eff = slipping_student_logits(s_logits, slip)
        loss = expected_sequence_divergence(teacher_logits, eff, weights_from, kind, start, T)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        eff = slipping_student_logits(s_logits, slip)
        ev = expected_sequence_divergence(teacher_logits, eff, "student", "reverse", start, T)
        probs = F.softmax(eff, -1)
        d = state_visitation(probs, start, T)
        off_road = d[:, n_core:].sum(-1).mean()
        recovery = probs[n_core:, 0].mean()
    return {
        "student_logits": s_logits.detach(),
        "train_loss": losses,
        "eval_on_policy_rkl": ev.item(),
        "off_road_time": off_road.item(),
        "recovery_prob": recovery.item(),
    }
