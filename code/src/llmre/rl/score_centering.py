"""Score centering: removing the drift term of the off-policy policy gradient.

This extends lesson 16.4 and :mod:`llmre.rl.mismatch`, which are unchanged. With
trainer next-token probabilities ``p``, sampler probabilities ``q`` and scores
``s_v = grad log p_v``, the policy-gradient update splits per prefix into

    E_q[R s_y] = E_q[R] * s_bar  +  Cov_q(R, s_y),     s_bar = sum_v q_v s_v.

The first term ("drift") pushes the trainer toward the sampler whatever the
rewards are. Subtracting ``s_bar`` from each sampled score removes it. Reference:
Marek and Ryabinin, "Score Centering Stabilizes Off-policy Reinforcement
Learning", arXiv 2609.20807 (equations 3-5, 8-14).

Functions return objectives to *maximize*, like :mod:`llmre.rl.grpo`. A
``weight_fn`` maps importance ratios ``p / q`` to weights (see :func:`tis_weight`).
Importance weighting is applied first and centering on top of it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

WeightFn = Callable[[torch.Tensor], torch.Tensor]


def tis_weight(cap: float = 2.0) -> WeightFn:
    """Truncated importance weight ``f(r) = min(r, cap)``."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    return lambda r: r.clamp(max=cap)


def _unit_weight(r: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(r)


def _reduce(per_token: torch.Tensor, mask: torch.Tensor, reduction: str) -> torch.Tensor:
    per_token = per_token.masked_fill(~mask, 0.0)
    if reduction == "none":
        return per_token
    if reduction == "sum":
        return per_token.sum()
    if reduction == "mean":
        return per_token.sum() / mask.sum().clamp_min(1)
    raise ValueError(f"unknown reduction {reduction!r}")


def _mask_and_rewards(tokens: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor | None):
    if advantages.dim() != 1 or advantages.shape[0] != tokens.shape[0]:
        raise ValueError("advantages must have shape (B,)")
    m = torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask.bool()
    return m, advantages.detach().unsqueeze(-1)


def score_centered_objective(
    train_logp: torch.Tensor,
    sampler_logp: torch.Tensor,
    tokens: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor | None = None,
    weight_fn: WeightFn | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Exact score centering, with the sampler's full next-token distribution.

    train_logp, sampler_logp: (B, T, V) float32; tokens: (B, T) int64;
    advantages: (B,) float32; mask: (B, T) bool, True on response tokens.

    Per token: ``R * (sg[w_y] log p_y - sum_v sg[q_v w_v] log p_v)``, whose
    gradient is ``R * (w_y s_y - sum_v q_v w_v s_v)``.
    """
    f = weight_fn or _unit_weight
    m, R = _mask_and_rewards(tokens, advantages, mask)
    q = sampler_logp.detach().exp()
    w = f((train_logp.detach() - sampler_logp.detach()).exp())
    sampled = train_logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    w_y = w.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    correction = ((q * w).detach() * train_logp).sum(dim=-1)
    return _reduce(R * (w_y.detach() * sampled - correction), m, reduction)


def top_k_score_centered_objective(
    train_logp: torch.Tensor,
    head_ids: torch.Tensor,
    head_sampler_logp: torch.Tensor,
    tokens: torch.Tensor,
    sampled_sampler_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor | None = None,
    weight_fn: WeightFn | None = None,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Score centering when the engine logged only its top-k tokens per step.

    train_logp: (B, T, V); head_ids, head_sampler_logp: (B, T, k) from the
    engine; sampled_sampler_logp: (B, T), ``log q`` of the sampled token, which
    may lie outside the head; tokens: (B, T); advantages: (B,).

    The sampler's unknown tail is modelled as the trainer's tail rescaled by
    ``rho = q_tail_mass / p_tail_mass``. The per-token objective is
    ``R * (sg[w_y] log p_y - sum_{v in head} sg[q_v w_v - alpha p_v] log p_v)``,
    with ``alpha = rho * f(1 / rho)``. Tail masses are floored at ``eps``.
    """
    f = weight_fn or _unit_weight
    m, R = _mask_and_rewards(tokens, advantages, mask)
    train_head = train_logp.gather(-1, head_ids)
    q_head = head_sampler_logp.detach().exp()
    p_head = train_head.detach().exp()
    q_tail = (1.0 - q_head.sum(dim=-1, keepdim=True)).clamp_min(eps)
    p_tail = (1.0 - p_head.sum(dim=-1, keepdim=True)).clamp_min(eps)
    rho = q_tail / p_tail
    alpha = rho * f(1.0 / rho)
    w_head = f((train_head.detach() - head_sampler_logp.detach()).exp())
    coef = (q_head * w_head - alpha * p_head).detach()
    correction = (coef * train_head).sum(dim=-1)
    sampled = train_logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    w_y = f((sampled.detach() - sampled_sampler_logp.detach()).exp())
    return _reduce(R * (w_y.detach() * sampled - correction), m, reduction)


def drift_signal_decomposition(
    train_probs: torch.Tensor, sampler_probs: torch.Tensor, rewards: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(drift, signal)`` for one softmax step, enumerated over its V actions.

    All inputs (V,). Since ``grad_theta log p_a = e_a - p``, ``s_bar = q - p``.
    ``drift + signal`` is the expected uncorrected update.
    """
    mean_r = (sampler_probs * rewards).sum()
    drift = mean_r * (sampler_probs - train_probs)
    signal = sampler_probs * (rewards - mean_r)
    return drift, signal


def centered_bandit_gradient(
    theta: torch.Tensor,
    rewards: torch.Tensor,
    ref_logp: torch.Tensor,
    beta: float,
    sampler_logp: torch.Tensor,
    cap: float | None = None,
) -> torch.Tensor:
    """Expected score-centered update on the KL-regularized bandit of 16.4.

    Same setup as :func:`llmre.rl.mismatch.softmax_bandit_gradient`. ``cap=None``
    is plain centering; a number composes it with ``min(pi / mu, cap)``.
    """
    logp = torch.log_softmax(theta, dim=-1)
    p = logp.exp()
    mu = sampler_logp.exp()
    R = rewards - beta * (logp - ref_logp)
    mw = mu if cap is None else torch.minimum(p, cap * mu)  # mu(a) * w(a)
    mean_r = (mu * R).sum()
    weighted = mw * R - p * (mw * R).sum()
    centre = mw - p * mw.sum()
    return weighted - mean_r * centre


def train_centered_bandit(
    rewards: torch.Tensor,
    ref_logp: torch.Tensor,
    beta: float,
    sampler_bias: torch.Tensor,
    cap: float | None = None,
    lr: float = 0.5,
    steps: int = 2000,
) -> torch.Tensor:
    """Exact gradient ascent with :func:`centered_bandit_gradient`; returns log pi."""
    ref = ref_logp.to(torch.float64)
    theta = ref.clone()
    rewards = rewards.to(torch.float64)
    bias = sampler_bias.to(torch.float64)
    for _ in range(steps):
        sampler_logp = torch.log_softmax(theta + bias, dim=-1)
        theta = theta + lr * centered_bandit_gradient(theta, rewards, ref, beta, sampler_logp, cap)
    return torch.log_softmax(theta, dim=-1)
