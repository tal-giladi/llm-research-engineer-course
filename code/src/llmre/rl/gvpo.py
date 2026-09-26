"""GVPO: Group Variance Policy Optimization (reading-level implementation).

This is Frontier update F.2. GRPO (:mod:`llmre.rl.grpo`) weights each sampled
completion by an importance ratio ``pi_theta / pi_old`` and clips it. Ratios are
unbounded in principle, which is one source of GRPO's instability.

GVPO (Zhang et al., arXiv 2504.19599; extended as GVPO++ in arXiv 2609.21432)
starts from the closed-form optimum of KL-constrained reward maximization::

    pi*(y|x) = pi_ref(y|x) * exp(R(x, y) / beta) / Z(x)

Rearranged, the *implicit reward* of a policy is
``R_theta(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x))``, and at the optimum
``R_theta = R - beta * log Z(x)``. ``Z(x)`` is intractable, but it is the same
constant for every completion of the same prompt, so it cancels when you
**center within the group**. GVPO minimizes the squared gap between centered
implicit reward and centered actual reward::

    L = 1/2 * sum_p sum_i [ (R_theta[p,i] - mean_i R_theta[p,:])
                          - (R[p,i]       - mean_i R[p,:]) ] ^ 2

No importance ratio, no clipping, and the loss is exactly 0 when
``pi_theta`` is proportional to ``pi_ref * exp(R / beta)`` on the sampled set.
This module is for understanding the idea; it is not a validated trainer.
"""

from __future__ import annotations

import torch


def gvpo_loss(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    beta: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """GVPO loss (to *minimize*) for a batch of prompt groups.

    Args:
        logprobs: ``log pi_theta(y_i|x_p)``, shape ``(num_prompts, group_size)``,
            float, differentiable (summed token log-probs per completion).
        ref_logprobs: ``log pi_ref(y_i|x_p)``, same shape, no grad.
        rewards: scalar reward per completion, same shape.
        beta: KL-constraint strength; smaller beta lets the policy move further.
        reduction: ``"mean"`` over prompts (default), ``"sum"``, or ``"none"``
            (per-prompt loss, shape ``(num_prompts,)``).

    Returns:
        Scalar loss, or ``(num_prompts,)`` for ``reduction="none"``.
    """
    if logprobs.dim() != 2:
        raise ValueError(
            f"logprobs must be (num_prompts, group_size), got {tuple(logprobs.shape)}"
        )
    implicit = beta * (logprobs - ref_logprobs)                       # (P, G)
    implicit_c = implicit - implicit.mean(dim=1, keepdim=True)        # Z(x) cancels
    reward_c = rewards - rewards.mean(dim=1, keepdim=True)            # (P, G)
    per_prompt = 0.5 * ((implicit_c - reward_c) ** 2).sum(dim=1)      # (P,)
    if reduction == "none":
        return per_prompt
    if reduction == "sum":
        return per_prompt.sum()
    if reduction == "mean":
        return per_prompt.mean()
    raise ValueError(f"unknown reduction {reduction!r}")
