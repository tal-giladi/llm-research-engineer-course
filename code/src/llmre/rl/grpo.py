"""GRPO: Group Relative Policy Optimization — PPO without a critic.

This is Module 16, lesson 16.1. PPO (lesson 15.2, :mod:`llmre.rl.ppo`) needs a
learned **value / critic** network to produce the baseline that turns a raw
return into an advantage ``A = R - V``. For an LLM that critic is a second
network the size of the policy: extra memory, extra compute, and it is finicky
to train (a bad value estimate injects bias and instability into every update).

**GRPO** (Shao et al. 2024, *DeepSeekMath*) removes the critic entirely. For each
prompt it samples a **group** of ``G`` completions, scores them, and uses the
group's own reward statistics as the baseline. The advantage of completion ``i``
is how far its reward sits from the group mean, in units of the group's spread::

    A_i = (r_i - mean(r)) / std(r)

That is a baseline computed *from the samples themselves* — no learned value
network. It works especially well when the reward is **verifiable** (lesson
16.2): the reward is cheap and exact, so sampling a group and comparing within it
is a reliable signal.

The optimization objective is PPO's clipped surrogate, unchanged, but fed the
group-relative advantage (and in full GRPO a KL-to-reference term is added, as in
PPO — see :func:`llmre.rl.ppo.kl_penalty`).

Everything here is plain tensor math on tiny inputs so the mechanism is
unambiguous; wiring it to a real GPT policy is the toy loop in
:mod:`llmre.reasoning.rlvr` and the lesson notes.
"""

from __future__ import annotations

import torch


def grpo_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Group-relative advantages: standardize each row of ``rewards``.

    Each **row** is one prompt's group of ``G`` sampled completions; each column
    is one completion's scalar reward. We standardize within the row::

        A[p, i] = (rewards[p, i] - mean_i rewards[p, :]) / (std_i rewards[p, :])

    so that within every group the advantages have mean ~0 and unit spread. A
    completion that beat its group average gets a positive advantage (push its
    tokens up); one that did worse gets a negative advantage (push them down).
    The group mean *is* the baseline — this replaces PPO's learned value network.

    The standard deviation is the **population** std (``unbiased=False``), so a
    row's advantages have population std exactly 1. If a group is unanimous
    (every reward equal, e.g. all-correct or all-wrong), the std is 0 and the row
    carries no signal; we guard the division with ``eps`` and return all-zero
    advantages for that row.

    Args:
        rewards: shape ``(num_prompts, group_size)`` float, any device. Row ``p``
            holds the ``G`` rewards for prompt ``p``'s sampled group.
        eps: small constant added to the std before dividing, to guard the
            std-0 (unanimous group) case.

    Returns:
        Advantages, shape/dtype/device identical to ``rewards``.
    """
    if rewards.dim() != 2:
        raise ValueError(
            f"rewards must be (num_prompts, group_size), got shape {tuple(rewards.shape)}"
        )
    mean = rewards.mean(dim=1, keepdim=True)                  # (num_prompts, 1)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)    # (num_prompts, 1)
    return (rewards - mean) / (std + eps)                     # (num_prompts, group_size)


def grpo_objective(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    reduction: str = "mean",
) -> torch.Tensor:
    """GRPO clipped surrogate **objective** (a quantity to *maximize*).

    Identical in form to PPO's clipped objective
    (:func:`llmre.rl.ppo.ppo_clip_objective`) — only the advantage differs: here
    it is the group-relative advantage from :func:`grpo_advantages` rather than
    ``reward - value``. Per element, with ratio ``r = exp(logprobs - old_logprobs)``::

        unclipped = r * A
        clipped   = clip(r, 1-eps, 1+eps) * A
        obj       = min(unclipped, clipped)

    The ``min`` caps the incentive to move the ratio far outside
    ``[1-eps, 1+eps]`` in a single update, so a group of samples can be reused for
    several gradient steps without the policy running away. On the very first
    on-policy step ``logprobs == old_logprobs`` so ``r == 1`` and the objective
    reduces to ``A * logprobs`` — the plain (group-baselined) policy gradient.

    This returns the **objective**; optimizers minimize, so the training loss is
    ``-grpo_objective(...)``.

    Args:
        logprobs: ``log pi_theta(a|s)`` under the current policy, shape ``(...,)``
            float, differentiable. For an LLM these are summed per-token
            log-probs of a sampled completion; here any shape works.
        old_logprobs: ``log pi_old(a|s)`` from the policy that generated the
            group, same shape (constant / no grad).
        advantages: group-relative ``A``, same shape (e.g. the output of
            :func:`grpo_advantages` flattened to match).
        clip_eps: clip half-width ``eps`` (typically ``0.2``).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar (``"mean"``/``"sum"``) or per-element ``(...,)`` (``"none"``).
    """
    ratio = torch.exp(logprobs - old_logprobs)                        # (...,)
    unclipped = ratio * advantages                                    # (...,)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    obj = torch.minimum(unclipped, clipped)                           # (...,)
    if reduction == "none":
        return obj
    if reduction == "sum":
        return obj.sum()
    if reduction == "mean":
        return obj.mean()
    raise ValueError(f"unknown reduction {reduction!r}")
