"""The core RLHF/PPO math on tiny tensors.

This is Module 15, lesson 15.2. In RLHF the language model is a **policy**
``pi(token | context)``: at each step it emits an action (the next token), and at
the end of a generation the reward model (lesson 15.1) scores the completion.
We want to raise the probability of token choices that led to high reward.

The building blocks, from simplest to what production RLHF actually uses:

* **REINFORCE** policy gradient: ``grad J = E[ grad log pi(a) * R ]`` — scale the
  gradient of each action's log-prob by the return. Correct but high variance.
* **Baseline / advantage**: subtract a state-value baseline ``V(s)`` to get the
  **advantage** ``A = R - V``. Same expected gradient, much lower variance —
  actions are pushed up only if they did *better than expected*.
* **PPO clipped surrogate**: reuse a batch of samples for several gradient steps
  without the policy running away. With ratio ``r = pi_theta / pi_old``::

      L_clip = E[ min( r * A , clip(r, 1-eps, 1+eps) * A ) ]

  The ``min`` of the clipped and unclipped terms removes the incentive to move
  the ratio far outside ``[1-eps, 1+eps]``.
* **KL penalty** to a frozen reference policy, so the tuned model does not drift
  from the SFT model and reward-hack the reward model.

Full PPO-on-an-LLM is heavy (a policy, a value head, a reward model, a reference,
and rollout generation). Here we implement the *math* on small tensors so the
mechanism is unambiguous; wiring it to a real GPT is left to the lesson notes.
"""

from __future__ import annotations

import torch


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """Baseline-subtracted advantage ``A = rewards - values``.

    The **advantage** measures how much better an action's return was than the
    baseline (the value function's prediction). Subtracting a baseline leaves the
    policy-gradient estimate unbiased while cutting its variance, because the
    gradient now says "push this action up only if it beat expectations."

    This is the simplest possible baseline (the full PPO paper uses Generalized
    Advantage Estimation over a trajectory; here ``rewards`` and ``values`` are
    per-sample terminal quantities, so ``A = R - V`` is exactly GAE with one
    step).

    Args:
        rewards: shape ``(...,)`` float — the (terminal) return of each sample.
        values: shape ``(...,)`` float, same shape — the baseline ``V(s)``.
        normalize: if ``True``, standardize the advantages to zero mean / unit
            std (a common PPO stabilization trick; needs >1 element).

    Returns:
        Advantages, same shape/dtype/device as ``rewards``.
    """
    adv = rewards - values
    if normalize:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv


def ppo_clip_objective(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    reduction: str = "mean",
) -> torch.Tensor:
    """PPO clipped surrogate **objective** (a quantity to *maximize*).

    Per element, with probability ratio ``r = exp(logprobs - old_logprobs)``::

        unclipped = r * A
        clipped   = clip(r, 1-eps, 1+eps) * A
        obj       = min(unclipped, clipped)

    Taking the ``min`` means: when ``A > 0`` (good action) the gain is capped once
    ``r > 1+eps`` (no reward for shoving the probability arbitrarily high); when
    ``A < 0`` (bad action) the objective is capped once ``r < 1-eps``. Either way
    the policy has no incentive to step far from ``pi_old`` in one update.

    This returns the **objective**; an optimizer *minimizes*, so the training loss
    is ``-ppo_clip_objective(...)``.

    Args:
        logprobs: ``log pi_theta(a|s)`` under the current policy, shape ``(...,)``
            float, differentiable.
        old_logprobs: ``log pi_old(a|s)`` from the policy that generated the data,
            same shape (treat as constant / no grad).
        advantages: ``A``, same shape.
        clip_eps: clip half-width ``eps`` (typically ``0.2``).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar (``"mean"``/``"sum"``) or per-element ``(...,)`` (``"none"``).
    """
    ratio = torch.exp(logprobs - old_logprobs)                    # (...,)
    unclipped = ratio * advantages                                # (...,)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    obj = torch.minimum(unclipped, clipped)                       # (...,)
    if reduction == "none":
        return obj
    if reduction == "sum":
        return obj.sum()
    if reduction == "mean":
        return obj.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def kl_penalty(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Per-token KL estimate of the policy from a frozen reference, ``>= 0``-ish.

    RLHF adds ``-kl_coef * KL(pi_theta || pi_ref)`` to the reward so the tuned
    policy stays near the reference (the SFT model) and cannot wander off to
    exploit quirks of the reward model. Given the log-probs the policy and the
    reference assign to the *sampled* tokens, the simplest unbiased sample
    estimate of the KL is::

        KL_hat = E_{a ~ pi_theta}[ log pi_theta(a) - log pi_ref(a) ]

    i.e. the mean log-ratio over the tokens the policy actually produced. (This is
    the "k1" estimator; it can go slightly negative on a finite sample. Production
    code often uses the always-nonnegative "k3" estimator
    ``exp(d) - 1 - d`` with ``d = ref_logprobs - logprobs``; we use the plain
    log-ratio here for transparency.)

    Args:
        logprobs: ``log pi_theta(a|s)`` for the sampled tokens, shape ``(...,)``.
        ref_logprobs: ``log pi_ref(a|s)`` for the same tokens, same shape.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar (``"mean"``/``"sum"``) or per-token ``(...,)`` (``"none"``).
    """
    log_ratio = logprobs - ref_logprobs                          # (...,)
    if reduction == "none":
        return log_ratio
    if reduction == "sum":
        return log_ratio.sum()
    if reduction == "mean":
        return log_ratio.mean()
    raise ValueError(f"unknown reduction {reduction!r}")
