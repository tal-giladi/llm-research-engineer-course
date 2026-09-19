"""Direct Preference Optimization (DPO) from scratch.

This is Module 15, lesson 15.3. RLHF (lesson 15.2) optimizes a policy against a
learned reward model with PPO — a reward model, a value model, sampling from the
policy, and an RL loop. **DPO** (Rafailov et al. 2023,
https://arxiv.org/abs/2305.18290) shows you can skip all of that.

Start from the KL-constrained RLHF objective. Its closed-form optimal policy is::

    pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp( r(x,y) / beta )

Invert it to write the reward in terms of the optimal policy::

    r(x,y) = beta * log( pi*(y|x) / pi_ref(y|x) ) + beta*log Z(x)

Substitute that into the Bradley-Terry preference model (lesson 15.1). The
prompt-only ``beta*log Z(x)`` term is identical for ``y_w`` and ``y_l`` and so
**cancels in the difference**, leaving a loss that contains no reward model and
no partition function — only the policy ``pi_theta``, a frozen reference
``pi_ref``, and preference pairs::

    L = -log sigmoid( beta * ( log pi_theta(y_w|x)/pi_ref(y_w|x)
                               - log pi_theta(y_l|x)/pi_ref(y_l|x) ) )

DPO needs **no reward model** and **no sampling**: just forward the policy and the
frozen reference on the chosen and rejected completions, gather the sequence
log-probabilities, and plug them in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def sequence_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Sum of log-probabilities the model assigns to a label sequence.

    ``log pi(y|x) = sum_t log pi(y_t | y_<t, x)``. For each position we take the
    log-softmax over the vocabulary and gather the log-prob of the label token,
    then **sum** over the sequence (positions whose label is ``ignore_index`` —
    prompt tokens or padding — contribute 0).

    Positions are assumed already aligned: ``logits[..., t, :]`` scores
    ``labels[..., t]``. In a real pipeline you shift so that position ``t``
    predicts token ``t+1`` and mask the prompt to ``ignore_index`` before calling
    this (same convention as :mod:`llmre.sft.masking`).

    Args:
        logits: shape ``(..., T, V)`` — e.g. ``(T, V)`` or ``(B, T, V)`` — float,
            any device. ``V`` = vocabulary size.
        labels: shape ``(..., T)`` matching ``logits`` without its last axis,
            dtype ``torch.long``. Entries equal to ``ignore_index`` are skipped.
        ignore_index: sentinel marking positions to skip (default ``-100``).

    Returns:
        Shape ``(...)`` (``logits`` without its last two axes): a scalar for
        ``(T, V)`` input, ``(B,)`` for ``(B, T, V)`` input. Same dtype/device as
        ``logits``.
    """
    logp = F.log_softmax(logits, dim=-1)                  # (..., T, V)
    mask = labels != ignore_index                         # (..., T) bool
    safe = labels.clamp(min=0)                            # avoid gather on -100
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)  # (..., T)
    tok_logp = tok_logp * mask                            # zero out ignored positions
    return tok_logp.sum(dim=-1)                           # (...,)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Direct Preference Optimization loss.

    Given the *sequence* log-probs (see :func:`sequence_logprob`) of the chosen
    and rejected completions under both the trainable policy and the frozen
    reference::

        pi_logratios  = policy_chosen_logps  - policy_rejected_logps
        ref_logratios = ref_chosen_logps     - ref_rejected_logps
        L = -log sigmoid( beta * (pi_logratios - ref_logratios) )

    The bracket is the DPO **implicit reward margin**: how much more the policy
    prefers the chosen over the rejected completion, *relative to the reference*.
    ``beta`` controls how far the policy may move from the reference (larger
    ``beta`` = stay closer). Minimizing the loss increases this margin.

    Args:
        policy_chosen_logps: ``log pi_theta(y_w|x)``, shape ``(B,)`` float.
        policy_rejected_logps: ``log pi_theta(y_l|x)``, shape ``(B,)`` float.
        ref_chosen_logps: ``log pi_ref(y_w|x)``, shape ``(B,)`` float (no grad;
            the reference is frozen).
        ref_rejected_logps: ``log pi_ref(y_l|x)``, shape ``(B,)`` float.
        beta: KL/temperature coefficient, typically ``0.1``.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"`` (return the
            per-pair ``(B,)`` loss).

    Returns:
        Scalar loss (``"mean"``/``"sum"``) or ``(B,)`` (``"none"``), same
        dtype/device as the policy inputs.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps      # (B,)
    ref_logratios = ref_chosen_logps - ref_rejected_logps          # (B,)
    logits = pi_logratios - ref_logratios                          # (B,) implicit margin
    per_pair = -F.logsigmoid(beta * logits)                        # (B,)
    if reduction == "none":
        return per_pair
    if reduction == "sum":
        return per_pair.sum()
    if reduction == "mean":
        return per_pair.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def dpo_reward_margin(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """The DPO **implicit rewards** ``(r_chosen, r_rejected)``, each ``(B,)``.

    DPO never trains an explicit reward model, but the derivation says the policy
    *is* one: ``r(x,y) = beta * log(pi_theta(y|x)/pi_ref(y|x))`` (up to the
    prompt constant). These implicit rewards are the standard quantity to log
    while training — "reward accuracy" is the fraction of pairs with
    ``r_chosen > r_rejected``.

    Returns:
        ``(chosen_rewards, rejected_rewards)``, both ``(B,)`` float, detached.
    """
    chosen = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected = beta * (policy_rejected_logps - ref_rejected_logps)
    return chosen.detach(), rejected.detach()
