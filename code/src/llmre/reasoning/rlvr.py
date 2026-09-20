"""A tiny, CPU-runnable RLVR loop that makes a toy policy reason correctly.

This is Module 16, lesson 16.2. It wires together the two ingredients:

* the **verifier** (:mod:`llmre.reasoning.verifiers`) — an automatic reward, 1
  for a correct answer and 0 otherwise; and
* **GRPO** (:mod:`llmre.rl.grpo`) — sample a group of answers per prompt,
  standardize the group's rewards into advantages, take a clipped policy-gradient
  step. No critic, no learned reward model.

To keep it CPU-instant we do **not** use a full GPT. The "policy" is a table of
logits, one row per prompt, over a small discrete answer space — a categorical
policy ``pi(answer | prompt)``. That is enough to demonstrate the whole RLVR
mechanism and, crucially, to *watch the correct-rate go up*: after training the
policy samples the verifier-accepted answer far more often than at the start.
Swapping this categorical policy for a GPT that emits answer *tokens* changes the
engineering, not the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from llmre.reasoning.verifiers import arithmetic_verifier
from llmre.rl.grpo import grpo_advantages, grpo_objective


@dataclass
class RLVRResult:
    """Outcome of a :func:`run_rlvr` run.

    Attributes:
        initial_correct_rate: fraction of sampled answers that were correct in
            the first training step (float in ``[0, 1]``).
        final_correct_rate: same fraction measured after the last step.
        history: the per-step correct-rate, a list of floats (length ``steps``).
    """

    initial_correct_rate: float
    final_correct_rate: float
    history: list = field(default_factory=list)


def _sample_group(logits_row: torch.Tensor, group_size: int) -> torch.Tensor:
    """Sample ``group_size`` answer indices from one prompt's categorical policy.

    Args:
        logits_row: shape ``(num_answers,)`` float — the policy logits for one
            prompt.
        group_size: number of samples ``G`` to draw (with replacement).

    Returns:
        Long tensor of shape ``(group_size,)`` of sampled answer indices.
    """
    probs = F.softmax(logits_row, dim=-1)                 # (num_answers,)
    return torch.multinomial(probs, group_size, replacement=True)   # (G,)


def run_rlvr(
    problems=("3 + 4", "5 + 2", "6 + 1", "2 + 3"),
    answer_space=tuple(range(10)),
    group_size: int = 8,
    steps: int = 150,
    lr: float = 0.2,
    clip_eps: float = 0.2,
    seed: int = 0,
    verifier=arithmetic_verifier,
) -> RLVRResult:
    """Train a toy categorical policy with GRPO + a verifiable reward.

    The policy is a learnable table ``logits`` of shape
    ``(num_prompts, num_answers)``: ``logits[p]`` is the distribution over the
    answer space for problem ``p``. Each step, for every prompt:

    1. **sample** a group of ``group_size`` answers from ``softmax(logits[p])``;
    2. **verify** each sampled answer, getting a reward in ``{0, 1}``;
    3. **advantage** — standardize the group's rewards with
       :func:`grpo_advantages` (the group mean is the baseline);
    4. **update** — ascend :func:`grpo_objective` (on-policy, so the ratio starts
       at 1 and this is the group-baselined policy gradient).

    The correct-rate — the fraction of sampled answers the verifier accepts —
    should climb from chance (~``1/num_answers``) toward 1.

    Args:
        problems: iterable of problem strings the verifier understands (default:
            single-digit additions the default answer space can represent).
        answer_space: the discrete set of candidate answers (indexable; the
            policy's columns correspond to these values in order).
        group_size: completions sampled per prompt per step (``G`` in GRPO).
        steps: number of gradient steps.
        lr: learning rate for the Adam optimizer over the logit table.
        clip_eps: PPO/GRPO clip half-width.
        seed: RNG seed for reproducibility.
        verifier: a ``(problem, answer) -> float`` reward in ``{0.0, 1.0}``.

    Returns:
        An :class:`RLVRResult` with the initial and final correct-rates and the
        full per-step history.

    Shapes/dtype/device:
        ``logits`` is ``(num_prompts, num_answers)`` float32 on CPU; all tensors
        stay on CPU. This is a toy — see the lesson's hardware note for what real
        reasoning RL costs.
    """
    torch.manual_seed(seed)
    problems = list(problems)
    answers = list(answer_space)
    num_prompts, num_answers = len(problems), len(answers)

    # The policy: one categorical distribution per prompt, initialized uniform-ish.
    logits = torch.zeros(num_prompts, num_answers, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)

    history: list = []
    for _ in range(steps):
        # 1) Sample a group of answers for each prompt.
        sampled = torch.stack(
            [_sample_group(logits[p], group_size) for p in range(num_prompts)]
        )  # (num_prompts, group_size) long, indices into `answers`

        # 2) Verify: reward 1.0 if the sampled answer is correct, else 0.0.
        rewards = torch.zeros(num_prompts, group_size)
        for p in range(num_prompts):
            for g in range(group_size):
                ans_value = answers[sampled[p, g].item()]
                rewards[p, g] = verifier(problems[p], ans_value)

        history.append(rewards.mean().item())  # fraction correct this step

        # 3) Group-relative advantages (no critic).
        advantages = grpo_advantages(rewards)                 # (num_prompts, G)

        # 4) Policy-gradient step. log pi(sampled answer | prompt) under current
        #    policy; old_logprobs detached => on-policy ratio 1 on this batch.
        logp_all = F.log_softmax(logits, dim=-1)              # (num_prompts, num_answers)
        logp = torch.gather(logp_all, 1, sampled)            # (num_prompts, G)
        obj = grpo_objective(
            logp, logp.detach(), advantages, clip_eps=clip_eps
        )
        loss = -obj  # optimizer minimizes; we want to maximize the objective

        opt.zero_grad()
        loss.backward()
        opt.step()

    return RLVRResult(
        initial_correct_rate=history[0],
        final_correct_rate=history[-1],
        history=history,
    )
