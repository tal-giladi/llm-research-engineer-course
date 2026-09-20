"""A runnable *miniature* of the DeepSeek-R1 training pipeline (Module 17, 17.3).

The real DeepSeek-R1 recipe (DeepSeek-AI 2025, https://arxiv.org/abs/2501.12948)
is enormous: a base LLM, curated long chain-of-thought data for a **cold-start SFT**
phase, then large-scale **reinforcement learning with verifiable rewards** (GRPO with
rule-based answer/format rewards), then rejection sampling + more SFT, then a final
RL stage. We cannot run that on a laptop. What we *can* do is reproduce the
**control flow** — the sequence of stages and how each one changes the policy — on a
toy task that runs in milliseconds on CPU, so the shape of the pipeline is concrete.

The "policy" here is deliberately trivial: a table of logits, one row per problem,
one column per candidate answer. Softmax over each row is the policy's answer
distribution for that problem. This is a *categorical* policy — no transformer, no
tokens — but it exercises the exact same two learning signals R1 uses:

* **Cold-start SFT** (:func:`_sft_step`) — supervised cross-entropy on a handful of
  *curated* (problem, correct-answer) demonstrations. Maps to R1's cold-start phase:
  fine-tune on a small set of high-quality curated CoT before any RL, to give the
  policy a sane starting format/behaviour.
* **Verifiable-reward RL** (:func:`_rl_step`) — for each problem, sample a *group* of
  answers from the current policy, score each with a **verifier** (reward 1 if the
  answer is correct, else 0), turn the group's rewards into a group-relative
  advantage, and take a policy-gradient step. Maps to R1's reasoning-RL phase: GRPO
  with a *rule-based, verifiable* reward (no learned reward model).

The verifier is defined **locally** in this file (:func:`toy_verifier`) on purpose —
Module 16 owns the shared ``verifiers``/``rlvr``/``grpo`` code, and we avoid importing
it here so the two modules never race on the same files. The group-relative advantage
below is the same idea as GRPO (DeepSeekMath, Shao et al. 2024) reduced to its core.

Nothing here claims to be OpenAI's or Anthropic's pipeline; it is a didactic
reconstruction of the *publicly documented* R1 report structure.

All tensors are tiny CPU ``float32``; the whole thing runs end to end in well under a
second.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# The toy task and its verifier
# --------------------------------------------------------------------------- #
def make_toy_task(
    n_problems: int = 6,
    n_actions: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a toy "reasoning" task: for each problem, exactly one answer is correct.

    Think of each problem as a math question and each action as one of ``n_actions``
    candidate final answers; the verifier can check which one is right (a
    *verifiable* reward, like "does this equal the gold answer?").

    Args:
        n_problems: number of distinct problems ``P``.
        n_actions: number of candidate answers per problem ``A``.
        seed: RNG seed for the (fixed) correct-answer key.

    Returns:
        ``(logits, correct)``:
          * ``logits`` — the policy parameters, shape ``(P, A)``, ``float32``, CPU,
            ``requires_grad=True``. Initialised to zeros so the starting policy is
            uniform (accuracy ``1/A`` in expectation) — nothing is learned yet.
          * ``correct`` — the gold answer index per problem, shape ``(P,)``,
            ``int64``, CPU. Used only by the verifier, never fed to the policy.
    """
    g = torch.Generator().manual_seed(seed)
    correct = torch.randint(0, n_actions, (n_problems,), generator=g)  # (P,) int64
    logits = torch.zeros(n_problems, n_actions, requires_grad=True)    # (P, A) f32
    return logits, correct


def toy_verifier(actions: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """Rule-based verifiable reward: 1.0 if the answer matches the gold key, else 0.0.

    This is the miniature of R1's rule-based reward — no learned reward model, just a
    deterministic check of correctness. In real R1 this is "does the extracted final
    answer equal the reference / does the code pass the tests"; here it is an integer
    comparison.

    Args:
        actions: sampled answer indices, any shape ``S``, ``int64``, CPU.
        correct: gold answer index per problem. Must broadcast against ``actions``
            (typically shape ``(P,)`` broadcast over a group axis), ``int64``, CPU.

    Returns:
        Reward tensor, same broadcast shape as ``actions``, ``float32``, CPU, with
        entries in ``{0.0, 1.0}``.
    """
    return (actions == correct).float()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _greedy_accuracy(logits: torch.Tensor, correct: torch.Tensor) -> float:
    """Fraction of problems whose *most likely* answer is the correct one.

    Args:
        logits: policy logits, shape ``(P, A)``, ``float32``, CPU.
        correct: gold answers, shape ``(P,)``, ``int64``, CPU.

    Returns:
        A Python ``float`` in ``[0, 1]``.
    """
    preds = logits.argmax(dim=-1)          # (P,) int64 — greedy answer per problem
    return (preds == correct).float().mean().item()


@torch.no_grad()
def _expected_reward(logits: torch.Tensor, correct: torch.Tensor) -> float:
    """Mean probability the policy assigns to the correct answer (soft accuracy).

    Unlike greedy accuracy this is smooth: it rises as probability shifts onto the
    right answer even before it becomes the argmax, which makes it a good progress
    signal for the RL stage.

    Args:
        logits: policy logits, shape ``(P, A)``, ``float32``, CPU.
        correct: gold answers, shape ``(P,)``, ``int64``, CPU.

    Returns:
        A Python ``float`` in ``[0, 1]``.
    """
    probs = F.softmax(logits, dim=-1)                      # (P, A)
    p_correct = probs.gather(-1, correct.unsqueeze(-1))    # (P, 1)
    return p_correct.mean().item()


# --------------------------------------------------------------------------- #
# Stage 1 — cold-start SFT
# --------------------------------------------------------------------------- #
def _sft_step(
    logits: torch.Tensor,
    correct: torch.Tensor,
    demo_idx: torch.Tensor,
    steps: int,
    lr: float,
) -> tuple[float, float]:
    """Cold-start supervised fine-tuning on a curated subset of problems.

    Supervised cross-entropy pushing the policy toward the *known* correct answer for
    just the ``demo_idx`` problems — the analogue of R1's cold-start SFT on a small
    curated CoT set. It only touches the demonstrated rows; the rest of the policy is
    left for the RL stage to discover on its own.

    Args:
        logits: policy logits, shape ``(P, A)``, ``float32``, CPU, ``requires_grad``.
            Updated **in place** via its ``.grad``.
        correct: gold answers, shape ``(P,)``, ``int64``, CPU.
        demo_idx: indices of the curated demonstration problems, shape ``(D,)``,
            ``int64``, CPU.
        steps: number of gradient steps.
        lr: SGD learning rate.

    Returns:
        ``(loss_before, loss_after)`` cross-entropy on the demo set, Python floats.
    """
    def demo_loss() -> torch.Tensor:
        # Cross-entropy of the demo rows against their gold answers. (D, A) vs (D,).
        return F.cross_entropy(logits[demo_idx], correct[demo_idx])

    loss_before = demo_loss().item()
    for _ in range(steps):
        loss = demo_loss()
        if logits.grad is not None:
            logits.grad.zero_()
        loss.backward()
        with torch.no_grad():
            # Plain SGD on the logit table. logits.grad is (P, A); only demo rows are
            # non-zero because the loss only reads those rows.
            logits -= lr * logits.grad
    loss_after = demo_loss().item()
    return loss_before, loss_after


# --------------------------------------------------------------------------- #
# Stage 2 — reinforcement learning with a verifiable reward (GRPO-style)
# --------------------------------------------------------------------------- #
def _rl_step(
    logits: torch.Tensor,
    correct: torch.Tensor,
    verifier: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    iters: int,
    group_size: int,
    lr: float,
    seed: int,
) -> None:
    """Improve the policy with a verifiable reward, GRPO-style (in place).

    For every problem we sample a **group** of ``group_size`` answers from the current
    policy, score each with ``verifier`` (1 correct / 0 wrong), and standardise the
    rewards *within the group* to an advantage
    ``A = (r - mean(r)) / (std(r) + eps)``. The loss is the REINFORCE/GRPO term
    ``-(A * logprob(action))`` averaged over the group and problems. Maximising reward
    = minimising this loss.

    Why group-relative? It is exactly GRPO's trick (DeepSeekMath): the group mean is a
    baseline that needs **no learned value network**. A group that is all-correct or
    all-wrong has zero advantage and produces no update — so once a problem is solved
    the policy stops being pushed, which keeps greedy accuracy from regressing.

    Args:
        logits: policy logits, shape ``(P, A)``, ``float32``, CPU, ``requires_grad``.
            Updated **in place**.
        correct: gold answers, shape ``(P,)``, ``int64``, CPU.
        verifier: maps ``(actions, correct) -> reward`` in ``{0,1}`` (see
            :func:`toy_verifier`).
        iters: number of RL iterations.
        group_size: samples drawn per problem per iteration (``G``).
        lr: SGD learning rate.
        seed: RNG seed for reproducible sampling.
    """
    P, _A = logits.shape
    g = torch.Generator().manual_seed(seed)
    eps = 1e-8

    for _ in range(iters):
        probs = F.softmax(logits, dim=-1)                  # (P, A)
        # Sample a group of G answers per problem: (P, G) int64.
        actions = torch.multinomial(probs, group_size, replacement=True, generator=g)

        # Verifiable reward for every sampled answer: (P, G) in {0,1}.
        rewards = verifier(actions, correct.unsqueeze(-1))

        # Group-relative advantage (the GRPO baseline): standardise within each row.
        mean = rewards.mean(dim=-1, keepdim=True)          # (P, 1)
        std = rewards.std(dim=-1, keepdim=True)            # (P, 1)
        advantage = (rewards - mean) / (std + eps)         # (P, G)

        # log pi(action) for each sampled action, differentiable in logits.
        logp_all = F.log_softmax(logits, dim=-1)           # (P, A)
        logp = logp_all.gather(-1, actions)                # (P, G)

        # Policy-gradient loss: maximise advantage-weighted log-prob.
        loss = -(advantage.detach() * logp).mean()

        if logits.grad is not None:
            logits.grad.zero_()
        loss.backward()
        with torch.no_grad():
            logits -= lr * logits.grad

    # Detach any stray graph state; logits keeps requires_grad for later stages.
    logits.grad = None


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
def mini_r1(
    n_problems: int = 6,
    n_actions: int = 4,
    n_demos: int = 2,
    sft_steps: int = 50,
    sft_lr: float = 0.5,
    rl_iters: int = 200,
    rl_group_size: int = 8,
    rl_lr: float = 0.3,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Run the miniature R1 pipeline end to end and report per-stage metrics.

    Stages, in R1 order:

    1. **base**   — the untrained uniform policy (a reference point).
    2. **sft**    — cold-start SFT on ``n_demos`` curated problems (:func:`_sft_step`).
       Maps to R1's cold-start supervised phase.
    3. **rl**     — verifiable-reward RL (GRPO-style) over *all* problems
       (:func:`_rl_step`). Maps to R1's reasoning-RL phase and lifts accuracy on the
       problems SFT never saw.

    Everything is deterministic given ``seed``.

    Args:
        n_problems: number of problems ``P``.
        n_actions: candidate answers per problem ``A``.
        n_demos: how many problems get curated cold-start demonstrations
            (``0 <= n_demos <= P``).
        sft_steps, sft_lr: cold-start SFT budget and learning rate.
        rl_iters, rl_group_size, rl_lr: RL budget, group size ``G``, learning rate.
        seed: master RNG seed.

    Returns:
        A dict keyed by stage name, each a dict of Python-float metrics::

            {
              "base": {"accuracy": .., "expected_reward": ..},
              "sft":  {"accuracy": .., "expected_reward": ..,
                       "demo_loss_before": .., "demo_loss_after": ..},
              "rl":   {"accuracy_before": .., "accuracy_after": ..,
                       "expected_reward_before": .., "expected_reward_after": ..},
            }

        ``rl["accuracy_after"] >= rl["accuracy_before"]`` by construction (see
        :func:`_rl_step`): the group-relative update never pushes a solved problem
        back toward a wrong answer.
    """
    if not 0 <= n_demos <= n_problems:
        raise ValueError(f"n_demos must be in [0, {n_problems}], got {n_demos}")

    logits, correct = make_toy_task(n_problems, n_actions, seed=seed)
    # Curated demo set = the first n_demos problems (arbitrary but fixed).
    demo_idx = torch.arange(n_demos, dtype=torch.long)

    metrics: dict[str, dict[str, float]] = {}

    # --- Stage: base (uniform policy) -------------------------------------- #
    metrics["base"] = {
        "accuracy": _greedy_accuracy(logits, correct),
        "expected_reward": _expected_reward(logits, correct),
    }

    # --- Stage: cold-start SFT --------------------------------------------- #
    if n_demos > 0:
        loss_before, loss_after = _sft_step(logits, correct, demo_idx, sft_steps, sft_lr)
    else:
        loss_before = loss_after = float("nan")
    metrics["sft"] = {
        "accuracy": _greedy_accuracy(logits, correct),
        "expected_reward": _expected_reward(logits, correct),
        "demo_loss_before": loss_before,
        "demo_loss_after": loss_after,
    }

    # --- Stage: verifiable-reward RL --------------------------------------- #
    acc_before = _greedy_accuracy(logits, correct)
    er_before = _expected_reward(logits, correct)
    _rl_step(
        logits, correct, toy_verifier,
        iters=rl_iters, group_size=rl_group_size, lr=rl_lr, seed=seed + 1,
    )
    metrics["rl"] = {
        "accuracy_before": acc_before,
        "accuracy_after": _greedy_accuracy(logits, correct),
        "expected_reward_before": er_before,
        "expected_reward_after": _expected_reward(logits, correct),
    }

    return metrics


if __name__ == "__main__":  # pragma: no cover
    from pprint import pprint

    pprint(mini_r1())
