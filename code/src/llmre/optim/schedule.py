"""Learning-rate schedule: linear warmup then cosine decay to a floor.

This is the schedule of lesson 03.3 (`lessons/module-03/lesson-03.md`) and the
one used by essentially every modern LLM pretraining run (GPT-3, LLaMA, OLMo,
...). It is a pure function of the step index, so the training loop just calls
it each step and writes the result into the optimizer's ``lr``.

Three phases:
    step < warmup_steps      : linear ramp   0 -> max_lr
    warmup_steps..max_steps  : cosine decay  max_lr -> min_lr
    step >= max_steps        : clamped flat  min_lr
"""

from __future__ import annotations

import math


def cosine_warmup_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Learning rate at a given step under linear-warmup + cosine-decay.

    Args:
        step: current optimizer step (int, ``0`` at the first update).
        warmup_steps: number of steps ``W`` of linear warmup. During warmup the
            lr is ``max_lr * (step + 1) / warmup_steps`` so that it reaches
            exactly ``max_lr`` at ``step == warmup_steps``.
        max_steps: total steps ``T`` over which the schedule runs; cosine decay
            spans ``[warmup_steps, max_steps]``.
        max_lr: peak learning rate (float), reached at the end of warmup.
        min_lr: floor learning rate (float), reached at ``step == max_steps`` and
            held for any later step.

    Returns:
        The scalar learning rate (float) for this step.

    Note:
        This is a plain Python float function — no tensors, no device. PyTorch's
        ``torch.optim.lr_scheduler`` classes wrap the same arithmetic in stateful
        objects; here we keep it a transparent function of ``step``.
    """
    # Phase 1: linear warmup. (step + 1) so the very first step is non-zero and
    # step == warmup_steps - 1 is the last warmup step below the peak; at
    # step == warmup_steps we fall through to cosine with progress 0 -> max_lr.
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # Phase 3: past the horizon, clamp to the floor.
    if step >= max_steps:
        return min_lr
    # Phase 2: cosine decay from max_lr (progress 0) down to min_lr (progress 1).
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0 as progress 0 -> 1
    return min_lr + coeff * (max_lr - min_lr)
