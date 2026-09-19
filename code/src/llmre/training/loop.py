"""The pretraining loop: the six lines that train every GPT, at full scale.

This is the payoff of Module 7 (`lessons/module-07/lesson-01.md`). Everything
built so far — the model (Module 6), AdamW (Module 3), the LR schedule and grad
clipping (Module 3), the data loader (Module 4) — meets here in one function.
The canonical inner step is:

    x, y = get_batch("train")          # data loader (Module 4)
    logits, loss = model(x, y)         # forward + cross-entropy (Module 6)
    (loss / grad_accum).backward()     # autograd (Module 2)
    clip_grad_norm_(params, grad_clip) # stabilize (Module 3)
    optimizer.step()                   # AdamW update (Module 3)
    optimizer.zero_grad()              # reset grads for the next step

with the learning rate written from ``cosine_warmup_lr`` before each step and
gradient accumulation wrapping the forward/backward so a large effective batch
fits in small memory. The loop is intentionally readable, not clever: this is
the reference you re-read whenever a fancier trainer confuses you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from llmre.optim.adamw import AdamW
from llmre.optim.clip import clip_grad_norm_
from llmre.optim.schedule import cosine_warmup_lr


@dataclass
class TrainConfig:
    """Hyperparameters for :func:`train`.

    Attributes:
        max_steps: number of optimizer steps ``T`` to run.
        micro_batch_size: sequences per micro-batch ``b`` (one forward/backward).
        grad_accum_steps: micro-batches ``G`` summed before each optimizer step.
        warmup_steps: linear-warmup steps for the LR schedule.
        max_lr: peak learning rate (end of warmup).
        min_lr: floor learning rate (end of cosine decay).
        weight_decay: decoupled AdamW weight decay.
        betas: AdamW ``(beta1, beta2)``.
        eps: AdamW epsilon.
        grad_clip: global-L2-norm clip threshold; ``0`` or ``None`` disables it.
        eval_interval: run an eval pass every this many steps (``0`` disables).
        eval_iters: batches averaged per eval pass.
        seed: torch RNG seed set at the start of the run (``None`` leaves it).
        device: device string for batches/model (e.g. ``"cpu"``, ``"cuda"``).
        log_interval: record a loss-history entry every this many steps.
    """

    max_steps: int = 100
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    warmup_steps: int = 10
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float | None = 1.0
    eval_interval: int = 0
    eval_iters: int = 5
    seed: int | None = None
    device: str = "cpu"
    log_interval: int = 1


def evaluate(model, get_batch, split: str, iters: int) -> float:
    """Mean loss over ``iters`` batches of ``split``, with grads off.

    Args:
        model: the GPT (put in ``eval()`` mode for the duration, restored after).
        get_batch: callable ``get_batch(split) -> (x, y)`` of ``(B, T)`` longs.
        split: which split to draw, e.g. ``"val"``.
        iters: number of batches to average.

    Returns:
        The mean scalar cross-entropy over the sampled batches, as a ``float``.
    """
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def train(model, get_batch, cfg: TrainConfig) -> dict:
    """Train ``model`` for ``cfg.max_steps`` optimizer steps.

    Args:
        model: a :class:`llmre.model.gpt.GPT` (or any module whose
            ``forward(x, y)`` returns ``(logits, loss)``), already on
            ``cfg.device``.
        get_batch: callable ``get_batch(split) -> (x, y)`` returning a batch of
            ``(micro_batch_size, block_size)`` int64 tensors on ``cfg.device``.
            ``split`` is ``"train"`` during optimization and ``"val"`` during
            eval; a loader that ignores the argument is fine.
        cfg: a :class:`TrainConfig`.

    Returns:
        A history dict with keys:
            ``"step"``   list of logged step indices,
            ``"loss"``   training loss at those steps (the accumulated
                         micro-batch mean for the step),
            ``"lr"``     learning rate used at those steps,
            ``"grad_norm"`` pre-clip global grad norm at those steps,
            ``"eval_step"`` / ``"eval_loss"`` the eval pass step indices and
                         mean val losses (empty if ``eval_interval == 0``).

    The step, in order: set the LR from the cosine-warmup schedule; for each of
    ``G`` micro-batches run forward/backward on ``loss / G`` (so the summed
    gradient equals the gradient of the mean loss over the full ``b*G`` batch);
    clip the global gradient norm; take the AdamW step; zero grads.
    """
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        params,
        lr=cfg.max_lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    history: dict = {
        "step": [],
        "loss": [],
        "lr": [],
        "grad_norm": [],
        "eval_step": [],
        "eval_loss": [],
    }

    model.train()
    for step in range(cfg.max_steps):
        # 1) Learning rate for this step, written straight into the optimizer.
        lr = cosine_warmup_lr(
            step, cfg.warmup_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr
        )
        optimizer.lr = lr

        # 2) Gradient accumulation: sum grads over G micro-batches. Scaling each
        #    micro loss by 1/G makes the summed gradient equal the gradient of
        #    the mean loss over the full b*G batch (means average, so 1/G * sum
        #    of G equal-sized means == mean over the union).
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch("train")
            _, loss = model(x, y)
            step_loss += loss.item() / cfg.grad_accum_steps
            (loss / cfg.grad_accum_steps).backward()

        # 3) Clip the global gradient norm (records the pre-clip norm).
        if cfg.grad_clip:
            grad_norm = clip_grad_norm_(params, cfg.grad_clip)
        else:
            grad_norm = float(
                torch.norm(
                    torch.stack(
                        [torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]
                    ),
                    2,
                )
            )

        # 4) One AdamW update, then clear grads for the next step.
        optimizer.step()

        if cfg.log_interval and (step % cfg.log_interval == 0 or step == cfg.max_steps - 1):
            history["step"].append(step)
            history["loss"].append(step_loss)
            history["lr"].append(lr)
            history["grad_norm"].append(grad_norm)

        if cfg.eval_interval and (
            (step + 1) % cfg.eval_interval == 0 or step == cfg.max_steps - 1
        ):
            val = evaluate(model, get_batch, "val", cfg.eval_iters)
            history["eval_step"].append(step)
            history["eval_loss"].append(val)

    return history
