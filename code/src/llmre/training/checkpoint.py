"""Saving and restoring a training run so it resumes *exactly*.

This is the checkpointing of lesson 07.3 (`lessons/module-07/lesson-03.md`). A
resume is only correct if it restores everything the next step depends on, not
just the weights:

* **model** — ``model.state_dict()`` (every parameter and buffer).
* **optimizer** — the from-scratch :class:`llmre.optim.adamw.AdamW` carries the
  step counter ``t`` and the first/second moment buffers ``m``/``v``; drop those
  and Adam restarts with cold, biased moments and diverges from the un-resumed
  run.
* **step** — where in the schedule we are (so the LR schedule continues).
* **RNG states** — torch, NumPy and Python ``random``; the data loader and
  dropout draw from these, so restoring them makes the *same* future batches and
  dropout masks appear.
* **extra** — any user payload (the run config, best-so-far loss, ...).

We keep everything in one dict and hand it to ``torch.save``. Because our
``AdamW`` is a plain object (not a ``torch.optim.Optimizer``), we serialize its
state field by field here rather than calling a ``.state_dict()`` it does not
have.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def _optimizer_state(optimizer) -> dict[str, Any]:
    """Pull the resumable state out of an :class:`llmre.optim.adamw.AdamW`.

    Saves the step counter and the per-parameter moment buffers (as CPU clones so
    the file is device-independent) plus the scalar hyperparameters, keyed
    positionally to match ``optimizer.params`` on reload.
    """
    return {
        "t": optimizer.t,
        "m": [buf.detach().cpu().clone() for buf in optimizer.m],
        "v": [buf.detach().cpu().clone() for buf in optimizer.v],
        "lr": optimizer.lr,
        "betas": (optimizer.beta1, optimizer.beta2),
        "eps": optimizer.eps,
        "weight_decay": optimizer.weight_decay,
    }


def _load_optimizer_state(optimizer, state: dict[str, Any]) -> None:
    """Restore :class:`AdamW` state in place, moving buffers onto each param.

    The moment buffers are copied back onto the device/dtype of the matching
    live parameter, so a checkpoint saved on CPU resumes correctly on GPU.
    """
    optimizer.t = state["t"]
    optimizer.lr = state["lr"]
    optimizer.beta1, optimizer.beta2 = state["betas"]
    optimizer.eps = state["eps"]
    optimizer.weight_decay = state["weight_decay"]
    optimizer.m = [
        buf.to(device=p.device, dtype=p.dtype)
        for buf, p in zip(state["m"], optimizer.params)
    ]
    optimizer.v = [
        buf.to(device=p.device, dtype=p.dtype)
        for buf, p in zip(state["v"], optimizer.params)
    ]


def save_checkpoint(
    path: str,
    model,
    optimizer,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a full-resume checkpoint to ``path``.

    Captures the model ``state_dict``, the optimizer state, the current ``step``,
    the current torch/NumPy/Python RNG states, and any ``extra`` payload, all in
    one ``torch.save`` blob.

    Args:
        path: destination file path.
        model: an ``nn.Module``; ``model.state_dict()`` is saved.
        optimizer: an :class:`llmre.optim.adamw.AdamW` instance.
        step: the completed optimizer-step count to resume from.
        extra: optional JSON-ish dict of run metadata (config, metrics, ...).

    Returns:
        None. The file at ``path`` is overwritten.
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": _optimizer_state(optimizer),
        "step": step,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    restore_rng: bool = True,
    map_location: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Restore a checkpoint written by :func:`save_checkpoint`.

    Loads the model weights (always), the optimizer state (if ``optimizer`` is
    given), and the RNG states (if ``restore_rng``), so training continues from
    the exact point it was saved.

    Args:
        path: checkpoint file to read.
        model: the ``nn.Module`` to load weights into (in place).
        optimizer: optional :class:`AdamW` to restore; skip for inference-only
            loads.
        restore_rng: if ``True`` (default), restore torch/NumPy/Python RNG so the
            future batch and dropout stream matches the original run.
        map_location: passed to ``torch.load`` to place tensors on a device
            (e.g. ``"cpu"``).

    Returns:
        ``(step, extra)`` — the saved optimizer-step count and the saved ``extra``
        dict, so the caller can resume its loop at ``step`` with its metadata.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        _load_optimizer_state(optimizer, ckpt["optimizer"])
    if restore_rng:
        rng = ckpt["rng"]
        # torch RNG state is a ByteTensor; ensure it lives on CPU before setting.
        torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return ckpt["step"], ckpt["extra"]
