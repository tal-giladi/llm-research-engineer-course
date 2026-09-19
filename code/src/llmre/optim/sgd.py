"""Stochastic gradient descent with (heavy-ball) momentum, from scratch.

This is the optimizer of lesson 03.1 (`lessons/module-03/lesson-01.md`). It is
deliberately written to reproduce ``torch.optim.SGD`` *numerically* for the
default configuration (``dampening=0``, ``nesterov=False``, ``weight_decay=0``),
so the test in ``code/tests/test_optim.py`` can assert the two agree
step-for-step. Nothing here is magic: the update is exactly

    v  <-  mu * v + g            (momentum buffer; on step 1, v = g)
    p  <-  p - lr * v

applied to every parameter tensor independently, in place.
"""

from __future__ import annotations

import torch


class SGD:
    """Minibatch gradient descent with a momentum buffer.

    The optimizer owns a list of parameter tensors (each a leaf tensor whose
    ``.grad`` autograd fills in during ``loss.backward()``) and mutates them in
    place. It keeps one *velocity* buffer per parameter, the running blend of
    past gradients that momentum accumulates.

    Args:
        params: iterable of ``torch.Tensor`` leaves (``requires_grad=True``),
            any float dtype, any device. Stored as a list, order preserved.
        lr: learning rate ``eta`` (float, > 0). Scales the step taken each call.
        momentum: coefficient ``mu`` in ``[0, 1)``. ``0`` recovers plain SGD;
            ``0.9`` is the usual heavy-ball value.

    Shapes/dtype/device: every velocity buffer has the same shape, dtype, and
    device as its parameter. No tensor changes shape; updates are elementwise.
    """

    def __init__(self, params, lr: float, momentum: float = 0.0) -> None:
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        # One velocity buffer per parameter, created lazily on the first step so
        # that on step 1 the buffer equals the gradient (matching torch.optim).
        self.velocities: list[torch.Tensor | None] = [None] * len(self.params)

    @torch.no_grad()
    def step(self) -> None:
        """Apply one update to every parameter, in place.

        Reads ``p.grad`` for each parameter, folds it into that parameter's
        velocity buffer, and subtracts ``lr * velocity`` from the parameter.
        Must be called after ``loss.backward()`` has populated the gradients.
        """
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.momentum != 0.0:
                buf = self.velocities[i]
                if buf is None:
                    # First step: initialize the buffer to the raw gradient,
                    # exactly as torch.optim.SGD does (no (1 - mu) scaling here).
                    buf = g.clone()
                    self.velocities[i] = buf
                else:
                    # v <- mu * v + g
                    buf.mul_(self.momentum).add_(g)
                update = buf
            else:
                update = g
            # p <- p - lr * update
            p.add_(update, alpha=-self.lr)

    @torch.no_grad()
    def zero_grad(self) -> None:
        """Reset every parameter's ``.grad`` to ``None``.

        Gradients accumulate across ``backward()`` calls, so they must be
        cleared between optimizer steps or each step would see the sum of all
        past minibatch gradients. Setting to ``None`` (rather than zeroing) frees
        the buffers and matches ``torch``'s default.
        """
        for p in self.params:
            p.grad = None
