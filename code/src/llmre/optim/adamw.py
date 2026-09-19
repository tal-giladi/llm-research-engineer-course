"""AdamW from scratch — decoupled weight decay + bias-corrected Adam.

This is the optimizer of lesson 03.2 (`lessons/module-03/lesson-02.md`) and the
default optimizer for the rest of the course (it is what trains the GPT in
Module 7). It reproduces ``torch.optim.AdamW`` numerically for a fixed
configuration, which the test in ``code/tests/test_optim.py`` checks to 1e-5
over several steps.

The per-parameter update, in the order ``torch.optim.AdamW`` applies it:

    p  <-  p * (1 - lr * weight_decay)          # decoupled weight decay
    m  <-  b1 * m + (1 - b1) * g                # 1st moment (EMA of g)
    v  <-  b2 * v + (1 - b2) * g^2              # 2nd moment (EMA of g^2)
    m_hat = m / (1 - b1^t)                      # bias correction
    v_hat = v / (1 - b2^t)
    p  <-  p - lr * m_hat / (sqrt(v_hat) + eps)

The weight decay is *decoupled*: it acts directly on the parameter and never
enters ``g``, so the moment estimates ``m`` and ``v`` are computed from the pure
task gradient. That is the one change AdamW makes over Adam-with-L2.
"""

from __future__ import annotations

import torch


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Args:
        params: iterable of ``torch.Tensor`` leaves (``requires_grad=True``),
            any float dtype, any device. Stored as a list, order preserved.
        lr: learning rate ``eta`` (float, > 0).
        betas: ``(beta1, beta2)`` EMA decay rates for the first and second
            moments. Defaults ``(0.9, 0.999)``.
        eps: ``epsilon`` added to the denominator for numerical stability
            (float, default ``1e-8``).
        weight_decay: decoupled decay coefficient ``lambda`` (float, default
            ``0.01``). Applied as ``p *= (1 - lr * weight_decay)`` each step.

    State per parameter (all same shape/dtype/device as the parameter):
        ``m`` first-moment buffer, ``v`` second-moment buffer, both initialized
        to zeros; plus a scalar step counter ``t`` shared across parameters.
    """

    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # global step counter; drives bias correction
        # Zero-initialized moment estimates, one pair per parameter.
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        """Apply one AdamW update to every parameter, in place.

        Increments the shared step counter first (so bias correction uses
        ``t = 1`` on the very first call), then, for each parameter, applies
        decoupled decay, updates the two moment EMAs from ``p.grad``, corrects
        their bias, and takes the scaled step. Call after ``loss.backward()``.
        """
        self.t += 1
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2 = 1.0 - self.beta2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            # 1) Decoupled weight decay: shrink the parameter itself. This does
            #    NOT touch g, which is the whole point of AdamW vs Adam+L2.
            if self.weight_decay != 0.0:
                p.mul_(1.0 - self.lr * self.weight_decay)
            # 2) Update biased first and second moment estimates.
            m, v = self.m[i], self.v[i]
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)          # m <- b1*m + (1-b1)*g
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)   # v <- b2*v + (1-b2)*g^2
            # 3) Bias-corrected step. denom = sqrt(v)/sqrt(bc2) + eps, matching
            #    torch, so step = (lr/bc1) * m / denom = lr * m_hat/(sqrt(v_hat)+eps).
            denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(self.eps)
            step_size = self.lr / bias_correction1
            p.addcdiv_(m, denom, value=-step_size)

    @torch.no_grad()
    def zero_grad(self) -> None:
        """Reset every parameter's ``.grad`` to ``None`` (see ``SGD.zero_grad``)."""
        for p in self.params:
            p.grad = None
