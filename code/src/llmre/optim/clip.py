"""Gradient clipping by global L2 norm, from scratch.

This is the clipping of lesson 03.3 (`lessons/module-03/lesson-03.md`). It bounds
the size of the *whole* gradient (all parameters concatenated into one vector),
which stops a single huge minibatch gradient from blowing up training. PyTorch
ships the same operation as ``torch.nn.utils.clip_grad_norm_``; the test in
``code/tests/test_optim.py`` checks this from-scratch version returns the same
pre-clip norm and rescales identically.
"""

from __future__ import annotations

import torch


def clip_grad_norm_(params, max_norm: float) -> float:
    """Rescale gradients in place so their global L2 norm is at most ``max_norm``.

    The *global* norm treats every parameter's gradient as part of one long
    vector: ``total = sqrt(sum over params of sum of g_i^2)``. If ``total`` is
    already ``<= max_norm`` nothing changes; otherwise every gradient is scaled
    by ``max_norm / total``, which shrinks the global norm to exactly
    ``max_norm`` while preserving its *direction*.

    Args:
        params: iterable of ``torch.Tensor`` leaves. Only those with a non-None
            ``.grad`` contribute and are scaled; any float dtype, any device.
        max_norm: the clipping threshold ``c`` (float, > 0).

    Returns:
        The **pre-clip** total L2 norm as a Python float. (Returning the norm
        before scaling matches ``torch.nn.utils.clip_grad_norm_`` and lets the
        training loop log how large the raw gradient was.)

    Note:
        The trailing underscore is the PyTorch convention for an in-place op.
        The scale is computed once from the global norm and applied to every
        gradient — this is not per-tensor clipping, which would distort the
        gradient's direction.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return 0.0
    # Global L2 norm: stack each gradient's own L2 norm into a vector, then take
    # the L2 norm of that vector. sqrt(sum_i ||g_i||^2) == sqrt(sum of all g^2).
    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), 2).to(device) for g in grads]), 2
    )
    # Scale factor, capped at 1 so we only ever shrink, never grow. A tiny eps
    # guards against divide-by-zero when the gradient is exactly zero.
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)
    return float(total_norm)
