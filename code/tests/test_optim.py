"""Tests for llmre.optim — from-scratch optimizers vs torch.optim.

Every from-scratch optimizer in ``llmre.optim`` is validated against the
corresponding ``torch.optim`` (or ``torch.nn.utils``) reference on a small toy
problem, so the lessons in Module 3 can claim numerical equivalence honestly.
"""

from __future__ import annotations

import torch

from llmre.optim.adamw import AdamW
from llmre.optim.clip import clip_grad_norm_
from llmre.optim.schedule import cosine_warmup_lr
from llmre.optim.sgd import SGD


def _toy_regression(seed: int = 0):
    """Return (X, y, w_init) for a tiny linear-regression problem.

    X: (16, 4) float64 features; y: (16,) float64 targets; w_init: (4,) float64
    initial weights. float64 keeps the two implementations comparable to ~1e-12
    of floating error so the 1e-5 assertions are about algorithm, not precision.
    """
    torch.manual_seed(seed)
    X = torch.randn(16, 4, dtype=torch.float64)
    true_w = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    y = X @ true_w + 0.1 * torch.randn(16, dtype=torch.float64)
    w_init = torch.randn(4, dtype=torch.float64)
    return X, y, w_init


def _loss(X, y, w):
    return ((X @ w - y) ** 2).mean()


def test_adamw_matches_torch_over_5_steps():
    X, y, w_init = _toy_regression()
    lr, betas, wd = 0.05, (0.9, 0.999), 0.1

    w_ours = w_init.clone().requires_grad_(True)
    opt_ours = AdamW([w_ours], lr=lr, betas=betas, eps=1e-8, weight_decay=wd)

    w_torch = w_init.clone().requires_grad_(True)
    opt_torch = torch.optim.AdamW([w_torch], lr=lr, betas=betas, eps=1e-8, weight_decay=wd)

    for _ in range(5):
        opt_ours.zero_grad()
        _loss(X, y, w_ours).backward()
        opt_ours.step()

        opt_torch.zero_grad()
        _loss(X, y, w_torch).backward()
        opt_torch.step()

        assert torch.allclose(w_ours, w_torch, atol=1e-5, rtol=0)


def test_sgd_momentum_matches_torch_over_5_steps():
    X, y, w_init = _toy_regression(seed=1)
    lr, mu = 0.02, 0.9

    w_ours = w_init.clone().requires_grad_(True)
    opt_ours = SGD([w_ours], lr=lr, momentum=mu)

    w_torch = w_init.clone().requires_grad_(True)
    opt_torch = torch.optim.SGD([w_torch], lr=lr, momentum=mu)

    for _ in range(5):
        opt_ours.zero_grad()
        _loss(X, y, w_ours).backward()
        opt_ours.step()

        opt_torch.zero_grad()
        _loss(X, y, w_torch).backward()
        opt_torch.step()

        assert torch.allclose(w_ours, w_torch, atol=1e-6, rtol=0)


def test_sgd_plain_matches_torch():
    X, y, w_init = _toy_regression(seed=2)
    lr = 0.05

    w_ours = w_init.clone().requires_grad_(True)
    opt_ours = SGD([w_ours], lr=lr, momentum=0.0)

    w_torch = w_init.clone().requires_grad_(True)
    opt_torch = torch.optim.SGD([w_torch], lr=lr, momentum=0.0)

    for _ in range(5):
        opt_ours.zero_grad()
        _loss(X, y, w_ours).backward()
        opt_ours.step()

        opt_torch.zero_grad()
        _loss(X, y, w_torch).backward()
        opt_torch.step()

        assert torch.allclose(w_ours, w_torch, atol=1e-6, rtol=0)


def test_cosine_warmup_endpoints_and_shape():
    warmup, total, max_lr, min_lr = 10, 100, 1.0, 0.1

    # Peak exactly at the end of warmup, floor exactly at the horizon.
    assert cosine_warmup_lr(warmup, warmup, total, max_lr, min_lr) == max_lr
    assert cosine_warmup_lr(total, warmup, total, max_lr, min_lr) == min_lr

    # Midpoint of the cosine phase is the average of peak and floor.
    mid = (warmup + total) // 2
    assert abs(cosine_warmup_lr(mid, warmup, total, max_lr, min_lr) - 0.55) < 1e-9

    # Warmup is a strictly increasing linear ramp; beyond the horizon is clamped.
    ramp = [cosine_warmup_lr(s, warmup, total, max_lr, min_lr) for s in range(warmup)]
    assert all(ramp[i] < ramp[i + 1] for i in range(len(ramp) - 1))
    assert cosine_warmup_lr(total + 50, warmup, total, max_lr, min_lr) == min_lr


def test_clip_grad_norm_matches_torch():
    torch.manual_seed(3)
    # Build two identical parameter sets carrying identical gradients.
    shapes = [(4, 4), (7,), (3, 2)]
    grads = [torch.randn(*s, dtype=torch.float64) for s in shapes]

    ours = [torch.zeros(*s, dtype=torch.float64, requires_grad=True) for s in shapes]
    ref = [torch.zeros(*s, dtype=torch.float64, requires_grad=True) for s in shapes]
    for p, g in zip(ours, grads):
        p.grad = g.clone()
    for p, g in zip(ref, grads):
        p.grad = g.clone()

    max_norm = 1.0
    returned = clip_grad_norm_(ours, max_norm)
    ref_norm = torch.nn.utils.clip_grad_norm_(ref, max_norm)

    # Same pre-clip norm reported...
    assert abs(returned - float(ref_norm)) < 1e-9
    # ...and the same rescaled gradients.
    for p_ours, p_ref in zip(ours, ref):
        assert torch.allclose(p_ours.grad, p_ref.grad, atol=1e-9, rtol=0)

    # A gradient already under the threshold must be left untouched.
    small = [torch.zeros(3, dtype=torch.float64, requires_grad=True)]
    small[0].grad = torch.tensor([0.1, 0.2, 0.2], dtype=torch.float64)
    before = small[0].grad.clone()
    clip_grad_norm_(small, max_norm=10.0)
    assert torch.allclose(small[0].grad, before, atol=0, rtol=0)
