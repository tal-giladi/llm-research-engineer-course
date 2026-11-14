"""Watch the gradient noise scale grow as a model trains (lesson 03.3, section 5).

Linear regression y = X w* + noise, trained with plain SGD. At a few points in
training we measure |G_B|^2 at a small and a big batch, average over many
draws, and estimate B_simple = tr(Sigma) / |G|^2 with
llmre.optim.batch_schedule.gradient_noise_scale.

CPU only, runs in a couple of seconds:
    cd code && py experiments/noise_scale_toy.py
"""

from __future__ import annotations

import torch

from llmre.optim.batch_schedule import gradient_noise_scale

torch.manual_seed(0)
N, D = 20_000, 16                      # dataset size, features
X = torch.randn(N, D)                  # (N, D) float32, cpu
w_star = torch.randn(D)                # (D,) the true weights
y = X @ w_star + 0.5 * torch.randn(N)  # (N,) targets with label noise
w = torch.zeros(D)                     # (D,) start far from w_star


def batch_grad(w: torch.Tensor, b: int) -> torch.Tensor:
    """Mean gradient of 0.5 * (x.w - y)^2 over a random batch of b examples. (D,)"""
    idx = torch.randint(0, N, (b,))
    xb, yb = X[idx], y[idx]            # (b, D), (b,)
    return xb.T @ (xb @ w - yb) / b    # (D,)


def noise_scale(w: torch.Tensor, b_small=8, b_big=256, draws=400) -> tuple[float, float, float]:
    gs = sum(batch_grad(w, b_small).pow(2).sum().item() for _ in range(draws)) / draws
    gb = sum(batch_grad(w, b_big).pow(2).sum().item() for _ in range(draws)) / draws
    return gradient_noise_scale(gs, gb, b_small, b_big)


lr = 0.05
for step in range(301):
    if step in (0, 25, 50, 100, 300):
        g_sq, tr, b = noise_scale(w)
        loss = 0.5 * (X @ w - y).pow(2).mean().item()
        print(f"step {step:4d}  loss {loss:8.4f}  |G|^2 {g_sq:9.4f}  tr(Sigma) {tr:9.2f}  B_simple {b:9.1f}")
    w -= lr * batch_grad(w, 32)
