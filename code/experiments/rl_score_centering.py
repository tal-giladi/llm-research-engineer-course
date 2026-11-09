"""Score centering on an exactly enumerable bandit (lesson 16.4 extension).

Runs on CPU in about a second. Not a pytest test::

    cd code && PYTHONPATH=src python experiments/rl_score_centering.py

A. A constant reward: how far the policy moves toward the sampler.
B. The KL-regularized bandit of 16.4: fixed points of each estimator.
C. Severe mismatch: alignment of each expected update with the true gradient.
"""

from __future__ import annotations

import torch

from llmre.rl.mismatch import (
    kl_regularized_optimum,
    kl_regularized_value,
    softmax_bandit_gradient,
    train_kl_bandit,
)
from llmre.rl.score_centering import centered_bandit_gradient, train_centered_bandit

D = torch.float64
R = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=D)
REF = torch.log_softmax(torch.zeros(4, dtype=D), -1)
BETA = 0.5
BIAS = torch.tensor([-1.0, 1.0, 0.0, 0.0], dtype=D)


def kl(lp: torch.Tensor, ref: torch.Tensor) -> float:
    return float((lp.exp() * (lp - ref)).sum())


def part_a() -> None:
    print("A. constant reward +1, fixed sampler, 500 exact steps, beta=0")
    sampler = torch.log_softmax(BIAS, -1)
    r = torch.ones(4, dtype=D)
    for name, grad in [
        ("plain PG", lambda th: softmax_bandit_gradient(th, r, REF, 0.0, sampler, "none")),
        ("score centering", lambda th: centered_bandit_gradient(th, r, REF, 0.0, sampler)),
    ]:
        theta = REF.clone()
        for _ in range(500):
            theta = theta + 0.5 * grad(theta)
        lp = torch.log_softmax(theta, -1)
        print(f"   {name:16s} pi = {lp.exp().numpy().round(3)}  KL(pi||sampler) = {kl(lp, sampler):.2e}")


def part_b() -> None:
    print("B. KL-regularized bandit: KL(learned || optimum) after 2000 exact steps")
    opt = kl_regularized_optimum(R, REF, BETA)
    print(f"   optimum pi* = {opt.exp().numpy().round(3)}")
    rows = [
        ("no correction", train_kl_bandit(R, REF, BETA, BIAS, "none")),
        ("TIS cap 1.0", train_kl_bandit(R, REF, BETA, BIAS, "tis", cap=1.0)),
        ("SC", train_centered_bandit(R, REF, BETA, BIAS)),
        ("SC + TIS cap 1.0", train_centered_bandit(R, REF, BETA, BIAS, cap=1.0)),
    ]
    for name, lp in rows:
        print(f"   {name:16s} pi = {lp.exp().numpy().round(3)}  KL = {kl(lp, opt):.2e}")


def part_c() -> None:
    print("C. cosine with the true gradient at one point, by sampler bias on the good arm")
    theta = torch.tensor([0.0, 0.5, 0.2, -0.2], dtype=D)
    th = theta.clone().requires_grad_(True)
    kl_regularized_value(th, R, REF, BETA).backward()
    true = th.grad

    def cos(g: torch.Tensor) -> float:
        return float(torch.dot(g, true) / (g.norm() * true.norm()))

    for scale in (1.0, 3.0, 5.0):
        q = torch.log_softmax(theta + scale * BIAS, -1)
        print(
            f"   bias x{scale:.0f}  PG {cos(softmax_bandit_gradient(theta, R, REF, BETA, q, 'none')):+.3f}"
            f"  TIS2 {cos(softmax_bandit_gradient(theta, R, REF, BETA, q, 'tis', 2.0)):+.3f}"
            f"  SC {cos(centered_bandit_gradient(theta, R, REF, BETA, q)):+.3f}"
            f"  SC+TIS2 {cos(centered_bandit_gradient(theta, R, REF, BETA, q, 2.0)):+.3f}"
        )


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
