"""Training-inference mismatch: where it comes from and what correcting it buys.

Runnable companion to lesson 16.4 (`lessons/module-16/lesson-04.md`). Three
parts, all on CPU in a few seconds:

A. Batch variance. The same 4 requests are scored alone (batch of 4) and inside
   a batch of 64 by a toy engine whose split-K choice depends on batch size
   (:func:`llmre.rl.mismatch.toy_logprobs`). Prints the largest log-prob
   difference with fp32 and bf16 accumulation, with and without batch
   invariance.
B. Accumulation over length. Per-token log-ratio noise of a given size is
   summed over responses of growing length; prints the spread of the
   sequence-level log ratio and the share of sequence-level weights that a
   cap of 2 truncates.
C. Consequence. Exact (enumerated) gradient ascent on a KL-regularized bandit
   whose sampler has a fixed logit bias; prints the KL from the learned policy
   to the analytical optimum for no correction, TIS with several caps, and full
   IS, plus a sampler that can never produce one action (top-k style).

It is NOT a pytest test::

    cd code && PYTHONPATH=src python experiments/rl_mismatch.py

Everything is seeded; numbers may differ in the last digits across CPUs.
"""

from __future__ import annotations

import torch

from llmre.rl.mismatch import (
    kl_regularized_optimum,
    toy_logprobs,
    train_kl_bandit,
    truncated_is_weights,
)


def part_a() -> None:
    print("A. batch variance: max |logp(alone) - logp(in batch of 64)| over 4 requests")
    torch.manual_seed(0)
    d, V = 64, 50
    W = torch.randn(d, V) * 0.5
    x = torch.randn(64, d)
    tok = torch.randint(0, V, (64,))
    for name, acc in [("fp32 accumulate", torch.float32), ("bf16 accumulate", torch.bfloat16)]:
        for inv in (False, True):
            alone = toy_logprobs(x[:4], W, tok[:4], batch_invariant=inv, acc_dtype=acc)
            inb = toy_logprobs(x, W, tok, batch_invariant=inv, acc_dtype=acc)[:4]
            print(f"   {name:16s} batch_invariant={inv!s:5s}  {float((alone - inb).abs().max()):.3e}")


def part_b() -> None:
    print("B. per-token noise sigma -> spread of the sequence log ratio, share of weights capped at 2")
    torch.manual_seed(1)
    for sigma in (1e-3, 1e-2):
        for T in (100, 1000, 10000):
            delta = torch.randn(2000, T) * sigma
            seq = delta.sum(-1)
            w = truncated_is_weights(delta, torch.zeros_like(delta), cap=2.0, level="sequence")[:, 0]
            capped = float((torch.exp(seq) > 2.0).float().mean())
            print(f"   sigma={sigma:.0e} T={T:5d}  std(sum log rho)={float(seq.std()):.3f}"
                  f"  capped={capped:6.1%}  mean w={float(w.mean()):.3f}")


def kl(lp: torch.Tensor, ref: torch.Tensor) -> float:
    return float((lp.exp() * (lp - ref)).sum())


def part_c() -> None:
    print("C. KL-regularized bandit (beta=0.5): KL(learned || optimum) after 2000 exact steps")
    K = 4
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    ref = torch.log_softmax(torch.zeros(K), -1)
    beta = 0.5
    opt = kl_regularized_optimum(r, ref, beta).double()
    bias = torch.tensor([-1.0, 1.0, 0.0, 0.0])  # sampler under-weights the good arm
    print(f"   optimum pi* = {opt.exp().numpy().round(3)}")
    for label, corr, cap in [("no correction", "none", 0), ("TIS cap 1.0", "tis", 1.0),
                             ("TIS cap 1.5", "tis", 1.5), ("TIS cap 2.0", "tis", 2.0),
                             ("full IS", "is", 0)]:
        lp = train_kl_bandit(r, ref, beta, bias, corr, cap=cap or 2.0)
        print(f"   {label:14s} pi = {lp.exp().numpy().round(3)}  KL = {kl(lp, opt):.2e}")
    r2 = torch.tensor([1.0, 0.0, 0.0, 0.5])
    opt2 = kl_regularized_optimum(r2, ref, beta).double()
    cut = torch.tensor([0.0, 0.0, 0.0, -float("inf")])  # sampler can never emit arm 3
    print(f"   support mismatch, optimum pi* = {opt2.exp().numpy().round(3)}")
    for label, corr in [("no correction", "none"), ("full IS", "is")]:
        lp = train_kl_bandit(r2, ref, beta, cut, corr)
        print(f"   {label:14s} pi = {lp.exp().numpy().round(3)}  KL = {kl(lp, opt2):.2e}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
