"""Lesson 09.4 practice: DDP + gradient accumulation + no_sync(), on CPU.

Runs W=2 processes (Gloo backend, CPU only) and checks three things:
  1. DDP with accumulation (A micro-batches per rank) gives the same gradient
     as one process computing the mean loss over the whole global batch.
  2. Wrapping the first A-1 micro-batches in model.no_sync() gives the same
     gradient as not using it at all.
  3. no_sync() cuts the number of gradient all-reduces from A to 1 per step.

Usage (from code/):  py experiments/accum_ddp_check.py
"""

import contextlib
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.parallel import DistributedDataParallel as DDP

W = 2        # workers (ranks)
A = 4        # micro-batches per rank per optimizer step
B_MICRO = 3  # examples per micro-batch
D = 5        # input features


def make_data():
    g = torch.Generator().manual_seed(0)
    n = W * A * B_MICRO                      # global batch = 2 * 4 * 3 = 24
    x = torch.randn(n, D, generator=g)
    y = torch.randn(n, 1, generator=g)
    return x, y


def make_model():
    torch.manual_seed(1)
    return torch.nn.Linear(D, 1)


def reference_grad():
    """One process, one backward, mean loss over all 24 examples."""
    x, y = make_data()
    model = make_model()
    torch.nn.functional.mse_loss(model(x), y).backward()
    return model.weight.grad.clone()


def worker(rank, use_no_sync, out):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29517" if use_no_sync else "29518"
    dist.init_process_group("gloo", rank=rank, world_size=W)

    x, y = make_data()
    # This rank's shard: rows rank, rank+W, rank+2W, ... (like DistributedSampler without shuffle).
    xs, ys = x[rank::W], y[rank::W]                     # [A*B_MICRO, D], [A*B_MICRO, 1]

    model = DDP(make_model())
    calls = {"n": 0}

    def counting_hook(state, bucket):
        state["n"] += 1
        return allreduce_hook(None, bucket)              # the normal averaging all-reduce

    model.register_comm_hook(calls, counting_hook)

    for a in range(A):
        xb = xs[a * B_MICRO:(a + 1) * B_MICRO]           # [B_MICRO, D]
        yb = ys[a * B_MICRO:(a + 1) * B_MICRO]
        is_last = a == A - 1
        ctx = contextlib.nullcontext() if (is_last or not use_no_sync) else model.no_sync()
        with ctx:                                        # forward AND backward inside
            loss = torch.nn.functional.mse_loss(model(xb), yb) / A
            loss.backward()

    if rank == 0:
        out.put((use_no_sync, model.module.weight.grad.tolist(), calls["n"]))
    dist.destroy_process_group()


def main():
    ref = reference_grad()
    ctx = mp.get_context("spawn")
    results = {}
    for use_no_sync in (True, False):
        q = ctx.Queue()
        mp.spawn(worker, args=(use_no_sync, q), nprocs=W, join=True)
        flag, grad, n_calls = q.get()
        results[flag] = (torch.tensor(grad), n_calls)

    g_sync, n_sync = results[True]
    g_plain, n_plain = results[False]
    print(f"global batch = W*A*b = {W}*{A}*{B_MICRO} = {W * A * B_MICRO}")
    print(f"max |grad(no_sync) - grad(single process)| = {(g_sync - ref).abs().max():.2e}")
    print(f"max |grad(no_sync) - grad(sync every step)| = {(g_sync - g_plain).abs().max():.2e}")
    print(f"all-reduce calls per optimizer step: with no_sync = {n_sync}, without = {n_plain}")
    assert torch.allclose(g_sync, ref, atol=1e-6)
    assert torch.allclose(g_sync, g_plain, atol=1e-6)
    assert (n_sync, n_plain) == (1, A)
    print("OK")


if __name__ == "__main__":
    main()
