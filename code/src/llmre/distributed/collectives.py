"""Collective communication primitives, simulated in a single process.

This module is the code companion to lesson ``09.1 · Data parallelism,
all-reduce, DDP`` (``lessons/module-09/lesson-01.md``). On real hardware the
operations here are performed by a collective library — **NCCL** on NVIDIA GPUs —
that moves data over NVLink / InfiniBand between many devices at once. We have a
single CPU and no NCCL, so **everything in this file is a pure-Python
simulation**: we hold every "worker's" tensor in one process, in an ordinary
Python list, and loop over the workers by hand. No data ever leaves the process
and nothing runs in parallel.

That is deliberate and it is enough, because a collective is defined by *the
number it must produce*, not by the wires it runs on. ``ring_all_reduce`` below
reproduces the exact bytes a real ring all-reduce would leave on every worker,
and it does so by actually walking the ring — reduce-scatter then all-gather,
``W`` chunks, ``2(W-1)`` communication steps — so you can read the algorithm off
the code. What we *cannot* show on one CPU is the wall-clock speedup; the
``<div class="hw">`` boxes in the lesson give the real-hardware picture.

Supported array types: 1-D or N-D NumPy ``ndarray`` and PyTorch ``Tensor``. The
ring works on a flattened view and restores the original shape at the end.
"""

from __future__ import annotations

from typing import List, Sequence


def _is_torch(x) -> bool:
    """True if ``x`` is a ``torch.Tensor`` (checked without importing torch eagerly)."""
    return type(x).__module__.split(".")[0] == "torch"


def _flatten_copy(x):
    """Return a mutable, contiguous 1-D copy of ``x`` (NumPy or torch)."""
    if _is_torch(x):
        return x.reshape(-1).clone()
    return x.reshape(-1).copy()


def _chunk_bounds(n: int, w: int) -> List[tuple]:
    """Split ``range(n)`` into ``w`` contiguous, as-equal-as-possible chunks.

    Returns a list of ``(start, stop)`` index pairs, mirroring how a real
    ring all-reduce partitions each worker's buffer into ``w`` pieces so that
    every communication step ships about ``1/w`` of the tensor. The first
    ``n % w`` chunks get one extra element when ``n`` is not divisible by ``w``.
    """
    base, rem = divmod(n, w)
    bounds, start = [], 0
    for i in range(w):
        size = base + (1 if i < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def ring_all_reduce(shards: Sequence, op: str = "mean"):
    """Simulate a bandwidth-optimal **ring all-reduce** over ``W`` workers.

    Each element of ``shards`` is one worker's tensor; **all shards must have the
    same shape and dtype**. The function returns the single tensor that every
    worker would hold after the collective: the element-wise ``sum`` of all
    shards (``op="sum"``) or that sum divided by ``W`` (``op="mean"``, the
    default — this is what data-parallel gradient averaging needs, see
    :func:`llmre.distributed.data_parallel.data_parallel_grads`).

    Shape / dtype / device: input tensors are ``(d0, d1, ...)`` NumPy arrays or
    torch tensors on any single device; the result has the identical shape and
    dtype and (for torch) sits on the same device as ``shards[0]``. This is a
    single-process simulation — no tensor is moved between devices.

    The algorithm (walked literally below), for ``W`` workers:

    1. Each worker splits its buffer into ``W`` chunks.
    2. **Reduce-scatter**, ``W-1`` steps: on each step every worker sends one
       chunk to its right neighbour and adds the chunk it receives into its own
       buffer. After the phase, worker ``i`` holds the fully summed value of
       exactly one chunk.
    3. **All-gather**, ``W-1`` steps: the finished chunks are passed around the
       ring until every worker has all ``W`` of them.

    Total: ``2(W-1)`` steps, each moving ``1/W`` of the tensor per link — the
    reason ring all-reduce is bandwidth-optimal and independent of ``W`` in the
    volume each link carries. The naive alternative (send everything to one
    worker, sum, broadcast back) moves far more over the busiest link.

    Args:
        shards: sequence of ``W >= 1`` equally-shaped NumPy/torch tensors.
        op: ``"mean"`` (default) or ``"sum"``.

    Returns:
        One tensor of the same shape/dtype/device as ``shards[0]``.
    """
    if op not in ("mean", "sum"):
        raise ValueError(f"op must be 'mean' or 'sum', got {op!r}")
    shards = list(shards)
    W = len(shards)
    if W == 0:
        raise ValueError("ring_all_reduce needs at least one shard")

    torch_mode = _is_torch(shards[0])
    orig_shape = shards[0].shape
    n = shards[0].reshape(-1).shape[0]

    # Each worker's local buffer: a flat, independently mutable copy.
    buf = [_flatten_copy(s) for s in shards]
    bounds = _chunk_bounds(n, W)

    def chunk(worker: int, c: int):
        lo, hi = bounds[c]
        return buf[worker][lo:hi]

    def set_chunk(worker: int, c: int, value):
        lo, hi = bounds[c]
        buf[worker][lo:hi] = value

    # ---- Phase 1: reduce-scatter (W-1 steps) ----
    # At step t, worker i sends chunk (i - t) mod W to worker (i+1) mod W, and
    # adds the chunk it receives from (i-1) mod W into the same slot.
    for t in range(W - 1):
        # Snapshot every chunk that is about to be sent (so adds don't race).
        sent = []
        for i in range(W):
            c = (i - t) % W
            piece = chunk(i, c)
            sent.append(piece.clone() if torch_mode else piece.copy())
        for i in range(W):
            src = (i - 1) % W
            c = (i - 1 - t) % W  # the chunk index src transmitted == this slot
            # sent[src] is a copy of exactly chunk c, so add it in whole.
            set_chunk(i, c, chunk(i, c) + sent[src])

    # ---- Phase 2: all-gather (W-1 steps) ----
    # After reduce-scatter, worker i owns the finished chunk (i+1) mod W.
    for t in range(W - 1):
        sent = []
        for i in range(W):
            c = (i + 1 - t) % W
            piece = chunk(i, c)
            sent.append(piece.clone() if torch_mode else piece.copy())
        for i in range(W):
            src = (i - 1) % W
            c = (i - t) % W  # = (src + 1 - t) mod W == the chunk src just sent
            set_chunk(i, c, sent[src])

    result = buf[0]
    if op == "mean":
        result = result / W
    return result.reshape(orig_shape)


def all_gather(shards: Sequence):
    """Simulate an **all-gather**: concatenate every worker's shard.

    In a real all-gather each of ``W`` workers contributes its own shard and
    ends up holding the concatenation of all ``W`` shards (used, e.g., by FSDP to
    reassemble a full parameter from its shards just before a layer runs —
    lesson 09.2). Here we return that concatenation once; in the real collective
    every worker would hold an identical copy of it.

    Shape / dtype / device: given ``W`` tensors each ``(s_k, ...)`` that agree on
    every dimension except the first, returns one tensor
    ``(s_0 + s_1 + ... + s_{W-1}, ...)`` of the same dtype/device, concatenated
    along axis 0 in worker order.

    Args:
        shards: sequence of ``W >= 1`` tensors (NumPy or torch) that share all
            trailing dimensions.

    Returns:
        The axis-0 concatenation of all shards.
    """
    shards = list(shards)
    if len(shards) == 0:
        raise ValueError("all_gather needs at least one shard")
    if _is_torch(shards[0]):
        import torch

        return torch.cat(shards, dim=0)
    import numpy as np

    return np.concatenate(shards, axis=0)
