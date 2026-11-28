"""Data parallelism, simulated in a single process.

Code companion to lesson ``09.1``. In real data-parallel training, ``W`` workers
each hold a full replica of the model, each processes a different **shard** of
the global batch, computes a gradient on its shard, and then an **all-reduce**
averages those gradients so every worker applies the identical update. We have
one CPU and no NCCL, so we **simulate** it: one process, a Python loop over the
``W`` workers, and :func:`llmre.distributed.collectives.ring_all_reduce` standing
in for the collective.

The whole point is the correctness identity this file demonstrates:

    averaging the per-shard **mean-loss** gradients == the gradient of the mean
    loss over the full (concatenated) batch,

provided the shards are equal in size. That equality is *why* data parallelism is
mathematically exact and not an approximation. See the lesson for the proof; the
test ``tests/test_distributed.py`` checks it to ``1e-6`` on a linear-regression
toy.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

from .collectives import ring_all_reduce


def split_batch(batch: Sequence, world_size: int) -> List:
    """Split ``batch`` into ``world_size`` equal contiguous shards along axis 0.

    Shape: ``batch`` is any sliceable container of length ``B`` (a NumPy array,
    torch tensor, or list) with ``B`` divisible by ``world_size``; returns a list
    of ``world_size`` shards each of length ``B / world_size``. Equal sizes are
    required for the averaging identity to hold exactly (unequal shards would need
    a size-weighted average instead of a plain mean).

    A ``tuple`` is treated as parallel fields of one batch, e.g. ``(X, y)``: each
    field is split along its first axis and every shard is again a tuple
    ``(X_shard, y_shard)``. All fields must have the same length ``B``.

    Args:
        batch: the global batch, first axis indexed by example, or a tuple of
            such arrays (inputs, targets, ...) sharing that first axis.
        world_size: number of simulated workers ``W``.

    Returns:
        List of ``W`` shards.
    """
    if isinstance(batch, tuple):
        lengths = {len(field) for field in batch}
        if len(lengths) != 1:
            raise ValueError(f"tuple batch fields have different lengths {sorted(lengths)}")
        per_field = [split_batch(field, world_size) for field in batch]
        return [tuple(field_shards) for field_shards in zip(*per_field)]
    B = len(batch)
    if world_size <= 0:
        raise ValueError("world_size must be >= 1")
    if B % world_size != 0:
        raise ValueError(
            f"batch length {B} is not divisible by world_size {world_size}; "
            "equal shards are required for exact gradient averaging"
        )
    step = B // world_size
    return [batch[w * step : (w + 1) * step] for w in range(world_size)]


def data_parallel_grads(
    grad_fn: Callable,
    params,
    batch: Sequence,
    world_size: int,
):
    """Simulate one data-parallel gradient computation and return the averaged grad.

    Splits ``batch`` into ``world_size`` equal shards, has each simulated worker
    compute its shard's **mean-loss** gradient via ``grad_fn``, then all-reduces
    (averages) the per-worker gradients with a ring all-reduce. The returned
    gradient equals the gradient of the mean loss over the whole batch (to
    floating-point precision), which is the correctness guarantee of data
    parallelism.

    Args:
        grad_fn: callable ``grad_fn(params, batch_shard) -> gradient``. It runs
            the whole local step for one worker: the **forward pass** (model
            output and mean loss on ``batch_shard``) and then the **backward
            pass** (gradient of that loss with respect to ``params``). This
            function is therefore called *before* any forward pass, not after.
            It MUST return the gradient of the **mean** loss over
            ``batch_shard`` (the same reduction the full-batch loss uses), as a
            single NumPy array or torch tensor whose shape matches ``params``.
        params: the current model weights (parameter values) at the start of
            this step, e.g. the weight vector ``theta`` in linear regression or
            the model's weight tensors. Passed to ``grad_fn`` unchanged and
            never updated here. In this simulation every worker shares the same
            ``params`` object, exactly as real replicas hold identical weights
            at step start.
        batch: the raw training examples for this step, i.e. inputs and their
            targets (for example ``X`` rows with matching ``y`` values), not
            activations or losses. Sliced along its first axis into
            ``world_size`` equal shards, so its length must be divisible by
            ``world_size``.
        world_size: number of simulated workers ``W``.

    Returns:
        The averaged gradient, same shape/dtype/device as one worker's gradient.
    """
    shards = split_batch(batch, world_size)
    per_worker_grads = [grad_fn(params, shard) for shard in shards]
    return ring_all_reduce(per_worker_grads, op="mean")
