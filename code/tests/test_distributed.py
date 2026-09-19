"""Tests for llmre.distributed (Module 9, single-process simulation).

Three things are checked:

1. ``ring_all_reduce`` produces exactly the true element-wise sum / mean of the
   shards, for several worker counts and both NumPy and torch, and it leaves the
   *same* result regardless of ``W`` (the ring is just a schedule for computing a
   sum, so the answer must not depend on how many workers carry it).
2. The simulated data-parallel averaged gradient over ``W`` shards equals the
   single full-batch gradient for a linear-regression toy, to ``1e-6`` — the
   correctness identity that makes data parallelism exact.
3. ``all_gather`` concatenation has the right shape and content.
"""

import numpy as np
import torch

from llmre.distributed import (
    all_gather,
    data_parallel_grads,
    ring_all_reduce,
    split_batch,
)


# ---------------------------------------------------------------------------
# 1. ring_all_reduce == true sum / mean
# ---------------------------------------------------------------------------

def test_ring_all_reduce_sum_matches_numpy():
    rng = np.random.default_rng(0)
    for W in (1, 2, 3, 4, 5, 8):
        shards = [rng.standard_normal((7,)) for _ in range(W)]
        truth = np.sum(shards, axis=0)
        got = ring_all_reduce(shards, op="sum")
        assert np.allclose(got, truth, atol=1e-12), (W, got, truth)


def test_ring_all_reduce_mean_matches_numpy():
    rng = np.random.default_rng(1)
    for W in (1, 2, 3, 4, 6, 8):
        shards = [rng.standard_normal((10,)) for _ in range(W)]
        truth = np.mean(shards, axis=0)
        got = ring_all_reduce(shards, op="mean")
        assert np.allclose(got, truth, atol=1e-12), (W, got, truth)


def test_ring_all_reduce_preserves_shape_2d():
    rng = np.random.default_rng(2)
    shards = [rng.standard_normal((4, 5)) for _ in range(3)]
    truth = np.sum(shards, axis=0)
    got = ring_all_reduce(shards, op="sum")
    assert got.shape == (4, 5)
    assert np.allclose(got, truth, atol=1e-12)


def test_ring_all_reduce_torch():
    torch.manual_seed(0)
    for W in (1, 2, 3, 4, 7):
        shards = [torch.randn(6, dtype=torch.float64) for _ in range(W)]
        truth = torch.stack(shards).sum(dim=0)
        got = ring_all_reduce(shards, op="sum")
        assert torch.allclose(got, truth, atol=1e-12), (W, got, truth)
        got_mean = ring_all_reduce(shards, op="mean")
        assert torch.allclose(got_mean, truth / W, atol=1e-12)


def test_ring_all_reduce_indivisible_length():
    # length 10 chunked over 3 workers is 4+3+3 — the ring must still be exact.
    rng = np.random.default_rng(3)
    shards = [rng.standard_normal((10,)) for _ in range(3)]
    got = ring_all_reduce(shards, op="sum")
    assert np.allclose(got, np.sum(shards, axis=0), atol=1e-12)


# ---------------------------------------------------------------------------
# 2. data-parallel averaged gradient == full-batch gradient
# ---------------------------------------------------------------------------

def _lin_reg_mean_grad(params, batch):
    """Gradient of the MEAN squared-error loss w.r.t. the weight vector.

    ``params`` is the weight vector ``w`` of shape ``(d,)``. ``batch`` is a tuple
    ``(X, y)`` with ``X`` of shape ``(n, d)`` and ``y`` of shape ``(n,)``. The
    loss is ``mean_i (X_i . w - y_i)^2`` and its gradient is
    ``(2/n) X^T (X w - y)``.
    """
    X, y = batch
    n = X.shape[0]
    resid = X @ params - y                     # (n,)
    return (2.0 / n) * (X.T @ resid)           # (d,)


class _XY:
    """Tiny sliceable batch: slicing indexes rows of X and entries of y together."""

    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, s):
        return self.X[s], self.y[s]


def test_data_parallel_grad_equals_full_batch():
    rng = np.random.default_rng(4)
    d = 5
    for W in (1, 2, 4, 8):
        B = 8 * W                              # divisible by W, equal shards
        X = rng.standard_normal((B, d))
        y = rng.standard_normal((B,))
        w = rng.standard_normal((d,))

        full = _lin_reg_mean_grad(w, (X, y))
        avg = data_parallel_grads(_lin_reg_mean_grad, w, _XY(X, y), W)

        assert avg.shape == (d,)
        assert np.allclose(avg, full, atol=1e-6), (W, avg, full)


def test_split_batch_requires_divisible():
    X = np.zeros((7, 2))
    y = np.zeros((7,))
    try:
        split_batch(_XY(X, y), 3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for indivisible batch")


# ---------------------------------------------------------------------------
# 3. all_gather shape / content
# ---------------------------------------------------------------------------

def test_all_gather_numpy():
    shards = [np.arange(0, 3), np.arange(3, 6), np.arange(6, 9)]
    got = all_gather(shards)
    assert got.shape == (9,)
    assert np.array_equal(got, np.arange(9))


def test_all_gather_torch_2d():
    shards = [torch.arange(0, 4).reshape(2, 2), torch.arange(4, 8).reshape(2, 2)]
    got = all_gather(shards)
    assert got.shape == (4, 2)
    assert torch.equal(got, torch.arange(8).reshape(4, 2))
