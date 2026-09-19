"""Tests for llmre.data.loader — document packing and next-token batching.

These back Module 4's data-loading lesson: that packing inserts the end-of-text
separators, that a batch has shape ``(B, T)``, and that the target ``y`` is the
input ``x`` shifted left by exactly one position.
"""

from __future__ import annotations

import torch

from llmre.data.loader import get_batch, pack_documents


def test_pack_documents_inserts_eot():
    eot = 999
    packed = pack_documents([[1, 2, 3], [4, 5]], eot_id=eot)
    assert packed.dtype == torch.long
    assert packed.dim() == 1
    assert packed.tolist() == [1, 2, 3, eot, 4, 5, eot]


def test_get_batch_shapes():
    data = torch.arange(100, dtype=torch.long)
    B, T = 4, 8
    x, y = get_batch(data, block_size=T, batch_size=B)
    assert x.shape == (B, T)
    assert y.shape == (B, T)
    assert x.dtype == torch.long and y.dtype == torch.long


def test_get_batch_target_is_input_shifted_by_one():
    # Contiguous data makes the shift checkable exactly: every window is a run of
    # consecutive integers, so its target window is that run plus one.
    data = torch.arange(200, dtype=torch.long)
    B, T = 6, 12
    torch.manual_seed(0)
    x, y = get_batch(data, block_size=T, batch_size=B)

    # The defining relationship: y[:, :-1] == x[:, 1:].
    assert torch.equal(y[:, :-1], x[:, 1:])
    # And because the stream is arange, y is exactly x + 1 everywhere.
    assert torch.equal(y, x + 1)


def test_get_batch_rejects_too_short_data():
    data = torch.arange(5, dtype=torch.long)
    try:
        get_batch(data, block_size=10, batch_size=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when block_size >= len(data)")
