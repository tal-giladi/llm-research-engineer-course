"""FlashAttention — the *algorithm*, written in pure PyTorch (Module 8, lesson 04).

This file demonstrates the FlashAttention algorithm of Dao et al. 2022
(``papers/index.md`` #7) on the CPU. It is **not** a fused CUDA/Triton kernel and
it is **not** fast — every operation still runs as an ordinary PyTorch op and the
intermediate block scores are still materialized in Python. What it *does* show,
exactly, is the two ideas that make the real kernel work:

1. **Tiling.** Q, K and V are cut into blocks. We loop over key/value blocks and
   accumulate the output one block at a time, so we never build the full
   ``(T, T)`` score matrix. The real kernel keeps each block in on-chip SRAM.
2. **Online softmax.** softmax needs the row maximum and the row sum, which
   normally require seeing the whole row at once. The online-softmax recurrence
   maintains a running max ``m`` and a running denominator ``l`` and *rescales*
   the partial output every time a new block shifts the max — so the final result
   is bit-for-bit the same math as ``softmax(QKᵀ/√d + mask) V`` without ever
   holding a full row of scores.

The output is numerically identical (to floating-point tolerance) to
``llmre.attention.scaled_dot_product_attention`` and to
``torch.nn.functional.scaled_dot_product_attention``. The point is the *bytes
moved*, not the FLOPs: the real kernel turns attention from ``O(T²)`` HBM traffic
into ``O(T)``, which is why it is faster and uses far less memory on a GPU even
though it does the same (actually slightly more) arithmetic.
"""

from __future__ import annotations

import math

import torch


def flash_attention_reference(q, k, v, causal: bool = False, block_size: int = 64):
    """Exact scaled-dot-product attention via the FlashAttention tiling algorithm.

    Pure-PyTorch reference implementation of the FlashAttention forward pass. It
    computes the same result as ``softmax(q @ k.transpose(-2,-1) / sqrt(d) + mask) @ v``
    but tiles the key/value axis into blocks and uses the online-softmax
    recurrence (running max ``m`` and running denominator ``l``) so the full
    ``(T, T)`` score matrix is never materialized as a single tensor.

    This is the *algorithm*, illustrated on CPU — not a fused CUDA/Triton kernel.
    On a real GPU the block loop lives inside one kernel with the blocks held in
    SRAM; here each block is an ordinary PyTorch op.

    Args:
        q: query tensor, shape ``(B, nh, T, hd)``, any float dtype, any device.
        k: key tensor, shape ``(B, nh, T, hd)``. Same dtype/device as ``q``.
        v: value tensor, shape ``(B, nh, T, hd)``. Same dtype/device as ``q``.
        causal: if ``True``, apply a causal mask so query position ``i`` may only
            attend to key positions ``j <= i`` (no peeking at the future).
        block_size: the tile length ``Bc`` along the sequence axis for both the
            query loop and the key/value loop. Any positive int; the result does
            not depend on it (only the memory-traffic story does).

    Returns:
        Tensor of shape ``(B, nh, T, hd)`` — the attention output, one attended
        value vector per query position — with the same dtype/device as ``q``.

    Note:
        The scale is ``1 / sqrt(hd)`` with ``hd = q.size(-1)``, matching
        ``scaled_dot_product_attention``. Accumulation is done in the input dtype;
        pass float64 inputs if you want to check parity to very tight tolerances.
    """
    B, nh, T, hd = q.shape
    scale = 1.0 / math.sqrt(hd)

    out = torch.empty_like(q)

    # Outer loop over query blocks. Each query block is processed independently:
    # it scans the key/value blocks and builds its own output rows.
    for i0 in range(0, T, block_size):
        i1 = min(i0 + block_size, T)
        q_blk = q[:, :, i0:i1, :]  # (B, nh, Bq, hd)
        Bq = i1 - i0

        # Running statistics for this query block, one entry per query row.
        # m: running row max (start at -inf so the first real score sets it).
        # l: running softmax denominator sum(exp(score - m)) (start at 0).
        # acc: running UNNORMALIZED output, sum(exp(score - m) * v) (start at 0).
        m = torch.full((B, nh, Bq, 1), float("-inf"), dtype=q.dtype, device=q.device)
        l = torch.zeros((B, nh, Bq, 1), dtype=q.dtype, device=q.device)
        acc = torch.zeros((B, nh, Bq, hd), dtype=q.dtype, device=q.device)

        # Inner loop over key/value blocks.
        for j0 in range(0, T, block_size):
            j1 = min(j0 + block_size, T)

            # Causal skip: if every key in this block is strictly in the future of
            # every query in the current query block, the whole block is masked.
            if causal and j0 > i1 - 1:
                break

            k_blk = k[:, :, j0:j1, :]  # (B, nh, Bk, hd)
            v_blk = v[:, :, j0:j1, :]  # (B, nh, Bk, hd)

            # Block scores: (B, nh, Bq, Bk). This is the only place a T×T-shaped
            # object appears, and it is only Bq×Bk — never the full matrix.
            s = (q_blk @ k_blk.transpose(-2, -1)) * scale

            if causal:
                # Per-element mask inside the block: forbid key j > query i.
                q_idx = torch.arange(i0, i1, device=q.device).view(Bq, 1)
                k_idx = torch.arange(j0, j1, device=q.device).view(1, j1 - j0)
                s = s.masked_fill(k_idx > q_idx, float("-inf"))

            # --- online-softmax update ---
            # 1) new running max after seeing this block.
            m_blk = s.max(dim=-1, keepdim=True).values          # (B,nh,Bq,1)
            m_new = torch.maximum(m, m_blk)
            # 2) correction factor that rescales the OLD accumulators to the new max.
            #    exp(m_old - m_new) <= 1; where m was -inf this is exp(-inf)=0 (via nan_to_num).
            corr = torch.exp(m - m_new).nan_to_num_(0.0)        # (B,nh,Bq,1)
            # 3) exponentiate this block's scores against the new max.
            p = torch.exp(s - m_new)                             # (B,nh,Bq,Bk)
            # 4) update denominator and unnormalized output.
            l = l * corr + p.sum(dim=-1, keepdim=True)
            acc = acc * corr + p @ v_blk
            m = m_new

        # Normalize once, at the end: divide the accumulated numerator by l.
        out[:, :, i0:i1, :] = acc / l

    return out
