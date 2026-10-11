"""Fused linear cross-entropy: the LM-head loss without the full logits tensor (Module 8, lesson 05).

This module extends two foundational pieces of the course, which stay unchanged:

* the plain loss in :mod:`llmre.model.gpt` (``logits = lm_head(x)`` then
  ``F.cross_entropy(logits, targets)``), which materializes the whole ``(N, V)``
  logits tensor; and
* the online-softmax recurrence of :mod:`llmre.attention.flash` (running max
  ``m`` and running denominator ``l``), reused here over the *vocabulary* axis
  instead of the key axis.

Everything here is the *algorithm* in pure PyTorch on the CPU, not a fused Triton
kernel. Each tile is an ordinary PyTorch op, so the code is not fast; what it
shows exactly is which intermediates are never built. Three strategies:

1. :func:`chunked_linear_cross_entropy` tiles over **positions** (rows). For each
   chunk of rows it builds that chunk's ``(n, V)`` logits, computes the loss *and
   the gradients* ``dh`` and ``dW`` right away, then drops the chunk. This is the
   Liger-Kernel / chunked-loss approach. Peak extra memory: one ``(n, V)`` chunk.
2. :func:`vocab_tiled_linear_cross_entropy` tiles over the **vocabulary**. It
   computes ``logsumexp`` online across vocabulary tiles (the FlashAttention
   recurrence) and the target logit as a single dot product ``h · W[y]``, then
   recomputes the tiles in backward. This is the Cut Cross-Entropy idea. Peak
   extra memory: one ``(N, Bv)`` tile (a real kernel also tiles ``N`` and keeps
   the tile in SRAM, so almost nothing reaches HBM).
3. :func:`fused_logprobs_and_entropy` is the scoring-only variant used by RL and
   distillation trainers: per-token log-probabilities and entropies, from the same
   online recurrence plus one extra running sum, with no ``(N, V)`` tensor.

All of them return the same numbers as the naive path to floating-point tolerance.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def logits_bytes(n_tokens: int, vocab_size: int, bytes_per_element: int = 4) -> int:
    """Bytes of one ``(n_tokens, vocab_size)`` logits tensor.

    Args:
        n_tokens: number of positions ``N = B * T``.
        vocab_size: vocabulary size ``V``.
        bytes_per_element: 4 for fp32, 2 for bf16/fp16.

    Returns:
        ``n_tokens * vocab_size * bytes_per_element`` as an int.
    """
    return int(n_tokens) * int(vocab_size) * int(bytes_per_element)


def _acc_dtype(t: torch.Tensor) -> torch.dtype:
    """Accumulation dtype: fp32 for bf16/fp16/fp32 inputs, fp64 for fp64 inputs."""
    return torch.promote_types(t.dtype, torch.float32)


def _check_inputs(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> None:
    if h.dim() != 2 or weight.dim() != 2:
        raise ValueError("h must be (N, C) and weight must be (V, C)")
    if h.shape[1] != weight.shape[1]:
        raise ValueError(f"hidden size mismatch: h {tuple(h.shape)}, weight {tuple(weight.shape)}")
    if targets.shape != (h.shape[0],):
        raise ValueError(f"targets must be (N,) = ({h.shape[0]},), got {tuple(targets.shape)}")


def naive_linear_cross_entropy(
    h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """The foundational path: build all logits, then cross-entropy (mean over valid rows).

    Args:
        h: final hidden states, ``(N, C)``.
        weight: LM-head weight, ``(V, C)`` (the ``nn.Linear(C, V)`` layout).
        targets: next-token ids, ``(N,)`` int64; ``ignore_index`` rows are skipped.

    Returns:
        Scalar loss. Builds the full ``(N, V)`` logits tensor.
    """
    _check_inputs(h, weight, targets)
    acc = _acc_dtype(h)
    logits = h @ weight.t()  # (N, V)
    loss = torch.nn.functional.cross_entropy(logits.to(acc), targets, ignore_index=ignore_index)
    return loss.to(h.dtype)


# --------------------------------------------------------------------------- #
# 1. Chunk over positions, gradients computed in the forward pass
# --------------------------------------------------------------------------- #
class _ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weight, targets, chunk_size, ignore_index):
        N = h.shape[0]
        acc = _acc_dtype(h)
        valid = targets != ignore_index                     # (N,) bool
        n_valid = max(int(valid.sum()), 1)                  # global count, used by every chunk
        safe_t = torch.where(valid, targets, torch.zeros_like(targets))

        loss_sum = torch.zeros((), dtype=acc, device=h.device)
        grad_h = torch.zeros_like(h, dtype=acc)   # (N, C)
        grad_w = torch.zeros_like(weight, dtype=acc)  # (V, C)
        w32 = weight.to(acc)

        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            hc = h[s:e].to(acc)                             # (n, C)
            z = hc @ w32.t()                                # (n, V)  the only big tensor, one chunk
            lse = torch.logsumexp(z, dim=-1)                # (n,)
            rows = torch.arange(e - s, device=h.device)
            z_y = z[rows, safe_t[s:e]]                      # (n,)
            v = valid[s:e].to(acc)                          # (n,)
            loss_sum = loss_sum + ((lse - z_y) * v).sum()
            # dL/dz = (softmax(z) - onehot(y)) * valid / n_valid, built in place over z
            g = torch.exp(z - lse[:, None])                 # (n, V) reuses the chunk's memory budget
            g[rows, safe_t[s:e]] -= 1.0
            g *= (v / n_valid)[:, None]
            grad_h[s:e] = g @ w32                           # (n, V) @ (V, C) -> (n, C)
            grad_w += g.t() @ hc                            # (V, n) @ (n, C) -> (V, C)
            del z, g                                        # the chunk is dropped here

        ctx.save_for_backward(grad_h.to(h.dtype), grad_w.to(weight.dtype))
        return (loss_sum / n_valid).to(h.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        grad_h, grad_w = ctx.saved_tensors
        return grad_h * grad_out, grad_w * grad_out, None, None, None


def chunked_linear_cross_entropy(
    h: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Mean cross-entropy of ``h @ weight.T`` against ``targets``, chunked over positions.

    Never holds more than ``chunk_size`` rows of logits. Gradients for ``h`` and
    ``weight`` are computed inside the forward loop (the logits gradient
    ``softmax - onehot`` is known as soon as the chunk's logits exist), so the
    backward pass only rescales them by the upstream gradient.

    Args:
        h: final hidden states ``(N, C)``, float (fp32/bf16), any device.
        weight: LM-head weight ``(V, C)``, same dtype/device as ``h``.
        targets: ``(N,)`` int64 token ids; ``ignore_index`` rows are skipped.
        chunk_size: rows per chunk ``n``; does not change the result.
        ignore_index: sentinel for masked positions (``-100`` by default).

    Returns:
        Scalar loss in ``h.dtype``: the sum of per-row losses divided by the
        *global* number of valid rows (same normalization as ``F.cross_entropy``).
    """
    _check_inputs(h, weight, targets)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    return _ChunkedLinearCE.apply(h, weight, targets, int(chunk_size), int(ignore_index))


# --------------------------------------------------------------------------- #
# 2. Tile over the vocabulary with an online logsumexp (Cut Cross-Entropy idea)
# --------------------------------------------------------------------------- #
def online_logsumexp_linear(h: torch.Tensor, weight: torch.Tensor, block_v: int = 4096) -> torch.Tensor:
    """``logsumexp(h @ weight.T, dim=-1)`` computed one vocabulary tile at a time.

    Running max ``m`` and running sum ``l = sum exp(z - m)``; when a tile raises the
    max, the old sum is rescaled by ``alpha = exp(m_old - m_new)`` (the online-softmax
    correction of lesson 08.4). Result: ``m + log(l)``.

    Args:
        h: ``(N, C)``; weight: ``(V, C)``; block_v: vocabulary tile width ``Bv``.

    Returns:
        ``(N,)`` tensor of log-normalizers (fp32, or fp64 for fp64 inputs).
    """
    N, V = h.shape[0], weight.shape[0]
    acc = _acc_dtype(h)
    h32 = h.to(acc)
    m = torch.full((N,), float("-inf"), dtype=acc, device=h.device)
    l = torch.zeros((N,), dtype=acc, device=h.device)
    for j in range(0, V, block_v):
        z = h32 @ weight[j:j + block_v].to(acc).t()        # (N, Bv) one tile
        m_new = torch.maximum(m, z.max(dim=-1).values)
        alpha = torch.exp(m - m_new)                        # exp(-inf) = 0 on the first tile
        l = l * alpha + torch.exp(z - m_new[:, None]).sum(dim=-1)
        m = m_new
    return m + torch.log(l)


class _VocabTiledLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weight, targets, block_v, ignore_index):
        acc = _acc_dtype(h)
        valid = targets != ignore_index
        n_valid = max(int(valid.sum()), 1)
        safe_t = torch.where(valid, targets, torch.zeros_like(targets))
        lse = online_logsumexp_linear(h, weight, block_v)          # (N,)
        z_y = (h.to(acc) * weight[safe_t].to(acc)).sum(dim=-1)     # (N,) one dot product per row
        loss = ((lse - z_y) * valid.to(acc)).sum() / n_valid
        ctx.save_for_backward(h, weight, safe_t, valid, lse)
        ctx.block_v, ctx.n_valid = block_v, n_valid
        return loss.to(h.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        h, weight, safe_t, valid, lse = ctx.saved_tensors
        block_v, n_valid = ctx.block_v, ctx.n_valid
        acc = _acc_dtype(h)
        h32 = h.to(acc)
        scale = (valid.to(acc) / n_valid) * grad_out.to(acc)      # (N,)
        grad_h = torch.zeros_like(h32)
        grad_w = torch.zeros(weight.shape, dtype=acc, device=weight.device)
        V = weight.shape[0]
        for j in range(0, V, block_v):
            w_t = weight[j:j + block_v].to(acc)                    # (Bv, C)
            z = h32 @ w_t.t()                                      # (N, Bv) recomputed, as in FlashAttention
            g = torch.exp(z - lse[:, None])                        # softmax restricted to this tile
            in_tile = (safe_t >= j) & (safe_t < j + w_t.shape[0])
            rows = torch.nonzero(in_tile, as_tuple=True)[0]
            g[rows, safe_t[rows] - j] -= 1.0                       # subtract onehot where y is in this tile
            g *= scale[:, None]
            grad_h += g @ w_t
            grad_w[j:j + w_t.shape[0]] = g.t() @ h32
        return grad_h.to(h.dtype), grad_w.to(weight.dtype), None, None, None


def vocab_tiled_linear_cross_entropy(
    h: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    block_v: int = 4096,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Mean cross-entropy with ``logsumexp`` computed online over vocabulary tiles.

    Forward keeps only ``(N,)`` statistics (``lse``) plus the inputs; backward
    recomputes each ``(N, Bv)`` logits tile from ``h`` and ``weight``. Costs one
    extra ``h @ W.T`` matmul in exchange for never storing logits.

    Args:
        h: ``(N, C)``; weight: ``(V, C)``; targets: ``(N,)`` int64.
        block_v: vocabulary tile width ``Bv``; does not change the result.
        ignore_index: sentinel for masked positions.

    Returns:
        Scalar loss in ``h.dtype``, equal to :func:`naive_linear_cross_entropy`.
    """
    _check_inputs(h, weight, targets)
    if block_v < 1:
        raise ValueError("block_v must be >= 1")
    return _VocabTiledLinearCE.apply(h, weight, targets, int(block_v), int(ignore_index))


# --------------------------------------------------------------------------- #
# 3. Scoring: per-token log-probs and entropy (what RL / distillation trainers need)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def fused_logprobs_and_entropy(
    h: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    block_v: int = 4096,
    temperature: float = 1.0,
):
    """Per-token ``log p(y)`` and entropy ``H(p)`` without building ``(N, V)``.

    Online over vocabulary tiles with three running quantities per row: max ``m``,
    sum ``l = sum exp(z - m)``, and ``s = sum exp(z - m) * z``. At the end
    ``lse = m + log l`` and ``H = lse - s / l`` (because ``H = lse - E_p[z]``).
    Logits are divided by ``temperature`` first, as RL trainers do when scoring
    samples that were drawn at that temperature.

    Args:
        h: ``(N, C)``; weight: ``(V, C)``; targets: ``(N,)`` int64 (all valid).
        block_v: vocabulary tile width.
        temperature: positive float.

    Returns:
        ``(logprobs, entropy)``, each ``(N,)`` in the accumulation dtype (fp32; fp64 for fp64 inputs).
    """
    _check_inputs(h, weight, targets)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    N, V = h.shape[0], weight.shape[0]
    acc = _acc_dtype(h)
    h32 = h.to(acc) / temperature
    m = torch.full((N,), float("-inf"), dtype=acc, device=h.device)
    l = torch.zeros((N,), dtype=acc, device=h.device)
    s = torch.zeros((N,), dtype=acc, device=h.device)
    for j in range(0, V, block_v):
        z = h32 @ weight[j:j + block_v].to(acc).t()        # (N, Bv)
        m_new = torch.maximum(m, z.max(dim=-1).values)
        alpha = torch.exp(m - m_new)
        e = torch.exp(z - m_new[:, None])
        l = l * alpha + e.sum(dim=-1)
        s = s * alpha + (e * z).sum(dim=-1)
        m = m_new
    lse = m + torch.log(l)
    z_y = (h32 * weight[targets].to(acc)).sum(dim=-1)
    return z_y - lse, lse - s / l
