"""Post-training weight quantization: scales, zero-points, RTN, GPTQ, AWQ, MXFP4.

Lesson 08.7 (added 2026-11-15 by the weekly curriculum review). This module extends two
earlier lessons without changing them:

* 08.3 (mixed precision, :mod:`llmre.training`): there, *training* keeps an fp32 master copy
  and runs matmuls in bf16/fp16. Here, a finished model's weights are stored in 8, 4 or 3
  bits *after* training, with no gradient step at all.
* 08.6 (:mod:`llmre.model.serving`): decode is memory-bound, so its time is roughly
  "bytes of weights read per step / bandwidth". Fewer bits per weight means fewer bytes.

Contents, in lesson order:

* :func:`quantize_symmetric`, :func:`quantize_asymmetric`, :func:`dequantize` — integer
  grids with a scale (and a zero-point), per tensor, per output channel, or per group of
  input columns (:func:`quantize_groupwise`).
* :func:`layer_output_error` — the objective every method below minimizes:
  ``|| X W^T - X Wq^T ||_F^2`` on calibration inputs ``X``, not the weight error itself.
* :func:`gptq_quantize` — quantize one input column at a time and push each column's
  rounding error onto the columns not yet quantized, using the inverse Hessian
  ``(X^T X)^{-1}`` (Frantar et al. 2023, built on Optimal Brain Quantization).
* :func:`awq_quantize` — scale input channels by ``mean|x|^alpha`` before rounding so the
  channels that see large activations lose less precision (Lin et al. 2024).
* :func:`mxfp4_quantize` — the OCP microscaling FP4 format (blocks of 32 E2M1 values that
  share one power-of-two scale), the format used by quantization-aware-trained releases.
* :func:`quantized_weight_bytes` — storage cost including scales.
* :func:`collect_linear_inputs`, :func:`quantize_gpt_weights` — apply any of the above to
  every ``nn.Linear`` inside the course :class:`~llmre.model.gpt.GPT` blocks, layer by
  layer, and return a *new* model (the original is untouched). Weights are stored back in
  float ("fake quantization"): the values are exactly the dequantized grid values, which
  is what a W4A16 kernel computes with, without the packed-int storage.

Everything is plain PyTorch on CPU. Production tools (GPTQModel, AutoAWQ, llm-compressor,
bitsandbytes, llama.cpp) implement the same arithmetic with packed storage and fused
dequantize-matmul kernels.
"""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Integer grids
# --------------------------------------------------------------------------------------


def _check_bits(bits: int) -> None:
    if not isinstance(bits, int) or not 2 <= bits <= 16:
        raise ValueError(f"bits must be an int in [2, 16], got {bits!r}")


def quantize_symmetric(w: torch.Tensor, bits: int, dim: int | None = None):
    """Symmetric (absmax) quantization to signed integers.

    ``scale = max|w| / qmax`` with ``qmax = 2**(bits-1) - 1``; ``q = clamp(round(w/scale))``.
    ``dim=None`` uses one scale for the whole tensor; ``dim=d`` reduces ``max|w|`` over
    dimension ``d`` only, e.g. ``dim=1`` on an ``(out, in)`` weight gives one scale per
    output channel (row).

    Returns ``(q, scale)``: ``q`` has ``w``'s shape and integer values stored as float
    (so it can be inspected and dequantized without dtype juggling); ``scale`` broadcasts
    against ``w``.
    """
    _check_bits(bits)
    qmax = 2 ** (bits - 1) - 1
    if dim is None:
        amax = w.abs().max()
    else:
        amax = w.abs().amax(dim=dim, keepdim=True)
    scale = amax / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q, scale


def quantize_asymmetric(w: torch.Tensor, bits: int, dim: int | None = None):
    """Asymmetric (min/max) quantization to unsigned integers with a zero-point.

    The range is widened to include 0 so that 0 is exactly representable (padding and
    ReLU zeros stay exact). ``scale = (max - min) / (2**bits - 1)``,
    ``zero = round(-min / scale)``, ``q = clamp(round(w / scale) + zero, 0, 2**bits - 1)``.

    Returns ``(q, scale, zero)``; dequantize with ``(q - zero) * scale``.
    """
    _check_bits(bits)
    qmax = 2**bits - 1
    if dim is None:
        lo, hi = w.min(), w.max()
    else:
        lo, hi = w.amin(dim=dim, keepdim=True), w.amax(dim=dim, keepdim=True)
    lo = torch.clamp(lo, max=0.0)
    hi = torch.clamp(hi, min=0.0)
    scale = (hi - lo) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zero = torch.round(-lo / scale)
    q = torch.clamp(torch.round(w / scale) + zero, 0, qmax)
    return q, scale, zero


def dequantize(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor | None = None) -> torch.Tensor:
    """Map integer codes back to real values: ``(q - zero) * scale`` (``zero=0`` if None)."""
    if zero is None:
        return q * scale
    return (q - zero) * scale


def fake_quantize(w: torch.Tensor, bits: int, dim: int | None = None, symmetric: bool = True) -> torch.Tensor:
    """``dequantize(quantize(w))``: the real values a quantized weight actually represents."""
    if symmetric:
        q, s = quantize_symmetric(w, bits, dim)
        return dequantize(q, s)
    q, s, z = quantize_asymmetric(w, bits, dim)
    return dequantize(q, s, z)


def quantize_groupwise(w: torch.Tensor, bits: int, group_size: int, symmetric: bool = True) -> torch.Tensor:
    """Round-to-nearest with one scale per (output row, group of ``group_size`` input columns).

    ``w`` is an ``(out, in)`` weight; ``in`` must be divisible by ``group_size``. Returns the
    dequantized weight (same shape). ``group_size == in`` is per-channel quantization.
    """
    if w.dim() != 2:
        raise ValueError("expected a 2-D (out, in) weight")
    out_f, in_f = w.shape
    if group_size <= 0 or in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={group_size}")
    g = w.reshape(out_f, in_f // group_size, group_size)
    return fake_quantize(g, bits, dim=2, symmetric=symmetric).reshape(out_f, in_f)


def rtn_quantize(w: torch.Tensor, bits: int, group_size: int | None = None, symmetric: bool = True) -> torch.Tensor:
    """Round-to-nearest baseline: per output channel (``group_size=None``) or groupwise."""
    if group_size is None:
        return fake_quantize(w, bits, dim=1, symmetric=symmetric)
    return quantize_groupwise(w, bits, group_size, symmetric)


# --------------------------------------------------------------------------------------
# The objective: layer output error on calibration data
# --------------------------------------------------------------------------------------


def layer_output_error(w: torch.Tensor, w_q: torch.Tensor, x: torch.Tensor) -> float:
    """``|| x w^T - x w_q^T ||_F^2`` for an ``(out, in)`` weight and ``(N, in)`` inputs.

    Equals ``sum_r d_r^T (x^T x) d_r`` over rows ``d_r`` of ``w_q - w``: the Hessian
    ``H = x^T x`` decides which weight errors matter.
    """
    d = (w_q - w).double()
    xd = x.double() @ d.T
    return float((xd**2).sum())


# --------------------------------------------------------------------------------------
# GPTQ
# --------------------------------------------------------------------------------------


def _column_quantizer(w_cols: torch.Tensor, bits: int, symmetric: bool):
    """Return a function that quantizes a column with scales fit on ``w_cols`` (out, k)."""
    if symmetric:
        _, s = quantize_symmetric(w_cols, bits, dim=1)
        qmax = 2 ** (bits - 1) - 1
        s = s[:, 0]
        return lambda col: torch.clamp(torch.round(col / s), -qmax - 1, qmax) * s
    _, s, z = quantize_asymmetric(w_cols, bits, dim=1)
    qmax = 2**bits - 1
    s, z = s[:, 0], z[:, 0]
    return lambda col: (torch.clamp(torch.round(col / s) + z, 0, qmax) - z) * s


def gptq_quantize(
    w: torch.Tensor,
    x: torch.Tensor | None = None,
    bits: int = 4,
    group_size: int | None = None,
    symmetric: bool = True,
    damp: float = 0.01,
    hessian: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """GPTQ: quantize columns left to right, compensating each error on later columns.

    Args:
        w: ``(out, in)`` weight (an ``nn.Linear.weight``).
        x: ``(N, in)`` calibration inputs to this layer; ``H = x^T x``. Or pass ``hessian``.
        bits, symmetric: the integer grid.
        group_size: None for one scale per output row (fit on the original row); otherwise a
            new scale is fit every ``group_size`` columns on the *already updated* weights,
            as in the reference implementation.
        damp: adds ``damp * mean(diag H)`` to the diagonal so ``H`` is invertible.
        scale: optional fixed ``(out,)`` symmetric step size (used by the lesson's hand
            example, where the grid is "multiples of 0.5").

    Returns the dequantized ``(out, in)`` weight, same dtype as ``w``.

    Algorithm (Frantar et al. 2023, Alg. 1 without lazy batching): with ``U`` the upper
    Cholesky factor of ``H^{-1}``, for each column ``i``:
    ``q_i = quant(w_i)``; ``e = (w_i - q_i) / U[i, i]``; ``w[:, j] -= e * U[i, j]`` for
    ``j > i``. If ``H`` is diagonal, ``U[i, j>i] = 0`` and this is exactly round-to-nearest.
    """
    if (x is None) == (hessian is None):
        raise ValueError("pass exactly one of x or hessian")
    W = w.detach().double().clone()
    out_f, in_f = W.shape
    H = (x.double().T @ x.double()) if hessian is None else hessian.double().clone()
    if H.shape != (in_f, in_f):
        raise ValueError(f"Hessian shape {tuple(H.shape)} does not match in_features={in_f}")
    if group_size is not None and in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={group_size}")

    # Inputs that are always zero carry no information: fix their weights to 0.
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    if damp > 0:
        H += damp * torch.diag(H).mean() * torch.eye(in_f, dtype=H.dtype)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    if scale is not None:
        s = scale.double().reshape(out_f)
        qmax = 2 ** (bits - 1) - 1
        quant: Callable = lambda col: torch.clamp(torch.round(col / s), -qmax - 1, qmax) * s
    elif group_size is None:
        quant = _column_quantizer(W, bits, symmetric)

    Q = torch.zeros_like(W)
    for i in range(in_f):
        if scale is None and group_size is not None and i % group_size == 0:
            quant = _column_quantizer(W[:, i : i + group_size], bits, symmetric)
        col = W[:, i]
        q = quant(col)
        Q[:, i] = q
        err = (col - q) / U[i, i]
        W[:, i + 1 :] -= err[:, None] * U[i, i + 1 :][None, :]
    return Q.to(w.dtype)


# --------------------------------------------------------------------------------------
# AWQ
# --------------------------------------------------------------------------------------


def awq_scales(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-input-channel scales ``s = mean|x|^alpha``, normalized so ``max(s) * min(s) = 1``."""
    act = x.detach().abs().double().mean(dim=0).clamp(min=1e-8)
    s = act**alpha
    return s / torch.sqrt(s.max() * s.min())


def awq_quantize(
    w: torch.Tensor,
    x: torch.Tensor,
    bits: int = 4,
    group_size: int | None = None,
    symmetric: bool = True,
    n_grid: int = 20,
):
    """Activation-aware weight quantization (Lin et al. 2024), the scale search only.

    For ``alpha`` in ``{0, 1/n_grid, ..., (n_grid-1)/n_grid}``: scale weight column ``j`` up
    by ``s_j``, round, scale back down: ``W_eff = RTN(W * s) / s``. Keep the ``alpha`` with
    the smallest :func:`layer_output_error`. In a deployed model the ``1/s`` is folded into
    the previous operation (a norm's gain or the previous linear layer), so it costs nothing
    at inference; here we keep it in ``W_eff``, which is mathematically the same layer.

    ``alpha = 0`` gives ``s = 1``, i.e. exactly :func:`rtn_quantize`, so the search can never
    do worse than RTN on the calibration set.

    Returns ``(w_eff, best_alpha)``.
    """
    W = w.detach().double()
    best = None
    for k in range(n_grid):
        alpha = k / n_grid
        s = awq_scales(x, alpha)
        w_eff = rtn_quantize(W * s[None, :], bits, group_size, symmetric) / s[None, :]
        err = layer_output_error(W, w_eff, x)
        if best is None or err < best[0]:
            best = (err, alpha, w_eff)
    return best[2].to(w.dtype), best[1]


# --------------------------------------------------------------------------------------
# MXFP4 (OCP Microscaling): a block floating-point format
# --------------------------------------------------------------------------------------

#: The non-negative values an FP4 E2M1 element can hold.
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def mxfp4_quantize(x: torch.Tensor, block: int = 32):
    """Quantize the last dimension in blocks of ``block`` values sharing one power-of-two scale.

    OCP MX v1.0: shared exponent ``e = floor(log2(max|x|)) - 2`` (2 is the largest exponent
    of E2M1, whose largest value is ``6 = 1.5 * 2**2``), stored as an 8-bit exponent
    (E8M0); each element is ``x / 2**e`` rounded to the nearest E2M1 value, saturating at
    ±6. Returns ``(dequantized, scales)`` with ``scales`` of shape ``(..., n_blocks)``.
    """
    if x.shape[-1] % block != 0:
        raise ValueError(f"last dim {x.shape[-1]} not divisible by block={block}")
    shp = x.shape
    xb = x.double().reshape(*shp[:-1], shp[-1] // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    scale = torch.where(amax > 0, 2.0 ** (torch.floor(torch.log2(safe)) - 2), torch.ones_like(amax))
    y = (xb / scale).clamp(-6.0, 6.0)
    grid = E2M1_VALUES.double()
    idx = (y.abs().unsqueeze(-1) - grid).abs().argmin(dim=-1)
    yq = torch.sign(y) * grid[idx]
    deq = (yq * scale).reshape(shp).to(x.dtype)
    return deq, scale.squeeze(-1)


# --------------------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------------------


def quantized_weight_bytes(
    out_features: int,
    in_features: int,
    bits: int,
    group_size: int | None = None,
    asymmetric: bool = False,
    scale_bytes: int = 2,
) -> float:
    """Bytes to store one quantized ``(out, in)`` weight: packed codes + one fp16 scale per
    group (+ one fp16 zero-point per group if ``asymmetric``). ``group_size=None`` means
    one group per output row. Real formats differ in small ways (some pack zero-points in
    ``bits`` bits); this is the first-order cost."""
    n = out_features * in_features
    groups = out_features if group_size is None else n // group_size
    per_group = scale_bytes * (2 if asymmetric else 1)
    return n * bits / 8 + groups * per_group


# --------------------------------------------------------------------------------------
# Applying it to the course GPT
# --------------------------------------------------------------------------------------


def _block_linears(model: nn.Module) -> list[str]:
    """Names of the ``nn.Linear`` modules inside the transformer blocks, in forward order.

    The token embedding and the tied LM head are deliberately left in full precision, as
    most weight-only PTQ recipes do: they are a small share of a large model's weights and
    the most sensitive to error.
    """
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and ".h." in f".{n}"]


@torch.no_grad()
def collect_linear_inputs(model: nn.Module, idx: torch.Tensor, names: list[str]) -> dict[str, torch.Tensor]:
    """Run ``model(idx)`` and return each named Linear's input, flattened to ``(B*T, in)``."""
    mods = dict(model.named_modules())
    store: dict[str, torch.Tensor] = {}
    hooks = []
    for n in names:
        def hook(mod, inp, out, n=n):
            store[n] = inp[0].detach().reshape(-1, inp[0].shape[-1]).clone()
        hooks.append(mods[n].register_forward_hook(hook))
    try:
        model(idx)
    finally:
        for h in hooks:
            h.remove()
    return store


@torch.no_grad()
def quantize_gpt_weights(
    model: nn.Module,
    calib_idx: torch.Tensor,
    method: str = "rtn",
    bits: int = 4,
    group_size: int | None = None,
    symmetric: bool = True,
) -> nn.Module:
    """Return a deep copy of ``model`` whose block Linear weights are quantized.

    ``method`` is ``"rtn"``, ``"gptq"`` or ``"awq"``. Layers are processed in forward order
    and each layer's calibration inputs are recomputed from the partially quantized copy, so
    later layers compensate for the errors of earlier ones (the sequential scheme of the
    GPTQ reference code). ``rtn`` needs no calibration data but accepts it for a uniform API.
    """
    if method not in {"rtn", "gptq", "awq"}:
        raise ValueError(f"unknown method {method!r}")
    qmodel = copy.deepcopy(model).eval()
    mods = dict(qmodel.named_modules())
    for name in _block_linears(qmodel):
        lin = mods[name]
        w = lin.weight.data
        if method == "rtn":
            wq = rtn_quantize(w, bits, group_size, symmetric)
        else:
            x = collect_linear_inputs(qmodel, calib_idx, [name])[name]
            if method == "gptq":
                wq = gptq_quantize(w, x, bits=bits, group_size=group_size, symmetric=symmetric)
            else:
                wq, _ = awq_quantize(w, x, bits=bits, group_size=group_size, symmetric=symmetric)
        lin.weight.data.copy_(wq)
    return qmodel


__all__ = [
    "E2M1_VALUES",
    "awq_quantize",
    "awq_scales",
    "collect_linear_inputs",
    "dequantize",
    "fake_quantize",
    "gptq_quantize",
    "layer_output_error",
    "mxfp4_quantize",
    "quantize_asymmetric",
    "quantize_gpt_weights",
    "quantize_groupwise",
    "quantize_symmetric",
    "quantized_weight_bytes",
    "rtn_quantize",
]
