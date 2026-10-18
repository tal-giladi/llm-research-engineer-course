"""Muon from scratch — momentum, orthogonalized by Newton-Schulz, for 2-D weights.

This is the optimizer of lesson 03.4 (`lessons/module-03/lesson-04.md`, added
2026-10-18). It **extends** the two optimizers you already built and changes
neither of them:

* ``llmre.optim.sgd.SGD`` (lesson 03.1) supplies the momentum buffer. Muon keeps
  exactly one buffer per matrix, like momentum-SGD.
* ``llmre.optim.adamw.AdamW`` (lesson 03.2) still trains every parameter that is
  not a hidden weight matrix (embeddings, the LM head, biases, norm gains).
  ``MuonWithAuxAdamW`` below composes the two; that hybrid is what real runs use.

The per-matrix update, in the order ``torch.optim.Muon`` applies it:

    B  <-  mu * B + (1 - mu) * G                  # momentum buffer (an EMA)
    U  =   (1 - mu) * G + mu * B   if nesterov    # look-ahead blend
           B                       otherwise
    O  =   NS_k(U)                                # ~ orthogonal factor of U
    W  <-  W * (1 - lr * weight_decay)            # decoupled weight decay
    W  <-  W - lr * scale(shape) * O              # step

``NS_k`` is ``k`` iterations of the quintic Newton-Schulz map
``X <- a X + (b A + c A^2) X`` with ``A = X X^T``, started from
``X = U / ||U||_F``. It pushes every singular value of ``U`` towards 1 without
an SVD, using only matrix multiplications. With the tuned coefficients
``(3.4445, -4.7750, 2.0315)`` it does *not* converge to exactly 1; after five
steps the singular values land roughly in ``[0.68, 1.13]``. That is on purpose
(see the lesson, section 3.3).

``orthogonalize_svd`` is the exact reference (``U V^T`` from an SVD) and the
cubic iteration ``newton_schulz(..., coeffs=CUBIC_COEFFS)`` is the classical,
convergent one; both exist for teaching and testing.

Everything here is plain PyTorch on any device; no fused kernels.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from llmre.optim.adamw import AdamW

#: Keller Jordan's tuned quintic coefficients (also ``torch.optim.Muon``'s defaults).
QUINTIC_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
#: The classical cubic Newton-Schulz polar iteration, X <- 1.5 X - 0.5 X X^T X.
CUBIC_COEFFS: tuple[float, float, float] = (1.5, -0.5, 0.0)


def newton_schulz(
    G: torch.Tensor,
    steps: int = 5,
    coeffs: tuple[float, float, float] = QUINTIC_COEFFS,
    eps: float = 1e-7,
    dtype: torch.dtype | None = torch.bfloat16,
) -> torch.Tensor:
    """Approximately orthogonalize a matrix with an odd-polynomial iteration.

    Args:
        G: ``(m, n)`` float tensor, any device.
        steps: number of iterations ``k``.
        coeffs: ``(a, b, c)`` of the map ``x -> a x + b x^3 + c x^5`` applied to
            every singular value. ``QUINTIC_COEFFS`` is fast and approximate;
            ``CUBIC_COEFFS`` converges to exactly 1 but needs more steps.
        eps: floor on the Frobenius norm, so a zero matrix stays zero.
        dtype: dtype the iteration runs in. ``torch.bfloat16`` matches
            ``torch.optim.Muon``; ``None`` keeps ``G.dtype`` (use float64 to
            study convergence without rounding noise).

    Returns:
        ``(m, n)`` tensor with ``G``'s dtype and device, with the same singular
        vectors as ``G`` and singular values pushed towards 1.

    The iteration always works on the "wide" orientation (rows <= columns) so
    the Gram matrix ``A = X X^T`` is the smaller ``(min, min)`` one.
    """
    if G.ndim != 2:
        raise ValueError(f"newton_schulz needs a 2-D matrix, got shape {tuple(G.shape)}")
    a, b, c = coeffs
    X = G if dtype is None else G.to(dtype)
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    # Frobenius norm >= spectral norm, so after this every singular value is <= 1,
    # inside the interval where the polynomial is designed to act.
    X = X / X.norm().clamp(min=eps)
    for _ in range(steps):
        A = X @ X.T                    # (r, r) Gram matrix, r = min(m, n)
        # torch.addmm(M, P, Q, beta=s, alpha=t) is s*M + t*(P @ Q) in ONE op with
        # one rounding. The readable form of the next two lines is
        #   B = b * A + c * (A @ A);  X = a * X + B @ X
        # and it is the same mathematics. We use addmm so that, in bfloat16,
        # the rounding matches torch.optim.Muon and the test can compare tightly.
        B = torch.addmm(A, A, A, beta=b, alpha=c)   # (r, r)
        X = torch.addmm(X, B, X, beta=a)            # (r, n): x -> a x + b x^3 + c x^5
    if tall:
        X = X.T
    return X.to(G.dtype)


def orthogonalize_svd(G: torch.Tensor) -> torch.Tensor:
    """Exact orthogonal polar factor ``U V^T`` of ``G = U S V^T`` (reduced SVD).

    The answer Newton-Schulz approximates. ``(m, n)`` in, ``(m, n)`` out, same
    dtype/device. Singular values of the result are exactly 1 on ``G``'s rank
    (directions with ``S = 0`` are not defined and should not be relied on).
    """
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    return U @ Vh


def lr_scale(shape: torch.Size | tuple[int, int], mode: str | None = "original") -> float:
    """Shape-dependent learning-rate multiplier for a ``(A, B)`` = (fan_out, fan_in) matrix.

    * ``"original"`` — ``sqrt(max(1, A / B))`` (Keller Jordan's Muon).
    * ``"match_rms_adamw"`` — ``0.2 * sqrt(max(A, B))`` (Moonlight / Kimi K2).
      An orthogonal ``(A, B)`` matrix has RMS ``1 / sqrt(max(A, B))``, so this
      makes the update's RMS 0.2, about AdamW's, and AdamW's lr / weight decay
      can be reused.
    * ``"spectral_unclamped"`` — ``sqrt(A / B)``.
    * ``None`` — 1.0 (no adjustment).
    Same formulas as ``torch.optim.Muon``'s ``adjust_lr_fn``.
    """
    A, B = shape[0], shape[1]
    if mode == "original":
        return math.sqrt(max(1.0, A / B))
    if mode == "match_rms_adamw":
        return 0.2 * math.sqrt(max(A, B))
    if mode == "spectral_unclamped":
        return math.sqrt(A / B)
    if mode is None:
        return 1.0
    raise ValueError(f"unknown lr_scale mode {mode!r}")


def newton_schulz_flops(m: int, n: int, steps: int = 5) -> int:
    """Matmul FLOPs of ``steps`` quintic iterations on an ``(m, n)`` matrix.

    With ``r = min(m, n)``, ``s = max(m, n)`` each iteration does
    ``X X^T`` (2 r^2 s), ``A @ A`` (2 r^3) and ``B @ X`` (2 r^2 s), i.e.
    ``4 r^2 s + 2 r^3``. A square ``d x d`` matrix costs ``6 d^3`` per iteration.
    """
    r, s = min(m, n), max(m, n)
    return steps * (4 * r * r * s + 2 * r ** 3)


class Muon:
    """Momentum orthogonalized by Newton-Schulz, for 2-D parameters only.

    Args:
        params: iterable of 2-D ``torch.Tensor`` leaves (hidden weight matrices).
        lr: learning rate (float, > 0).
        momentum: ``mu`` of the EMA buffer (default 0.95).
        nesterov: use the look-ahead blend ``(1 - mu) G + mu B`` (default True).
        weight_decay: decoupled decay coefficient, applied as
            ``W *= 1 - lr * weight_decay`` (unscaled lr, as in torch).
        ns_steps: Newton-Schulz iterations (default 5).
        ns_coeffs: polynomial coefficients (default ``QUINTIC_COEFFS``).
        ns_dtype: dtype of the iteration (default bfloat16, like torch).
        adjust_lr: ``lr_scale`` mode (default ``"original"``).
        orthogonalizer: ``"newton_schulz"`` (default) or ``"svd"`` (exact
            ``U V^T``; slow, for experiments and tests).

    State per parameter: one momentum buffer, same shape/dtype/device as the
    parameter (half of AdamW's two).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        ns_coeffs: tuple[float, float, float] = QUINTIC_COEFFS,
        ns_dtype: torch.dtype | None = torch.bfloat16,
        adjust_lr: str | None = "original",
        orthogonalizer: str = "newton_schulz",
    ) -> None:
        self.params = list(params)
        for p in self.params:
            if p.ndim != 2:
                raise ValueError(
                    f"Muon only handles 2-D parameters, got shape {tuple(p.shape)}; "
                    "give it to AdamW instead (see MuonWithAuxAdamW)"
                )
        if orthogonalizer not in {"newton_schulz", "svd"}:
            raise ValueError(f"unknown orthogonalizer {orthogonalizer!r}")
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.ns_coeffs = ns_coeffs
        self.ns_dtype = ns_dtype
        self.adjust_lr = adjust_lr
        self.orthogonalizer = orthogonalizer
        self.bufs = [torch.zeros_like(p) for p in self.params]

    def orthogonalize(self, U: torch.Tensor) -> torch.Tensor:
        """Apply the configured orthogonalizer to one ``(m, n)`` update."""
        if self.orthogonalizer == "svd":
            return orthogonalize_svd(U)
        return newton_schulz(U, self.ns_steps, self.ns_coeffs, dtype=self.ns_dtype)

    @torch.no_grad()
    def step(self) -> None:
        """One Muon update of every parameter, in place. Call after ``backward()``."""
        for p, buf in zip(self.params, self.bufs):
            if p.grad is None:
                continue
            g = p.grad
            # 1) Momentum as an EMA: buf <- mu * buf + (1 - mu) * g.
            buf.lerp_(g, 1.0 - self.momentum)
            # 2) Nesterov look-ahead: (1 - mu) * g + mu * buf.
            update = g.lerp(buf, self.momentum) if self.nesterov else buf
            # 3) Replace the update by (approximately) its orthogonal factor.
            O = self.orthogonalize(update)
            # 4) Decoupled weight decay, then the shape-scaled step.
            if self.weight_decay != 0.0:
                p.mul_(1.0 - self.lr * self.weight_decay)
            p.add_(O, alpha=-self.lr * lr_scale(p.shape, self.adjust_lr))

    @torch.no_grad()
    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def split_params_for_muon(model: torch.nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split a model's parameters into ``(muon_params, adamw_params)``.

    Muon gets the hidden 2-D weight matrices. AdamW gets everything else:
    1-D tensors (biases, LayerNorm/RMSNorm gains) and every ``nn.Embedding``
    or LM-head weight, which are lookup tables / output classifiers rather than
    linear maps between hidden spaces. Shared (tied) tensors appear once, so the
    GPT's tied ``wte``/``lm_head`` weight goes to AdamW exactly once.
    """
    excluded: set[int] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) or name.split(".")[-1] == "lm_head":
            for p in module.parameters(recurse=False):
                excluded.add(id(p))
    muon, adamw, seen = [], [], set()
    for p in model.parameters():  # .parameters() already deduplicates tied tensors
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        if p.ndim == 2 and id(p) not in excluded:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


class MuonWithAuxAdamW:
    """The hybrid real runs use: Muon for hidden matrices, AdamW for the rest.

    Args:
        model: the ``nn.Module`` to optimize; split with ``split_params_for_muon``.
        muon_kwargs: keyword arguments for ``Muon`` (e.g. ``lr=0.02``).
        adamw_kwargs: keyword arguments for ``llmre.optim.adamw.AdamW``
            (``lr`` is required there, e.g. ``lr=3e-4``).
    """

    def __init__(self, model: torch.nn.Module, muon_kwargs: dict | None = None,
                 adamw_kwargs: dict | None = None) -> None:
        muon_params, adamw_params = split_params_for_muon(model)
        self.muon = Muon(muon_params, **(muon_kwargs or {}))
        self.adamw = AdamW(adamw_params, **({"lr": 3e-4} | (adamw_kwargs or {})))

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def zero_grad(self) -> None:
        self.muon.zero_grad()
        self.adamw.zero_grad()
