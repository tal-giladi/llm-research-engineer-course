"""LoRA (Low-Rank Adaptation) from scratch — parameter-efficient fine-tuning.

Full fine-tuning updates every weight in the model, and the AdamW optimizer
(Module 3) keeps *two* extra tensors (first and second moment) per trainable
parameter. For GPT-2 small that is ~124M params x 3 = a lot of optimizer memory.
**LoRA** (Hu et al. 2021, https://arxiv.org/abs/2106.09685) freezes the
pretrained weight ``W`` and learns a small low-rank update instead::

    y = W x + (alpha / r) * B (A x)

with ``A`` of shape ``(r, in)`` and ``B`` of shape ``(out, r)``, where the rank
``r`` is tiny (e.g. 8) compared to the layer width. Only ``A`` and ``B`` are
trained, so the optimizer state shrinks by the ratio of full to low-rank params.
It works because fine-tuning updates are empirically close to low-rank: the
delta a task needs lives in a small subspace, so a rank-``r`` factorization
captures most of it. Initialising ``B = 0`` makes the update start at exactly
zero, so a freshly-wrapped layer behaves identically to the pretrained one and
training begins from the pretrained model, not a perturbed one.

This is Module 14, lesson 14.3. It builds on the ``nn.Linear`` of lesson 05.4.
The production library is PEFT; we implement the mechanism first.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap a frozen :class:`torch.nn.Linear` with a trainable low-rank update.

    The wrapped ``base`` linear is frozen (``requires_grad = False`` on its
    weight and bias). Two new parameters carry the update:

        ``A``: shape ``(r, in_features)``, dtype/device of the base weight,
            initialised with Kaiming-uniform noise (as in the reference impl).
        ``B``: shape ``(out_features, r)``, initialised to **zeros** so the
            initial update ``B A`` is exactly zero.

    Forward: ``base(x) + (alpha / r) * F.linear(F.linear(x, A), B)``.

    Args:
        base: a pretrained ``nn.Linear(in_features, out_features)`` to adapt.
        r: LoRA rank, ``1 <= r <= min(in, out)``. Smaller = fewer trainable
            params. Typical values 4-64.
        alpha: LoRA scaling numerator; the update is scaled by ``alpha / r``.
            Decoupling ``alpha`` from ``r`` lets you change ``r`` without
            re-tuning the effective update size.

    Shapes:
        forward input ``x`` is ``(..., in_features)`` float; output is
        ``(..., out_features)`` on the base weight's device.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        if r < 1:
            raise ValueError(f"LoRA rank r must be >= 1, got {r}")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r

        # Freeze the pretrained weight (and bias, if any).
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        w = base.weight
        self.A = nn.Parameter(torch.empty(r, self.in_features, dtype=w.dtype, device=w.device))
        self.B = nn.Parameter(torch.zeros(self.out_features, r, dtype=w.dtype, device=w.device))
        # A ~ Kaiming-uniform (matches the official LoRA implementation); B stays
        # zero so the initial low-rank update B @ A is the zero matrix.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``base(x) + (alpha/r) * B (A x)``; see class docstring for shapes."""
        base_out = self.base(x)                              # (..., out)
        lora_out = F.linear(F.linear(x, self.A), self.B)     # (..., r) -> (..., out)
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        """The effective weight update ``(alpha/r) * B @ A``, shape ``(out, in)``."""
        return self.scaling * (self.B @ self.A)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold ``B A`` into ``W`` and return a plain, standalone ``nn.Linear``.

        Produces a fresh ``nn.Linear(in, out)`` whose weight is
        ``base.weight + (alpha/r) * B @ A`` and whose bias is a copy of the base
        bias. The returned linear reproduces :meth:`forward` (to floating-point
        precision) with zero LoRA overhead — used to ship a merged model for
        inference so it costs exactly what the base model costs.

        Returns:
            A new ``nn.Linear`` on the base weight's device/dtype.
        """
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.weight.copy_(self.base.weight + self.delta_weight())
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze every parameter except the ``A``/``B`` of each :class:`LoRALinear`.

    Call this after wrapping some of a model's linears in :class:`LoRALinear` to
    guarantee the optimizer only ever sees the adapter parameters (so no base
    weight is accidentally updated and no optimizer state is spent on it).

    Args:
        model: any module tree containing ``LoRALinear`` submodules. Mutated in
            place: sets ``requires_grad`` appropriately on every parameter.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad_(True)
            module.B.requires_grad_(True)
