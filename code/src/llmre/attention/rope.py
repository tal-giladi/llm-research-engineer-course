"""Rotary Position Embedding (RoPE) — from scratch (Module 12, lesson 01).

Absolute learned position embeddings (Module 5, ``wpe``) add a per-position
vector to the token embedding *once*, at the input. They have two weaknesses:
they cannot extend past ``block_size`` rows (position 2048 has no learned
vector if you only trained 1024), and the model only ever sees *absolute*
positions, never the *relative* offset between two tokens, which is what most
linguistic relationships actually depend on.

**RoPE** (Su et al. 2021, ``papers/index.md`` #8) fixes both. Instead of adding
anything, it *rotates* each query and key vector by an angle proportional to its
position, in 2-D coordinate pairs, with a different frequency per pair. The key
algebraic fact (derived in the lesson) is:

    <RoPE(q, m), RoPE(k, n)> depends on q, k and (m - n) only.

So the attention score between positions ``m`` and ``n`` is a function of their
*relative* offset ``m - n`` — relative position falls out of applying an
absolute rotation to each side. RoPE touches only Q and K, never V and never the
residual stream.

We implement it the LLaMA way, using complex numbers: a 2-D rotation by angle
``θ`` is multiplication by the unit complex number ``e^{iθ} = cos θ + i sin θ``
(``cis θ``). Packing each adjacent pair of real channels ``(x_0, x_1)`` into one
complex number ``x_0 + i x_1`` turns "rotate every pair" into a single complex
multiply.
"""

from __future__ import annotations

import torch


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute the per-(position, pair) rotation factors ``e^{iθ}``.

    A head vector of size ``dim`` is split into ``dim // 2`` coordinate pairs.
    Pair ``p`` (0-indexed) rotates with angular frequency

        ``ω_p = base ** (-2p / dim)``,

    so pair 0 turns fastest (ω = 1 rad per position) and the last pair turns
    slowest. At position ``m`` the rotation angle for pair ``p`` is ``θ = m·ω_p``,
    and the rotation itself is the unit complex number ``e^{iθ}``.

    Args:
        dim: the per-head dimension ``hd`` (``n_embd // n_head``). Must be even,
            because RoPE rotates the channels two at a time.
        max_seq_len: largest position index ``T`` to precompute rows for.
        base: the geometric base ``θ`` of the frequency schedule (10000 in the
            original paper and in LLaMA). Larger base -> slower rotation ->
            longer effective wavelengths.

    Returns:
        A complex tensor ``freqs_cis`` of shape ``(max_seq_len, dim // 2)`` and
        dtype ``torch.complex64``, on the CPU. Entry ``[m, p]`` is ``e^{i·m·ω_p}``.
        Move it to your model's device once and slice ``freqs_cis[:T]`` per batch.
    """
    assert dim % 2 == 0, "RoPE needs an even head dimension (channels rotate in pairs)"
    # One frequency per pair: ω_p = base^(-2p/dim) for p = 0, 1, ..., dim/2 - 1.
    exponents = torch.arange(0, dim, 2, dtype=torch.float32) / dim  # (dim/2,): 2p/dim
    freqs = 1.0 / (base**exponents)                                 # (dim/2,): ω_p
    # Angles θ[m, p] = m · ω_p for every position m and pair p.
    t = torch.arange(max_seq_len, dtype=torch.float32)             # (max_seq_len,)
    angles = torch.outer(t, freqs)                                 # (max_seq_len, dim/2)
    # e^{iθ} as a complex tensor of unit-modulus numbers (magnitude 1, phase θ).
    freqs_cis = torch.polar(torch.ones_like(angles), angles)       # complex (T, dim/2)
    return freqs_cis


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate queries and keys by their position angles (RoPE).

    Each length-``hd`` head vector is read as ``hd // 2`` complex numbers (adjacent
    channel pairs ``(x_0, x_1) -> x_0 + i x_1``), multiplied by the precomputed
    ``e^{iθ}`` for its position, then read back as real. Multiplying a complex
    number by ``e^{iθ}`` rotates it by ``θ`` in the plane, so this rotates every
    coordinate pair by that pair's position-dependent angle.

    Args:
        q: queries, shape ``(B, nh, T, hd)``, any float dtype, any device.
        k: keys, shape ``(B, nh, T, hd)`` (same nh here; for GQA apply RoPE with
            the key's own head count). Same dtype/device as ``q``.
        freqs_cis: output of :func:`precompute_freqs_cis`, complex, shape
            ``(T_max, hd // 2)``. Only the first ``T`` rows are used; it must live
            on the same device as ``q``.

    Returns:
        ``(q_rot, k_rot)`` with the same shapes, dtypes and device as the inputs.
        Norms are preserved (rotation is length-preserving); only directions
        within each 2-D pair change.
    """
    B, nh, T, hd = q.shape
    assert hd % 2 == 0, "RoPE needs an even head dimension"
    # View the last axis as (hd/2) pairs, then as complex: (B, nh, T, hd/2).
    q_c = torch.view_as_complex(q.float().reshape(B, nh, T, hd // 2, 2))
    k_c = torch.view_as_complex(k.float().reshape(B, nh, T, hd // 2, 2))
    # Broadcast the rotation over batch and head axes: (1, 1, T, hd/2).
    rot = freqs_cis[:T].view(1, 1, T, hd // 2)
    # Complex multiply = rotate each pair by its angle; view_as_real re-interleaves
    # to (B, nh, T, hd/2, 2), flatten(3) merges the pair axis back to hd.
    q_rot = torch.view_as_real(q_c * rot).flatten(3)
    k_rot = torch.view_as_real(k_c * rot).flatten(3)
    return q_rot.type_as(q), k_rot.type_as(k)
