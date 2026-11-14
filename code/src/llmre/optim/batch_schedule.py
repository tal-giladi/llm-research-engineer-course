"""Batch-size warmup ("batch ramp") and the gradient noise scale behind it.

This is section 5 of lesson 03.3 (`lessons/module-03/lesson-03.md`). Many large
pretraining runs start with a small global batch and grow it early in training
(GPT-3, Gopher, Llama 3 405B, DeepSeek-V3). The reason is the *gradient noise
scale* of McCandlish et al. 2018: early in training every example's gradient
points roughly the same way, so a small batch already gives a good direction
and a big batch wastes tokens; later the useful signal is small next to the
per-example noise, so a big batch is needed to average the noise away.

Three pieces:
    batch_size_at          : the schedule, global batch (sequences) vs tokens seen
    grad_accum_for         : how a batch size becomes micro-batches x accumulation
    plan_batch_ramp        : step-by-step plan of a ramped run (for inspection)
    gradient_noise_scale   : estimate B_simple = tr(Sigma) / |G|^2 from two batch sizes
    progress_fraction      : 1 / (1 + B_noise / B), per-step progress vs an infinite batch
"""

from __future__ import annotations


def batch_size_at(
    tokens_seen: int,
    start_batch: int,
    final_batch: int,
    ramp_tokens: int,
    granularity: int = 1,
) -> int:
    """Global batch size (in sequences) after ``tokens_seen`` training tokens.

    Linear ramp in *tokens* (as GPT-3 describes it), rounded down to a multiple
    of ``granularity``, clamped to ``[start_batch, final_batch]``.

    Args:
        tokens_seen: training tokens consumed before this step (int, >= 0).
        start_batch: batch size at token 0, in sequences.
        final_batch: batch size once ``tokens_seen >= ramp_tokens``.
        ramp_tokens: number of tokens over which the batch grows linearly.
        granularity: the batch must be a multiple of this, usually
            ``micro_batch_size * world_size`` so it splits into whole
            micro-batches on every GPU. ``start_batch`` and ``final_batch``
            must already be multiples of it.

    Returns:
        The batch size (int, sequences) to use for the next optimizer step.
    """
    if start_batch % granularity or final_batch % granularity:
        raise ValueError("start_batch and final_batch must be multiples of granularity")
    if tokens_seen >= ramp_tokens:
        return final_batch
    raw = start_batch + (final_batch - start_batch) * tokens_seen / ramp_tokens
    rounded = int(raw // granularity) * granularity  # round DOWN to whole micro-batches
    return max(start_batch, min(final_batch, rounded))


def grad_accum_for(batch_size: int, micro_batch_size: int, world_size: int = 1) -> int:
    """Gradient-accumulation steps ``G`` so that ``b * G * world_size == batch_size``.

    The micro-batch ``b`` is what fits in GPU memory, so it stays fixed; a batch
    ramp is implemented by changing how many micro-batches are summed per step.
    """
    per_step = micro_batch_size * world_size
    if batch_size % per_step:
        raise ValueError(f"batch_size {batch_size} is not a multiple of {per_step}")
    return batch_size // per_step


def plan_batch_ramp(
    total_tokens: int,
    seq_len: int,
    start_batch: int,
    final_batch: int,
    ramp_tokens: int,
    micro_batch_size: int,
    world_size: int = 1,
) -> list[dict]:
    """Simulate a ramped run and return one row per optimizer step.

    Each row: ``step``, ``tokens_seen`` (before the step), ``batch`` (sequences),
    ``grad_accum`` and ``tokens`` (consumed by this step). Pure Python, no
    tensors: print it before training to see exactly what the run will do.
    """
    gran = micro_batch_size * world_size
    rows, tokens_seen, step = [], 0, 0
    while tokens_seen < total_tokens:
        b = batch_size_at(tokens_seen, start_batch, final_batch, ramp_tokens, gran)
        rows.append({
            "step": step,
            "tokens_seen": tokens_seen,
            "batch": b,
            "grad_accum": grad_accum_for(b, micro_batch_size, world_size),
            "tokens": b * seq_len,
        })
        tokens_seen += b * seq_len
        step += 1
    return rows


def gradient_noise_scale(
    g_small_sq: float, g_big_sq: float, b_small: int, b_big: int
) -> tuple[float, float, float]:
    """Estimate the simple gradient noise scale from two batch sizes.

    If ``G`` is the true gradient and ``Sigma`` the per-example gradient
    covariance, a mean gradient over ``B`` examples satisfies
    ``E[|G_B|^2] = |G|^2 + tr(Sigma) / B``. Measuring ``|G_B|^2`` at two batch
    sizes gives two equations in the two unknowns (McCandlish et al. 2018, app. A.1):

        |G|^2     = (b_big * g_big_sq - b_small * g_small_sq) / (b_big - b_small)
        tr(Sigma) = (g_small_sq - g_big_sq) / (1 / b_small - 1 / b_big)

    Args:
        g_small_sq: squared L2 norm of the mean gradient over ``b_small`` examples.
        g_big_sq: squared L2 norm of the mean gradient over ``b_big`` examples.
        b_small, b_big: the two batch sizes, ``b_small < b_big``.

    Returns:
        ``(g_sq, trace_sigma, b_simple)`` with ``b_simple = trace_sigma / g_sq``.
        Single estimates are noisy: average ``g_sq`` and ``trace_sigma`` over
        many steps before dividing.
    """
    g_sq = (b_big * g_big_sq - b_small * g_small_sq) / (b_big - b_small)
    trace_sigma = (g_small_sq - g_big_sq) / (1.0 / b_small - 1.0 / b_big)
    return g_sq, trace_sigma, trace_sigma / g_sq


def progress_fraction(batch_size: float, b_noise: float) -> float:
    """Per-step loss improvement relative to an infinite batch: ``1 / (1 + B_noise / B)``.

    McCandlish et al. 2018 eq. 2.7, assuming the learning rate is tuned for each
    batch size. Near 0 when ``B << B_noise`` (steps mostly noise), near 1 when
    ``B >> B_noise`` (extra examples barely help).
    """
    return 1.0 / (1.0 + b_noise / batch_size)
