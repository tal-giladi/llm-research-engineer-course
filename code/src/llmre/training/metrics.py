"""Training throughput and cost accounting: FLOPs, MFU, tokens per step.

This is the arithmetic of lesson 07.4 (`lessons/module-07/lesson-04.md`). None of
it touches a tensor — these are plain scalar formulas the training loop calls to
turn "wall-clock seconds per step" into the two numbers that actually tell you
whether a run is healthy: **tokens/sec** (raw speed) and **MFU** (how much of the
hardware you are actually using).

Two facts do all the work:

* **The 6N rule.** A forward+backward pass over a transformer costs about
  ``6 * N`` floating-point operations *per token*, where ``N`` is the number of
  (non-embedding) parameters. The 6 splits as 2 for the forward pass and 4 for
  the backward pass (backward computes both the gradient w.r.t. inputs and the
  gradient w.r.t. weights, so it does roughly twice the matmul work of forward).
* **MFU** (model FLOPs utilization) is the achieved FLOPs/s divided by the GPU's
  peak FLOPs/s. It is the fraction of the machine you are really using.

We estimate ``N`` as the standard non-embedding count ``12 * n_layer * n_embd^2``
(the 4·d² of attention's Q/K/V/O projections plus the 8·d² of the MLP's two
Linears, per layer). This deliberately ignores embeddings, LayerNorms and biases:
they are a small fraction of a real model's params and of its FLOPs, and dropping
them is exactly the approximation behind the "6ND" scaling-law bookkeeping.
"""

from __future__ import annotations


def non_embedding_params(config) -> int:
    """Estimate ``N``, the non-embedding parameter count, from a ``GPTConfig``.

    Args:
        config: any object exposing integer ``n_layer`` and ``n_embd`` attributes
            (e.g. :class:`llmre.model.config.GPTConfig`). No tensors, no device.

    Returns:
        ``12 * n_layer * n_embd**2`` as an ``int``. This is the classic
        transformer estimate: per layer, attention's four ``C x C`` projections
        contribute ``4 * C**2`` and the MLP's ``C->4C`` and ``4C->C`` Linears
        contribute ``8 * C**2``, for ``12 * C**2`` per layer. Token/positional
        embeddings, LayerNorm gains and biases are omitted on purpose.
    """
    return 12 * config.n_layer * config.n_embd**2


def model_flops_per_token(config, n_params: int | None = None) -> float:
    """FLOPs for one forward+backward pass, per token, via the 6N rule.

    Args:
        config: a ``GPTConfig``-like object (see :func:`non_embedding_params`).
            Ignored when ``n_params`` is given explicitly.
        n_params: optional explicit parameter count ``N`` to use instead of the
            ``12 * n_layer * n_embd**2`` non-embedding estimate — pass e.g.
            ``model.num_params()`` if you want the total-parameter figure.

    Returns:
        ``6 * N`` as a ``float``: the estimated FLOPs to process one token through
        a forward pass (``2N``) and its backward pass (``4N``). Multiply by the
        number of tokens in a step to get the step's FLOPs.

    Note:
        This is the parameter-only estimate. It omits the attention score/`softmax`
        term (``~6 * n_layer * T * C`` per token for context length ``T``), which
        is small when ``T`` is much less than ``12 * C`` but grows with context
        length — see the lesson for when that correction matters.
    """
    N = n_params if n_params is not None else non_embedding_params(config)
    return 6.0 * N


def tokens_per_step(
    micro_bs: int,
    grad_accum: int,
    world_size: int,
    block_size: int,
) -> int:
    """Number of tokens consumed by one optimizer step.

    Args:
        micro_bs: micro-batch size ``b`` — sequences per forward/backward on one
            worker.
        grad_accum: gradient-accumulation steps ``G`` — micro-batches summed
            before each optimizer step.
        world_size: number of data-parallel workers ``W`` (``1`` on a single
            device).
        block_size: sequence length ``T`` — tokens per sequence.

    Returns:
        ``micro_bs * grad_accum * world_size * block_size`` as an ``int``. The
        global batch size in *sequences* is ``b * G * W``; multiplying by ``T``
        gives tokens per step, the unit scaling laws are written in.
    """
    return micro_bs * grad_accum * world_size * block_size


def mfu(flops_per_token: float, tokens_per_sec: float, peak_flops: float) -> float:
    """Model FLOPs utilization: achieved FLOPs/s as a fraction of hardware peak.

    Args:
        flops_per_token: FLOPs per token (e.g. from :func:`model_flops_per_token`).
        tokens_per_sec: measured training throughput in tokens per second.
        peak_flops: the accelerator's advertised peak throughput in FLOPs/s for
            the dtype in use (e.g. ``312e12`` for A100 bf16 dense).

    Returns:
        ``(flops_per_token * tokens_per_sec) / peak_flops`` as a ``float`` in
        ``[0, 1]`` — the fraction of the machine's arithmetic capacity the run is
        actually using. Large well-tuned pretraining runs commonly land around
        0.3–0.5 (reasonable industry practice, not a fixed law).
    """
    achieved = flops_per_token * tokens_per_sec
    return achieved / peak_flops
