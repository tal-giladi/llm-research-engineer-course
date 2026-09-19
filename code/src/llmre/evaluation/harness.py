"""A small, reproducible evaluation harness for language models.

Evaluation is a first-class research discipline: every experiment must produce
numbers you can reproduce and compare across runs. This module gives you three
building blocks, all implemented from scratch on top of the model's forward pass
(:meth:`llmre.model.gpt.GPT.forward`, which returns ``(logits, loss)``) and the
numerically stable :func:`llmre.evaluation.metrics.log_softmax`:

* :func:`evaluate_perplexity` — the pretraining/held-out metric. Streams a token
  sequence through the model in non-overlapping blocks and returns
  ``exp(mean cross-entropy)`` (see :mod:`llmre.evaluation.metrics`).
* :func:`sequence_loglikelihood` — the total log-probability the model assigns to
  a sequence, ``sum_t log p(x_t | x_<t)``. The primitive behind likelihood-based
  task evaluation.
* :func:`multiple_choice_score` — scores each answer option by the (optionally
  length-normalized) log-likelihood of its continuation and returns the argmax
  option index. This is how zero-/few-shot multiple-choice benchmarks (ARC,
  HellaSwag, MMLU) are graded without any weight updates.

Referenced from lessons 13.1 and 13.3 (``lessons/module-13/``).

Conventions (matching the rest of ``llmre``):

* ``model`` is any callable returning ``(logits, loss)`` where ``logits`` has
  shape ``(B, T, V)`` float; a :class:`llmre.model.gpt.GPT` is the canonical one.
* Token id tensors are ``torch.long``. Everything is done under ``torch.no_grad``
  and with the model in ``eval()`` mode, so no autograd graph is built and dropout
  is disabled — evaluation must be deterministic.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from llmre.evaluation.metrics import log_softmax


def _model_device(model) -> torch.device:
    """Best-effort device of a model's parameters, defaulting to CPU.

    A stub/toy model used in tests may have no parameters; we fall back to CPU.
    """
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


@torch.no_grad()
def evaluate_perplexity(
    model,
    data: torch.Tensor,
    block_size: int,
    batch_size: int = 8,
    max_batches: int | None = None,
) -> float:
    """Perplexity of ``model`` over a held-out token stream.

    The stream ``data`` (a flat sequence of token ids) is cut into
    non-overlapping ``(x, y)`` windows of length ``block_size``: ``x`` is
    ``data[i : i + block_size]`` and ``y`` is the same window shifted by one,
    ``data[i + 1 : i + 1 + block_size]``. Windows are grouped into batches of
    ``batch_size`` and pushed through ``model(x, targets=y)``. The model returns
    the mean cross-entropy (in nats) over that batch's ``B * block_size`` target
    positions; we accumulate a token-weighted mean over all batches and return
    its exponential.

    Args:
        model: callable ``(idx, targets) -> (logits, loss)``; ``loss`` is the
            scalar mean cross-entropy over the batch (as :class:`~llmre.model.gpt.GPT`
            returns). Put in ``eval()`` mode here to disable dropout.
        data: shape ``(L,)`` dtype ``torch.long``, any device. The held-out
            token stream. Must satisfy ``L >= block_size + 1``.
        block_size: context length ``T`` of each window; must be
            ``<= model.cfg.block_size``.
        batch_size: number of windows per forward pass. Only affects speed and
            floating-point summation order, not the mathematical result.
        max_batches: if given, stop after this many batches (a quick estimate on
            a long stream). ``None`` evaluates every full window.

    Returns:
        The perplexity (a Python ``float``): ``exp`` of the mean per-token
        cross-entropy. A uniform model over a vocabulary of size ``V`` scores
        ``V``; a perfect model scores ``1``.

    Shapes:
        Each batch feeds ``x, y`` of shape ``(B, block_size)`` long and consumes
        ``B * block_size`` target tokens toward the mean.
    """
    if data.dim() != 1:
        raise ValueError(f"data must be 1-D, got shape {tuple(data.shape)}")
    device = _model_device(model)
    if hasattr(model, "eval"):
        model.eval()

    n_windows = (data.numel() - 1) // block_size
    if n_windows < 1:
        raise ValueError(
            f"data length {data.numel()} too short for block_size {block_size} "
            f"(need at least block_size + 1 tokens)"
        )

    # Build the (n_windows, block_size) input and target matrices by slicing the
    # stream. x[i] is window i; y[i] is window i shifted one token to the right.
    total_nats = 0.0
    total_tokens = 0
    batches_done = 0

    for start in range(0, n_windows, batch_size):
        stop = min(start + batch_size, n_windows)
        xs, ys = [], []
        for i in range(start, stop):
            off = i * block_size
            xs.append(data[off : off + block_size])
            ys.append(data[off + 1 : off + 1 + block_size])
        x = torch.stack(xs).to(device)          # (B, block_size) long
        y = torch.stack(ys).to(device)          # (B, block_size) long

        _, loss = model(x, y)                   # loss: mean nats over B*block_size
        n_tok = x.numel()                       # B * block_size target positions
        total_nats += float(loss) * n_tok
        total_tokens += n_tok

        batches_done += 1
        if max_batches is not None and batches_done >= max_batches:
            break

    mean_ce = total_nats / total_tokens         # nats per token
    return math.exp(mean_ce)


@torch.no_grad()
def sequence_loglikelihood(model, ids: torch.Tensor) -> float:
    """Total log-probability the model assigns to a sequence, in nats.

    Computes ``sum_{t=1}^{L-1} log p(ids[t] | ids[:t])`` — the log-likelihood of
    every token given its prefix, summed. The first token has no prefix under a
    plain LM, so it is not scored. Feeding the whole sequence once gives all the
    needed conditional distributions in a single forward pass: the logits at
    position ``t`` are the model's prediction for token ``t + 1``.

    Args:
        model: callable ``idx -> (logits, loss)`` with ``logits`` shape
            ``(B, T, V)``.
        ids: shape ``(L,)`` dtype ``torch.long``, ``L >= 2``. A single sequence
            of token ids (no batch dimension).

    Returns:
        The summed next-token log-probability as a Python ``float`` (<= 0). Less
        negative means the model finds the sequence more probable.

    Shapes:
        ``ids`` is unsqueezed to ``(1, L)`` for the forward pass; logits come back
        ``(1, L, V)`` and we read positions ``0 .. L-2`` to predict tokens
        ``1 .. L-1``.
    """
    if ids.dim() != 1:
        raise ValueError(f"ids must be 1-D, got shape {tuple(ids.shape)}")
    if ids.numel() < 2:
        raise ValueError("need at least 2 tokens to score a next-token likelihood")
    if hasattr(model, "eval"):
        model.eval()
    device = _model_device(model)
    ids = ids.to(device)

    logits, _ = model(ids.unsqueeze(0))          # (1, L, V)
    logits = logits[0]                           # (L, V)
    logp = log_softmax(logits[:-1])              # (L-1, V): predict tokens 1..L-1
    targets = ids[1:]                            # (L-1,)
    chosen = logp[torch.arange(targets.numel(), device=device), targets]  # (L-1,)
    return float(chosen.sum())


@torch.no_grad()
def multiple_choice_score(
    model,
    context_ids: torch.Tensor,
    option_ids_list: Sequence[torch.Tensor],
    length_normalize: bool = True,
) -> int:
    """Pick the answer option the model finds most likely, by log-likelihood.

    Multiple-choice benchmarks are graded without any weight update: form the
    prompt (the ``context``) followed by each candidate answer (an ``option``),
    ask the model for the log-probability it assigns to the *option's tokens
    given the context*, and pick the highest. Only the continuation tokens are
    scored — the shared context contributes equally to every option and would
    otherwise just add a constant.

    With ``length_normalize=True`` we divide each option's summed log-probability
    by its token count, giving the mean per-token log-probability. This corrects
    the bias that longer options accumulate more (always negative) log-prob terms
    and would otherwise look less likely purely for being longer. This is the
    "length-normalized log-likelihood" scoring used by lm-eval-harness for tasks
    like HellaSwag and ARC.

    Args:
        model: callable ``idx -> (logits, loss)``, ``logits`` shape ``(B, T, V)``.
        context_ids: shape ``(Lc,)`` dtype ``torch.long``, ``Lc >= 1``. The prompt
            (question, plus any few-shot examples) shared by all options.
        option_ids_list: a sequence of ``(Lo,)`` long tensors, one per candidate
            answer; lengths may differ.
        length_normalize: if ``True`` score by mean per-token log-prob, else by
            the summed log-prob.

    Returns:
        The index (Python ``int``) of the highest-scoring option in
        ``option_ids_list``.

    Shapes:
        For option ``o`` we run the concatenation ``[context; option]`` of shape
        ``(1, Lc + Lo)`` through the model, read logits at positions
        ``Lc-1 .. Lc+Lo-2`` (each predicts one option token), and sum their
        log-probabilities.
    """
    if context_ids.dim() != 1 or context_ids.numel() < 1:
        raise ValueError("context_ids must be a 1-D tensor with at least 1 token")
    if len(option_ids_list) == 0:
        raise ValueError("need at least one option to score")
    if hasattr(model, "eval"):
        model.eval()
    device = _model_device(model)
    context_ids = context_ids.to(device)
    Lc = context_ids.numel()

    scores: list[float] = []
    for option_ids in option_ids_list:
        if option_ids.dim() != 1 or option_ids.numel() < 1:
            raise ValueError("each option must be a 1-D tensor with at least 1 token")
        option_ids = option_ids.to(device)
        seq = torch.cat([context_ids, option_ids]).unsqueeze(0)   # (1, Lc+Lo)
        logits, _ = model(seq)                                    # (1, Lc+Lo, V)
        logits = logits[0]                                        # (Lc+Lo, V)
        # logits at position p predict token p+1, so the option's tokens
        # (absolute positions Lc .. Lc+Lo-1) are predicted by positions
        # Lc-1 .. Lc+Lo-2.
        Lo = option_ids.numel()
        pred = log_softmax(logits[Lc - 1 : Lc - 1 + Lo])          # (Lo, V)
        chosen = pred[torch.arange(Lo, device=device), option_ids]  # (Lo,)
        total = float(chosen.sum())
        scores.append(total / Lo if length_normalize else total)

    return int(max(range(len(scores)), key=lambda i: scores[i]))
