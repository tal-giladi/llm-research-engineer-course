"""Language-modeling evaluation metrics, implemented from scratch.

This module implements the two quantities every language model is judged by
during pretraining: the **cross-entropy loss** (the training signal) and its
exponential, **perplexity** (the human-readable "effective branching factor").

Both are implemented directly on top of a numerically stable log-softmax so you
can see exactly what ``torch.nn.functional.cross_entropy`` computes. The tests
in ``code/tests/test_metrics.py`` assert that this from-scratch version matches
PyTorch's fused kernel to within 1e-5.

Referenced from lesson 01.4 (`lessons/module-01/lesson-04.md`).
"""

from __future__ import annotations

import torch


def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable log of the softmax, over the last dimension.

    Softmax turns a row of real-valued logits into a probability distribution:
    ``softmax(z)_i = exp(z_i) / sum_j exp(z_j)``. Taking the log of that gives
    ``log_softmax(z)_i = z_i - logsumexp(z)``. We compute it in log space and
    subtract the per-row max before exponentiating, so no ``exp`` ever sees a
    large positive number and overflows to ``inf``. Subtracting a constant ``m``
    from every logit leaves the softmax unchanged (the constant cancels between
    numerator and denominator), so this is exact, not an approximation.

    Args:
        logits: shape ``(N, V)``, dtype float (float32/float64), any device.
            ``N`` rows (e.g. token positions), ``V`` classes (the vocabulary).

    Returns:
        Tensor of shape ``(N, V)``, same dtype and device as ``logits``, whose
        rows are log-probabilities (``exp`` of each row sums to 1).
    """
    m = logits.max(dim=-1, keepdim=True).values          # (N, 1): per-row max
    shifted = logits - m                                 # (N, V): max is now 0
    # logsumexp over the shifted logits, per row -> (N, 1)
    lse = shifted.exp().sum(dim=-1, keepdim=True).log()
    return shifted - lse                                 # (N, V): log-probs


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy in **nats**, from scratch (no ``F.cross_entropy``).

    For each row ``n`` the model outputs logits over the vocabulary; the true
    next token is the integer ``targets[n]``. The per-position loss is the
    negative log-probability the model assigned to that true token,
    ``-log q(targets[n])``, i.e. cross-entropy between the model's predicted
    distribution and the one-hot true distribution. We return the mean over all
    ``N`` rows. Because we use the natural logarithm the unit is nats, matching
    ``torch.nn.functional.cross_entropy``.

    Args:
        logits: shape ``(N, V)``, dtype float, any device. Raw (unnormalized)
            scores, one row per position, ``V`` = vocabulary size.
        targets: shape ``(N,)``, dtype ``torch.long``, same device as ``logits``.
            Each entry is a class index in ``[0, V)``.

    Returns:
        Scalar tensor (shape ``()``), same dtype/device as ``logits``: the mean
        cross-entropy in nats.

    Note:
        This equals ``torch.nn.functional.cross_entropy(logits, targets)`` (its
        default reduction is also the mean). PyTorch fuses log-softmax and the
        gather into one kernel; the value is identical to what we compute here.
    """
    logp = log_softmax(logits)                           # (N, V)
    n = logits.shape[0]
    rows = torch.arange(n, device=logits.device)         # (N,)
    # Gather the log-prob of the true token in each row -> (N,)
    true_logp = logp[rows, targets]
    return -true_logp.mean()                             # scalar, nats


def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Perplexity = ``exp(cross_entropy)``, the effective branching factor.

    Perplexity is the exponential of the mean cross-entropy (in nats). It reads
    as "on average the model is as uncertain as if it were choosing uniformly
    among this many equally likely tokens". A perfect model scores 1; a uniform
    model over ``V`` tokens scores exactly ``V``.

    Args:
        logits: shape ``(N, V)``, dtype float, any device.
        targets: shape ``(N,)``, dtype ``torch.long``, same device as ``logits``.

    Returns:
        Scalar tensor (shape ``()``): the perplexity (dimensionless).
    """
    return cross_entropy(logits, targets).exp()
