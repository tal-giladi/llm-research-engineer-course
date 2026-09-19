"""Calibration: does a model's stated confidence match how often it is right?

A model is **calibrated** if, among all the predictions it makes with confidence
~0.7, about 70% are actually correct. Accuracy tells you *how often* the model is
right; calibration tells you whether you can *trust its probabilities*. They are
independent: a model can be accurate but overconfident, or inaccurate but honest
about it.

The standard scalar summary is the **Expected Calibration Error (ECE)**. Sort the
predictions into ``n_bins`` confidence buckets; in each bucket compare the mean
confidence to the empirical accuracy; average the gaps, weighted by how many
predictions fall in each bucket. Zero means perfectly calibrated.

Referenced from lesson 13.2 (``lessons/module-13/lesson-02.md``).
"""

from __future__ import annotations

import torch


def expected_calibration_error(
    confidences,
    correct,
    n_bins: int = 10,
) -> float:
    r"""Expected Calibration Error over equal-width confidence bins.

    Partition ``[0, 1]`` into ``n_bins`` equal-width buckets. For bucket
    ``b`` with the set of predictions ``B_b`` falling in it, let
    ``acc(B_b)`` be the fraction of those that are correct and ``conf(B_b)`` the
    mean predicted confidence. Then

    .. math::
        \mathrm{ECE} = \sum_{b=1}^{M} \frac{|B_b|}{N}\,
                       \bigl| \mathrm{acc}(B_b) - \mathrm{conf}(B_b) \bigr|,

    where ``N`` is the total number of predictions. Empty buckets contribute 0.

    Args:
        confidences: shape ``(N,)`` (tensor, list, or array) of floats in
            ``[0, 1]`` — the model's confidence in each prediction (typically the
            max softmax probability of the predicted class).
        correct: shape ``(N,)`` of 0/1 or boolean values — whether each
            prediction was correct.
        n_bins: number of equal-width bins across ``[0, 1]`` (default 10).

    Returns:
        The ECE as a Python ``float`` in ``[0, 1]``. ``0.0`` means the confidence
        matches the accuracy in every populated bin.
    """
    conf = torch.as_tensor(confidences, dtype=torch.float64).flatten()
    corr = torch.as_tensor(correct, dtype=torch.float64).flatten()
    if conf.shape != corr.shape:
        raise ValueError(
            f"confidences and correct must have the same length, "
            f"got {tuple(conf.shape)} and {tuple(corr.shape)}"
        )
    n = conf.numel()
    if n == 0:
        return 0.0

    # Equal-width bin edges 0 = e_0 < e_1 < ... < e_M = 1.
    edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # Left-open, right-closed bins (lo, hi]; the first bin also includes 0.
        if b == 0:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf > lo) & (conf <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        acc = float(corr[in_bin].mean())
        avg_conf = float(conf[in_bin].mean())
        ece += (count / n) * abs(acc - avg_conf)
    return ece
