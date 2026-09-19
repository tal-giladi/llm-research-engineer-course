"""llmre.agents.stats — seed-variance reporting helpers (Module 19.2).

An ML result from a single random seed is a sample of size one: you cannot tell
signal from seed noise. The honest report of a metric across ``k`` seeds is a
**mean with a spread** and, better, a **confidence interval**. These two tiny,
dependency-light functions produce exactly that, so a lesson (and an experiment
report) can state ``mean +/- std`` and a CI without pulling in SciPy.

Everything here is plain Python floats — no tensors. Standard deviation is the
**sample** std (Bessel's correction, ``ddof=1``): with a handful of seeds you are
estimating the population spread from a small sample, so dividing by ``n-1`` is
the unbiased choice.
"""

from __future__ import annotations

import math
from statistics import NormalDist

# Two-sided Student-t critical values t_{1-alpha/2, df} for the three confidence
# levels used in practice. Rows are degrees of freedom (df = n - 1); the final
# entry is the df -> inf limit (the standard normal z). Values are the standard
# textbook table, verified against SciPy's stats.t.ppf.
_T_TABLE: dict[int, dict[int, float]] = {
    90: {  # 90% two-sided -> alpha/2 = 0.05
        1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
        8: 1.860, 9: 1.833, 10: 1.812, 15: 1.753, 20: 1.725, 30: 1.697,
    },
    95: {  # 95% two-sided -> alpha/2 = 0.025
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042,
    },
    99: {  # 99% two-sided -> alpha/2 = 0.005
        1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707, 7: 3.499,
        8: 3.355, 9: 3.250, 10: 3.169, 15: 2.947, 20: 2.845, 30: 2.750,
    },
}


def mean_std(values) -> tuple[float, float]:
    """Return the ``(mean, sample_std)`` of a sequence of numbers.

    Args:
        values: any non-empty iterable of real numbers (e.g. one metric per
                seed). Length ``n >= 1``.

    Returns:
        ``(mean, std)`` as plain floats. ``std`` uses ``ddof=1`` (divide by
        ``n-1``); with ``n == 1`` the sample std is undefined, so it is
        reported as ``0.0``.

    Raises:
        ValueError: if ``values`` is empty.
    """
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        raise ValueError("mean_std requires at least one value")
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


def _t_critical(df: int, confidence: float) -> float:
    """Two-sided critical value for ``confidence`` at ``df`` degrees of freedom.

    Uses the built-in :data:`_T_TABLE` for the standard levels/df, the next
    smaller tabulated df (conservative) for untabulated small df, and the normal
    ``z`` value (df -> inf) for ``df > 30`` or any non-standard confidence.
    """
    pct = round(confidence * 100)
    table = _T_TABLE.get(pct)
    # Normal approximation: z = Phi^{-1}(1 - alpha/2).
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    if table is None or df > 30:
        return z
    if df in table:
        return table[df]
    # Untabulated small df: use the largest tabulated df <= this one (a slightly
    # wider, conservative interval), or z if none is smaller.
    smaller = [k for k in table if k <= df]
    return table[max(smaller)] if smaller else z


def confidence_interval(values, confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided confidence interval for the mean across seeds.

    The interval is ``mean +/- t * s / sqrt(n)`` where ``s`` is the sample std
    and ``t`` is the Student-t critical value for ``n-1`` degrees of freedom
    (falling back to the normal ``z`` for large ``n`` or non-standard
    confidence). This is the right interval for the *mean* of a small number of
    noisy seed results.

    Args:
        values:     non-empty iterable of real numbers (one per seed), ``n >= 1``.
        confidence: coverage probability in ``(0, 1)``, e.g. ``0.95``.

    Returns:
        ``(low, high)`` floats. With ``n == 1`` the spread is unknown, so the
        interval collapses to ``(mean, mean)`` — an explicit signal that a single
        seed carries no uncertainty estimate.

    Raises:
        ValueError: if ``values`` is empty or ``confidence`` is not in ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        raise ValueError("confidence_interval requires at least one value")
    mean, std = mean_std(xs)
    if n == 1:
        return mean, mean
    t = _t_critical(n - 1, confidence)
    margin = t * std / math.sqrt(n)
    return mean - margin, mean + margin
