"""pass@k: the probability that at least one of k attempts solves a problem.

This is Frontier update F.1 (post-training as a benchmarked discipline). The
MLPerf post-training benchmark validates its RLVR coding agent with **pass@4**:
a problem counts as solved if at least one of four attempts passes the hidden
tests. It was chosen because it had the lowest variance of the metrics studied.

The naive estimator — draw exactly k samples, check if any passed — is noisy.
The standard **unbiased estimator** (Chen et al. 2021, *Evaluating Large
Language Models Trained on Code*) draws ``n >= k`` samples, counts ``c``
correct, and computes the probability that a random size-``k`` subset of those
``n`` contains at least one correct one::

    pass@k = 1 - C(n - c, k) / C(n, k)

``C(n - c, k) / C(n, k)`` is the chance that all ``k`` picks come from the
``n - c`` failures. We compute it as a running product to avoid huge binomials::

    C(n - c, k) / C(n, k) = prod_{i = n - c + 1}^{n} (1 - k / i)
"""

from __future__ import annotations

from typing import Iterable


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem.

    Args:
        n: number of samples drawn for the problem (``n >= k``).
        c: how many of those ``n`` samples passed the verifier (``0 <= c <= n``).
        k: attempt budget being estimated.

    Returns:
        A Python ``float`` in ``[0, 1]``.
    """
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0                      # every size-k subset contains a success
    prob_all_fail = 1.0
    for i in range(n - c + 1, n + 1):
        prob_all_fail *= 1.0 - k / i
    return 1.0 - prob_all_fail


def mean_pass_at_k(results: Iterable[tuple[int, int]], k: int) -> float:
    """Average pass@k over a benchmark given ``(n, c)`` per problem."""
    scores = [pass_at_k(n, c, k) for n, c in results]
    if not scores:
        raise ValueError("no problems given")
    return sum(scores) / len(scores)
