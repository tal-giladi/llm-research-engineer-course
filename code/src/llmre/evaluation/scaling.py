"""Scaling-law arithmetic: the 6ND compute rule, power-law fitting, and the
Chinchilla compute-optimal allocation.

This is the code companion to Module 10 (`lessons/module-10/`). Nothing here
touches a tensor — these are plain scalar / NumPy formulas that turn the
empirical laws of Kaplan et al. 2020 and Hoffmann et al. 2022 into three
functions you can call before spending a single GPU-hour:

* :func:`training_flops` — the ``C = 6ND`` training-compute rule (lesson 10.1).
* :func:`fit_power_law` — recover the exponent of a power law ``L = c * x**(-a)``
  by least-squares regression in log-log space (lesson 10.2).
* :func:`chinchilla_optimal` — given a compute budget ``C`` and the parametric
  fit ``L(N, D) = E + A/N**alpha + B/D**beta``, return the loss-minimising
  ``(N*, D*)`` under the constraint ``C = 6ND`` (lesson 10.3).

The FLOP counting mirrors :mod:`llmre.training.metrics` (Module 7): there,
``model_flops_per_token`` returns ``6N`` per token; here, :func:`training_flops`
multiplies by the token budget ``D`` to get the whole run's ``6ND``.
"""

from __future__ import annotations

import numpy as np


def training_flops(N: float, D: float) -> float:
    """Total training compute in FLOPs via the ``C = 6ND`` rule.

    A forward+backward pass over a transformer costs about ``6N`` floating-point
    operations per token (``2N`` forward, ``4N`` backward — see lesson 07.4 and
    :func:`llmre.training.metrics.model_flops_per_token`). Multiplying by the
    number of tokens ``D`` the run sees gives the whole run's compute.

    Args:
        N: number of (non-embedding) model parameters. Plain Python/NumPy scalar,
            no tensor, no device.
        D: number of training tokens processed over the whole run.

    Returns:
        ``6.0 * N * D`` as a ``float`` — the estimated total training FLOPs.

    Example:
        >>> training_flops(85e6, 1.7e9)
        8.67e+17
    """
    return 6.0 * N * D


def fit_power_law(x, y) -> tuple[float, float]:
    """Fit a decaying power law ``y = coefficient * x**(-exponent)`` in log-log space.

    Scaling laws are written in the *decay* convention: loss falls as a positive
    power of the resource, e.g. ``L(N) = (N_c / N)**alpha_N`` with ``alpha_N > 0``.
    Taking logs makes this a straight line,

        ``log(y) = log(coefficient) - exponent * log(x)``,

    so we recover the exponent by ordinary least squares on ``(log x, log y)``
    and negate the slope (the slope is ``-exponent`` because ``y`` decreases in
    ``x``). A returned ``exponent`` is therefore *positive* for data that falls
    as ``x`` grows.

    Args:
        x: 1-D array-like of positive resource values (parameters, tokens, or
            compute). Must be strictly positive (logs are taken).
        y: 1-D array-like of positive losses, same length as ``x``.

    Returns:
        ``(exponent, coefficient)`` as Python ``float``s for the model
        ``y = coefficient * x**(-exponent)``. ``exponent`` is the scaling-law
        power (positive when ``y`` decreases in ``x``); ``coefficient`` is the
        prefactor, i.e. the fitted ``y`` at ``x = 1``.

    Raises:
        ValueError: if ``x`` and ``y`` differ in length, have fewer than two
            points, or contain non-positive values.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if x.size < 2:
        raise ValueError("need at least two points to fit a line")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law fit needs strictly positive x and y (logs are taken)")

    log_x = np.log(x)
    log_y = np.log(y)
    # Least-squares line log_y = slope * log_x + intercept. np.polyfit(deg=1)
    # returns [slope, intercept]. slope == -exponent in the decay convention.
    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    exponent = -float(slope)
    coefficient = float(np.exp(intercept))
    return exponent, coefficient


def chinchilla_loss(N: float, D: float, A: float, B: float,
                    alpha: float, beta: float, E: float) -> float:
    """The Chinchilla parametric loss ``L(N, D) = E + A/N**alpha + B/D**beta``.

    Args:
        N: parameter count.
        D: token count.
        A, B, alpha, beta, E: the fitted constants of Hoffmann et al. 2022. ``E``
            is the irreducible loss (entropy of text); ``A/N**alpha`` is the
            finite-model penalty; ``B/D**beta`` is the finite-data penalty.

    Returns:
        The predicted loss as a ``float`` (nats per token in the paper's fit).
    """
    return E + A / N**alpha + B / D**beta


def chinchilla_optimal(compute_flops: float, A: float, B: float,
                       alpha: float, beta: float, E: float = 0.0) -> tuple[float, float]:
    """Compute-optimal ``(N*, D*)`` for a fixed budget under ``C = 6ND``.

    Given the parametric fit ``L(N, D) = E + A/N**alpha + B/D**beta`` and a fixed
    compute budget ``C = 6ND``, we substitute ``D = C / (6N)`` and minimise over
    ``N``. Setting the derivative to zero gives a closed form:

        ``N* = [ (alpha * A) / (beta * B) * (C / 6)**beta ] ** (1 / (alpha + beta))``
        ``D* = C / (6 * N*)``.

    Equivalently ``N* ∝ C**a`` and ``D* ∝ C**b`` with ``a = beta/(alpha+beta)``
    and ``b = alpha/(alpha+beta)`` — the exponents Hoffmann et al. report as
    ``a ≈ 0.46``, ``b ≈ 0.54``, i.e. scale ``N`` and ``D`` almost together. ``E``
    does not affect the optimum (it is an additive constant in ``N`` and ``D``)
    and is accepted only so the signature matches the full loss.

    Args:
        compute_flops: the total compute budget ``C`` in FLOPs.
        A, B, alpha, beta: the fitted constants (``alpha, beta > 0``).
        E: irreducible loss; ignored for the optimisation (default ``0.0``).

    Returns:
        ``(N_star, D_star)`` as ``float``s, satisfying ``6 * N_star * D_star ==
        compute_flops`` to floating-point precision.

    Raises:
        ValueError: if ``compute_flops <= 0`` or ``alpha + beta <= 0``.
    """
    if compute_flops <= 0:
        raise ValueError("compute_flops must be positive")
    if alpha + beta <= 0:
        raise ValueError("alpha + beta must be positive")

    half_budget = compute_flops / 6.0
    ratio = (alpha * A) / (beta * B)
    N_star = (ratio * half_budget**beta) ** (1.0 / (alpha + beta))
    D_star = compute_flops / (6.0 * N_star)
    return float(N_star), float(D_star)
