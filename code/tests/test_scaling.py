"""Tests for llmre.evaluation.scaling (Module 10).

Three checks, one per public function:

* :func:`fit_power_law` recovers a *known* exponent: we generate exact data
  ``y = c * x**(-0.3)`` and assert the fit returns ``0.3`` (and ``c``).
* :func:`training_flops` is exactly ``6 * N * D``.
* :func:`chinchilla_optimal` returns ``(N*, D*)`` that (a) satisfy the budget
  constraint ``6 * N* * D* == C``, (b) give a sane tokens-per-parameter ratio,
  and (c) match an independent brute-force grid minimisation of the loss.
"""
import numpy as np

from llmre.evaluation.scaling import (
    chinchilla_loss,
    chinchilla_optimal,
    fit_power_law,
    training_flops,
)


def test_fit_power_law_recovers_known_exponent():
    # Exact power law y = 5 * x**(-0.3). The fit must recover exponent 0.3
    # (positive, decay convention) and coefficient 5.
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    c_true, a_true = 5.0, 0.3
    y = c_true * x ** (-a_true)

    exponent, coefficient = fit_power_law(x, y)

    assert abs(exponent - a_true) < 1e-6, exponent
    assert abs(coefficient - c_true) < 1e-6, coefficient


def test_fit_power_law_recovers_exponent_with_noise():
    # With mild multiplicative noise the fit should still land close to the truth.
    rng = np.random.default_rng(0)
    x = np.logspace(0, 3, 40)
    a_true, c_true = 0.076, 2.5  # ~ Kaplan's alpha_N ballpark
    y = c_true * x ** (-a_true) * np.exp(rng.normal(0.0, 0.02, size=x.shape))

    exponent, _ = fit_power_law(x, y)
    assert abs(exponent - a_true) < 0.01, exponent


def test_training_flops_is_6ND():
    assert training_flops(85e6, 1.7e9) == 6 * 85e6 * 1.7e9
    assert training_flops(1, 1) == 6.0
    N, D = 1.3e9, 2.6e10
    assert training_flops(N, D) == 6.0 * N * D


# Chinchilla replication constants (Besiroglu et al. 2024 revised fit), which
# reproduce the headline "~20 tokens per parameter" across scales.
A, B, ALPHA, BETA, E = 482.01, 2085.43, 0.3478, 0.3658, 1.82


def test_chinchilla_optimal_satisfies_budget():
    C = 1e21
    N_star, D_star = chinchilla_optimal(C, A, B, ALPHA, BETA, E)
    # 6 N* D* must equal the budget to floating-point precision.
    assert np.isclose(6.0 * N_star * D_star, C, rtol=1e-9)


def test_chinchilla_optimal_sane_tokens_per_param():
    # Across a wide compute range the tokens/param ratio should sit in a sane
    # band around the ~20x Chinchilla rule of thumb.
    for C in [1e19, 1e21, 5.76e23]:
        N_star, D_star = chinchilla_optimal(C, A, B, ALPHA, BETA, E)
        ratio = D_star / N_star
        assert 10.0 < ratio < 40.0, (C, ratio)


def test_chinchilla_optimal_matches_grid_minimum():
    # The closed form must agree with a brute-force minimisation of L(N, C/6N).
    C = 5.76e23  # Gopher's compute budget
    N_star, D_star = chinchilla_optimal(C, A, B, ALPHA, BETA, E)

    N_grid = np.logspace(9, 12, 400_000)
    D_grid = C / (6.0 * N_grid)
    L = chinchilla_loss(N_grid, D_grid, A, B, ALPHA, BETA, E)
    N_best = N_grid[np.argmin(L)]

    # Within the grid spacing (log-spaced, ~1e-5 relative step here).
    assert abs(np.log(N_star) - np.log(N_best)) < 1e-3, (N_star, N_best)
    # And this budget/constants reproduce Chinchilla ~70B params, ~1.4T tokens.
    assert 5e10 < N_star < 1e11, N_star
    assert 1e12 < D_star < 2e12, D_star


def test_chinchilla_optimal_scales_together():
    # N* and D* both grow with the compute budget (never one at the other's cost).
    N1, D1 = chinchilla_optimal(1e20, A, B, ALPHA, BETA, E)
    N2, D2 = chinchilla_optimal(1e22, A, B, ALPHA, BETA, E)
    assert N2 > N1 and D2 > D1
