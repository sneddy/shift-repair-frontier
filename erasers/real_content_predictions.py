"""A priori sample-budget predictions for the real-content demonstration.

The contents of a real-data pool are irrelevant to these predictions if the
same content is shared by each pair and the injected contrast noise is
isotropic Gaussian. This module does not inspect or fit experimental data.

Targets concern expected covariance-adjusted residual over calibration fits.
They are not a bound for every individual fitted editor or a certificate.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Iterable

import numpy as np
from scipy.integrate import quad
from scipy.stats import poisson


def _check(d: int, kappa: float) -> None:
    if int(d) != d or d < 2:
        raise ValueError("d must be an integer at least two.")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative.")


def _poisson_support(kappa: float, tail_probability: float = 1e-13):
    """Support whose omitted mass is at most tail_probability, up to roundoff."""
    if kappa == 0:
        return np.array([0], dtype=np.int64), np.array([1.0])
    lam = kappa / 2
    lower = max(0, int(poisson.ppf(tail_probability / 2, lam)))
    upper = int(poisson.isf(tail_probability / 2, lam)) + 1
    j = np.arange(lower, upper + 1)
    return j, poisson.pmf(j, lam)


def exact_rank_one_risk(d: int, kappa: float) -> float:
    """R=(d−1) E[1/(d+2J)], J~Poisson(kappa/2)."""
    _check(d, kappa)
    j, mass = _poisson_support(kappa)
    return float(np.sum(mass * (d - 1) / (d + 2 * j)))


def exact_rank_one_risk_quad(d: int, kappa: float) -> float:
    """Independent integral evaluation, rescaled to remain stable at large κ.

    R=(d−1)∫_0^1 t^(d−1) exp[κ(t²−1)/2]dt. Substituting
    t=exp(−x/(d+κ)) gives the integral below on [0,∞).
    """
    _check(d, kappa)
    scale = d + kappa

    def integrand(x):
        return math.exp(-d * x / scale + (kappa / 2) * math.expm1(-2 * x / scale))

    value, _ = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-12, limit=200)
    return float((d - 1) * value / scale)


def rank_one_residual_variance(d: int, kappa: float) -> float:
    """Variance from the conditional Beta second moment, independently of fits.

    L|J~Beta((d−1)/2,(1+2J)/2), hence E[L²|J]=(d²−1)/[(d+2J)(d+2J+2)].
    """
    _check(d, kappa)
    j, mass = _poisson_support(kappa)
    mean = exact_rank_one_risk(d, kappa)
    second = float(np.sum(mass * (d*d - 1) / ((d + 2*j) * (d + 2*j + 2))))
    return max(0.0, second - mean * mean)


def expected_monte_carlo_se(d: int, kappa: float, repetitions: int = 512) -> float:
    if int(repetitions) != repetitions or repetitions < 2:
        raise ValueError("At least two independent calibration repetitions are required.")
    return math.sqrt(rank_one_residual_variance(d, kappa) / repetitions)


def _check_budget(d: int, theta: float, s2: float, target: float) -> None:
    _check(d, 0)
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be positive; zero signal is a directional limit.")
    if not np.isfinite(s2) or s2 <= 0:
        raise ValueError("s2 is the positive variance of the paired contrast noise.")
    if not np.isfinite(target) or not 0 < target < 1:
        raise ValueError("target must lie strictly between zero and one.")


def minimal_pairs(d: int, theta: float, s2: float, target: float) -> int:
    """Smallest integer N>=1 with expected residual at most target.

    Monotonicity follows by coupling Poisson random variables at increasing
    κ and noting that (d−1)/(d+2J) decreases with J. Integer binary search
    avoids treating a continuous root as an integer sample budget.
    """
    _check_budget(d, theta, s2, target)
    per_pair = theta * theta / s2
    # This upper endpoint is guaranteed sufficient by the upper bound in Equation (8) of the paper.
    lower = 1
    upper = max(1, math.ceil((d - 1) / (target * per_pair)))
    while lower < upper:
        middle = (lower + upper) // 2
        if exact_rank_one_risk(d, middle * per_pair) <= target:
            upper = middle
        else:
            lower = middle + 1
    return lower


def sample_budget_prediction(
    d: int,
    theta: float,
    s2: float,
    target: float,
    repetitions: int = 512,
    relative_budgets: Iterable[float] = (0.5, 1.0, 2.0),
) -> dict:
    """Exact integer budget, theorem bracket, and predeclared check cells.

    necessary_lower_pairs <= N* <= sufficient_upper_pairs. A necessary
    lower bound is not a sufficient budget. ``s2`` may be any positive
    contrast variance here, but the proposed full-pair construction further
    requires 0<s2<2 to retain identity fresh within-source covariance.
    """
    _check_budget(d, theta, s2, target)
    per_pair = theta * theta / s2
    n_star = minimal_pairs(d, theta, s2, target)
    necessary = max(1, math.ceil(((d - 1) / target - d) / per_pair))
    sufficient = max(1, math.ceil((d - 1) / (target * per_pair)))
    cells = []
    for multiplier in relative_budgets:
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Budget multipliers must be positive and finite.")
        n = max(1, math.ceil(multiplier * n_star))
        kappa = n * per_pair
        cells.append({
            "budget_multiplier": float(multiplier), "n_pairs": n,
            "kappa": kappa, "predicted_mean_residual": exact_rank_one_risk(d, kappa),
            "predicted_mc_standard_error": expected_monte_carlo_se(d, kappa, repetitions),
        })
    return {
        "d": int(d), "theta": float(theta), "contrast_noise_variance": float(s2),
        "expected_residual_target": float(target),
        "exact_minimum_pairs": n_star,
        "necessary_lower_pairs": necessary, "sufficient_upper_pairs": sufficient,
        "risk_at_previous_integer": exact_rank_one_risk(d, (n_star - 1) * per_pair),
        "risk_at_exact_budget": exact_rank_one_risk(d, n_star * per_pair),
        "mc_repetitions": repetitions, "check_cells": cells,
        "scope": "Expected residual over repeated calibration fits; not a per-fit guarantee.",
    }


def default_prediction_table(repetitions: int = 512) -> list[dict]:
    return [sample_budget_prediction(d, 1.0, s2, epsilon, repetitions)
            for d in (16, 64) for s2 in (0.25, 1.0, 1.75) for epsilon in (0.5, 0.25)]


def verify_numerics() -> dict:
    """Bounded, data-independent cross-checks and integer-boundary audits."""
    quadrature_errors = []
    for d in (2, 16, 64, 256):
        values = []
        for kappa in (0.0, 0.1, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0):
            risk = exact_rank_one_risk(d, kappa)
            reference = exact_rank_one_risk_quad(d, kappa)
            quadrature_errors.append(abs(risk - reference))
            assert abs(risk - reference) < 2e-11, (d, kappa, risk, reference)
            values.append(risk)
            assert rank_one_residual_variance(d, kappa) >= 0
            if kappa > 0:
                assert (d - 1) / (d + kappa) <= risk + 1e-12
                assert risk <= (d - 1) * min(1/d, 1/kappa) + 1e-12
        assert all(a > b for a, b in zip(values[:-1], values[1:]))
    predictions = default_prediction_table()
    for p in predictions:
        assert p["necessary_lower_pairs"] <= p["exact_minimum_pairs"] <= p["sufficient_upper_pairs"]
        assert p["risk_at_exact_budget"] <= p["expected_residual_target"]
        assert p["risk_at_previous_integer"] > p["expected_residual_target"]
    # Independent closed-form κ=0 check for moments of a random direction.
    for d in (2, 16, 64, 256):
        variance_at_zero = 2 * (d - 1) / (d*d * (d + 2))
        assert abs(rank_one_residual_variance(d, 0) - variance_at_zero) < 1e-13
    return {
        "risk_comparisons": len(quadrature_errors),
        "maximum_absolute_poisson_quadrature_difference": max(quadrature_errors),
        "sample_budget_rows_checked": len(predictions),
        "checked": ["Poisson versus independent integral", "strict risk monotonicity",
                    "necessary/sufficient bounds", "exact integer crossing", "variance at zero"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit all prediction cells and numerical checks.")
    arguments = parser.parse_args()
    checks = verify_numerics()
    predictions = default_prediction_table()
    if arguments.json:
        print(json.dumps({"verification": checks, "predictions": predictions}, indent=2))
    else:
        print("d  s²   target  necessary  exact N*  sufficient  R(N*−1)  R(N*)")
        for p in predictions:
            print(f"{p['d']:2d} {p['contrast_noise_variance']:4.2f} {p['expected_residual_target']:6.2f}"
                  f" {p['necessary_lower_pairs']:10d} {p['exact_minimum_pairs']:9d}"
                  f" {p['sufficient_upper_pairs']:11d} {p['risk_at_previous_integer']:8.5f}"
                  f" {p['risk_at_exact_budget']:7.5f}")
        print(json.dumps(checks, indent=2))
