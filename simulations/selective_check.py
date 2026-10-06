"""Exact selective minimax frontier (Theorem on selective repair) with a quadrature check.

Usage: python selective_check.py
Writes selective_results.{csv,json} beside itself and figures/selective_frontier.{pdf,png}.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "selective-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import gammaln, ive
from scipy.stats import beta, chi2, ncx2, poisson


SCRIPT = Path(__file__).resolve()
OUTPUT = SCRIPT.parent
ASSETS = OUTPUT / "figures"
DIMENSION = 64
SIGNALS = (8., 32., 128.)
ALPHA = .05
EPSILON = .5
P_MIN = 1e-10
COLORS = ("#2364aa", "#d66a24", "#17866b")
SAMPLE_PROBABILITIES = (1e-10, 1e-5, 1e-3, .1, 1.)


def poisson_components(kappa: float):
    """Retain a rigorously bounded mass sufficient for the smallest p used.

The omitted Poisson tail bounds the absolute errors of both the action
probability and weighted loss. Dividing by p bounds conditional-risk error.
SciPy's survival function avoids rounding 1-small_probability to one.
"""
    mu = kappa / 2
    stop = max(20, math.ceil(mu + 10*np.sqrt(mu+1) + 20))
    target = P_MIN * 1e-14
    while poisson.sf(stop, mu) > target:
        stop += 10
    j = np.arange(stop+1, dtype=float)
    weights = np.exp(-mu + j*np.log(mu) - gammaln(j+1))
    return j, weights, float(poisson.sf(stop, mu))


def posterior_risk_squared_radius(t: float, d: int, kappa: float) -> float:
    """Posterior-optimal risk at ||X||^2=t; independent Bessel expression."""
    concentration = np.sqrt(kappa*t)
    if concentration == 0:
        return (d-1)/d
    return float((d-1)*ive(d/2, concentration)
                 / (concentration*ive(d/2-1, concentration)))


def direct_selective_integral(d: int, kappa: float, p: float, threshold: float):
    # Divide inside the integrand so absolute quadrature tolerances remain
    # informative even when p is tiny.
    value, error = quad(
        lambda t: posterior_risk_squared_radius(t, d, kappa)
        * ncx2.pdf(t, d, kappa) / p,
        threshold, np.inf, epsabs=2e-11, epsrel=2e-11, limit=250,
    )
    return float(value), float(error)


def calculate():
    rows, checks, examples, certificates, truncation = [], [], [], [], []
    radius_threshold = float(chi2.ppf(1-ALPHA, DIMENSION)/EPSILON)
    for kappa in SIGNALS:
        j, weights, omitted = poisson_components(kappa)
        degrees = DIMENSION+2*j
        ungated = float(weights @ ((DIMENSION-1)/degrees))
        gate_p = float(ncx2.sf(radius_threshold, DIMENSION, kappa))
        assert gate_p >= P_MIN
        probabilities = sorted(set(np.geomspace(P_MIN, 1., 241).tolist()
                                   + list(SAMPLE_PROBABILITIES) + [gate_p]))
        truncation.append({"kappa": kappa, "largest_j": int(j[-1]),
                           "omitted_poisson_mass": omitted,
                           "worst_conditional_truncation_bound": omitted/P_MIN})
        signal_rows = []
        for p in probabilities:
            threshold = float(ncx2.isf(p, DIMENSION, kappa)) if p < 1 else 0.
            tails = chi2.sf(threshold, degrees)
            action = float(weights @ tails)
            weighted_risk = float((weights*tails) @ ((DIMENSION-1)/degrees))
            risk = weighted_risk/p
            lower = max(0., 1-(1-ungated)/p)
            kind = ("confidence_gate" if p == gate_p else
                    "sample" if p in SAMPLE_PROBABILITIES else "curve")
            row = {"d": DIMENSION, "kappa": kappa, "minimum_action_frequency": p,
                   "threshold_squared": threshold, "conditional_residual": risk,
                   "ungated_residual": ungated, "lower_bound": lower,
                   "mixture_action_probability": action, "kind": kind}
            assert abs(action-p)/p < 2e-10
            assert -1e-12 <= lower <= risk+1e-12 <= 1+2e-12
            rows.append(row)
            signal_rows.append(row)
            if kind == "sample":
                examples.append(row)
            if kind != "curve":
                independent, error = direct_selective_integral(DIMENSION, kappa, p, threshold)
                discrepancy = abs(independent-risk)
                assert discrepancy < 2e-9
                checks.append({"d": DIMENSION, "kappa": kappa, "p": p,
                               "mixture": risk, "posterior_integral": independent,
                               "absolute_discrepancy": discrepancy,
                               "reported_quadrature_error": error})
            if kind == "confidence_gate":
                joint_bad = float((weights*tails) @ beta.sf(
                    EPSILON, (DIMENSION-1)/2, (1+2*j)/2))
                assert joint_bad <= ALPHA+1e-12
                certificates.append({**row, "alpha": ALPHA, "epsilon": EPSILON,
                                     "joint_bad_action_probability": joint_bad,
                                     "conditional_bad_action_probability": joint_bad/p})
        assert np.all(np.diff([row["conditional_residual"] for row in signal_rows]) >= -1e-12)
        assert abs(signal_rows[-1]["conditional_residual"]-ungated) < 1e-12
        grid = np.geomspace(1e-4, 2000., 300)
        assert np.all(np.diff([posterior_risk_squared_radius(t, DIMENSION, kappa)
                              for t in grid]) < 0)
    return rows, {
        "model": "X ~ N(sqrt(kappa) u, I_d), ||u||=1; rank-one projection or identity",
        "dimension": DIMENSION, "signals": SIGNALS,
        "p_range": [P_MIN, 1.], "alpha": ALPHA, "epsilon": EPSILON,
        "objective": "minimax conditional expected residual subject to inf_u P_u(action)>=p",
        "threshold": "tau_p = chi2_noncentral(d,kappa).isf(p)",
        "formula": "(d-1)/p E_Poisson(kappa/2)[chi2_sf(tau_p,d+2J)/(d+2J)]",
        "interpretation": "Fixed-kappa benchmark; threshold uses kappa. Diamonds are the known-scale confidence gate at its true action frequency; their location on these curves does not provide conditional alpha coverage.",
        "truncation": truncation, "independent_checks": checks,
        "maximum_absolute_integral_discrepancy": max(x["absolute_discrepancy"] for x in checks),
        "sample_points": examples, "confidence_gate_points": certificates,
        "rows": rows,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__},
        "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    }


def plot(rows, certificates):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .7,
        "lines.linewidth": 1.9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 260,
    })
    fig, ax = plt.subplots(figsize=(6.9, 2.8), layout="constrained")
    handles = []
    for kappa, color in zip(SIGNALS, COLORS):
        selected = [row for row in rows if row["kappa"] == kappa]
        line, = ax.plot([row["minimum_action_frequency"] for row in selected],
                        [row["conditional_residual"] for row in selected],
                        color=color, label=rf"$\kappa={int(kappa)}$")
        handles.append(line)
        gate = next(row for row in certificates if row["kappa"] == kappa)
        ax.plot(gate["minimum_action_frequency"], gate["conditional_residual"],
                marker="D", ms=5.4, mfc=color, mec="white", mew=.8,
                color=color, linestyle="none", zorder=5, clip_on=False)
        ax.plot(1., selected[-1]["conditional_residual"], marker="o", ms=4,
                mfc="white", mec=color, mew=1.1, linestyle="none", clip_on=False)
    ax.axhline(EPSILON, color="#8b939c", linestyle=(0, (3, 3)), lw=.85, zorder=0)
    ax.text(1.2e-10, EPSILON+.018, r"Target $\epsilon=0.5$", fontsize=8.5,
            color="#59616b", va="bottom")
    handles.append(Line2D([], [], color="#414a56", marker="D", linestyle="none",
                          ms=5, label="Confidence gate"))
    ax.legend(handles=handles, loc="lower right", ncol=2, frameon=False,
              columnspacing=1.6, handlelength=2.2, borderaxespad=.5)
    ax.set(xscale="log", xlim=(P_MIN, 1.2), ylim=(0., 1.),
           xlabel=r"Minimum action frequency, $p$",
           ylabel="Conditional residual fraction",
           title=r"Exact selective frontier ($d=64$, cost 1)")
    ax.set_xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.])
    ax.set_yticks([0., .25, .5, .75, 1.])
    ax.grid(axis="y", alpha=.15, linewidth=.6)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(ASSETS / f"selective_frontier.{suffix}", bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


def main():
    rows, result = calculate()
    with (OUTPUT/"selective_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT/"selective_results.json").write_text(json.dumps(result, indent=2)+"\n")
    plot(rows, result["confidence_gate_points"])
    print("Figure:", ASSETS/"selective_frontier.pdf")
    print("Maximum independent integral discrepancy:", result["maximum_absolute_integral_discrepancy"])
    print("Selected exact sample points (kappa, p, conditional residual):")
    for row in result["sample_points"]:
        print(f"{row['kappa']:g} {row['minimum_action_frequency']:.6g} {row['conditional_residual']:.10f}")
    print("Confidence gate points (kappa, p, conditional residual, joint bad action):")
    for row in result["confidence_gate_points"]:
        print(f"{row['kappa']:g} {row['minimum_action_frequency']:.10g} "
              f"{row['conditional_residual']:.10f} {row['joint_bad_action_probability']:.10g}")


if __name__ == "__main__":
    main()
