"""MP, centered rank-one SAL and binary LEACE fitted to Gaussian paired endpoints.

Checks Corollary (empirical erasers) with fitted maps: residual against the exact cost-one frontier,
and linear/affine costs. Usage: python gaussian_content.py   (writes results/gaussian_content.*)
"""

import csv
import json
import platform
from pathlib import Path
import numpy as np
from scipy.stats import poisson
from erasers import fit_erasers, stable_rowspace_residual

HERE = Path(__file__).resolve().parent
PREFIX = HERE / "results" / "gaussian_content"
CONFIG = {
    "dimensions": [16, 64],
    "pair_count_over_dimension": [0.25, 0.5, 1.0, 2.0, 4.0],
    "theta": 1.0,
    "contrast_noise_variance": 1.0,
    "repetitions_per_cell": 512,
    "master_seed": 20260920,
    "methods": ["MP", "SAL", "LEACE"],
    "calibration": "N independent paired endpoints (paired realization of the paper's appendix)",
    "fresh_within_source_covariance": "identity",
    "cost_reference": "V ~ N(0, I); linear and affine costs recorded separately",
}


def exact_rank_one_risk(d, kappa):
    """(d-1) E[1/(d+2J)]; omitted Poisson tail probability <= 1e-14."""
    upper = int(poisson.ppf(1 - 1e-14, kappa / 2)) + 1
    j = np.arange(upper + 1)
    return float(np.sum((d - 1) * poisson.pmf(j, kappa / 2) / (d + 2 * j)))


def generate_pairs(seed, n, d, theta, s2):
    if not 0 < s2 < 2:
        raise ValueError("The full-pair realization requires 0 < s² < 2.")
    rng = np.random.default_rng(seed)
    direction = np.zeros(d)
    direction[0] = 1.0
    content = rng.normal(size=(n, d)) * np.sqrt(1 - s2 / 2)
    plus = content + theta * direction / 2 + rng.normal(size=(n, d)) * np.sqrt(s2 / 2)
    minus = content - theta * direction / 2 + rng.normal(size=(n, d)) * np.sqrt(s2 / 2)
    return plus, minus, direction


def run():
    PREFIX.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in CONFIG["dimensions"]:
        for ratio in CONFIG["pair_count_over_dimension"]:
            n = round(d * ratio)
            for repeat in range(CONFIG["repetitions_per_cell"]):
                seed = int(np.random.SeedSequence([CONFIG["master_seed"], d, n, repeat]).generate_state(1, dtype=np.uint64)[0])
                plus, minus, direction = generate_pairs(seed, n, d, CONFIG["theta"], CONFIG["contrast_noise_variance"])
                fits = fit_erasers(plus, minus)
                contrast = (plus - minus).mean(axis=0)
                w = contrast / np.linalg.norm(contrast)
                analytic_residual = float(1 - (w @ direction) ** 2)
                endpoint_mean = np.concatenate((plus, minus)).mean(axis=0)
                for name, fit in fits.items():
                    singular = np.linalg.svd(fit.matrix, compute_uv=False)
                    rank = int(np.count_nonzero(singular > d * np.finfo(float).eps * singular[0]))
                    kernel_error = float(np.linalg.norm(fit.matrix @ w))
                    residual = stable_rowspace_residual(fit.matrix, direction)
                    residual_error = abs(residual - analytic_residual)
                    idempotence_error = float(np.linalg.norm(fit.matrix @ fit.matrix - fit.matrix))
                    assert rank == d - 1, (name, d, n, seed, rank)
                    assert kernel_error < 1e-9, (name, kernel_error)
                    assert residual_error < 1e-9, (name, residual_error)
                    assert idempotence_error < 1e-8, (name, idempotence_error)
                    if name == "LEACE":
                        assert np.linalg.norm(fit.transform(endpoint_mean) - endpoint_mean) < 1e-9
                    if name in ("MP", "SAL"):
                        assert abs(fit.linear_cost - 1) < 1e-10
                    rows.append({
                        "d": d, "n_pairs": n, "n_over_d": ratio,
                        "repeat": repeat, "seed": seed, "method": name,
                        "residual": residual, "analytic_residual": analytic_residual,
                        "linear_cost": fit.linear_cost, "affine_cost": fit.affine_cost,
                        "rank": rank, "kernel_error": kernel_error,
                        "residual_computation_error": residual_error,
                        "idempotence_error": idempotence_error,
                        "covariance_support_error": fit.covariance_support_error,
                    })
                assert np.linalg.norm(fits["MP"].matrix - fits["SAL"].matrix) < 1e-10
            print(f"Completed d={d}, N={n}, repeats={CONFIG['repetitions_per_cell']}", flush=True)
    with PREFIX.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(CONFIG)
    metadata["python"] = platform.python_version()
    metadata["numpy"] = np.__version__
    metadata["full_pair_generator"] = "C~N(0,(1-s2/2)I); H+/-=C+/-theta*e1/2+E+/-; E iid N(0,s2*I/2)"
    PREFIX.with_suffix(".config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return rows


def load_rows():
    with PREFIX.with_suffix(".csv").open() as f:
        rows = list(csv.DictReader(f))
    integer = {"d", "n_pairs", "repeat", "seed", "rank"}
    return [{k: (int(v) if k in integer else str(v) if k == "method" else float(v)) for k, v in row.items()} for row in rows]


def summarize(rows):
    results = []
    for d in CONFIG["dimensions"]:
        for ratio in CONFIG["pair_count_over_dimension"]:
            n = round(d * ratio)
            exact = exact_rank_one_risk(d, n * CONFIG["theta"] ** 2 / CONFIG["contrast_noise_variance"])
            for name in CONFIG["methods"]:
                selected = [r for r in rows if r["d"] == d and r["n_pairs"] == n and r["method"] == name]
                item = {"d": d, "n_pairs": n, "n_over_d": ratio, "method": name, "repetitions": len(selected), "exact_expected_residual": exact}
                for key in ("residual", "linear_cost", "affine_cost"):
                    values = np.array([r[key] for r in selected])
                    item[key + "_mean"] = float(values.mean())
                    item[key + "_se"] = float(values.std(ddof=1) / np.sqrt(len(values)))
                    item[key + "_median"] = float(np.median(values))
                    item[key + "_q25"] = float(np.quantile(values, 0.25))
                    item[key + "_q75"] = float(np.quantile(values, 0.75))
                item["residual_z_score"] = (item["residual_mean"] - exact) / item["residual_se"]
                results.append(item)
    report = {
        "cells": results,
        "max_kernel_error": max(r["kernel_error"] for r in rows),
        "max_residual_computation_error": max(r["residual_computation_error"] for r in rows),
        "max_idempotence_error": max(r["idempotence_error"] for r in rows),
        "max_covariance_support_error": max(r["covariance_support_error"] for r in rows),
        "max_abs_monte_carlo_z_score": max(abs(r["residual_z_score"]) for r in results),
        "interpretation": "Risk is the population covariance-adjusted residual, not held-out probe accuracy. Kernel equality gives per-fit agreement; algorithms can have different costs. Cost summaries are diagnostics, not a comparison at the same fixed budget.",
    }
    PREFIX.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    PREFIX.parent.mkdir(exist_ok=True)
    summary = summarize(run())
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2))
