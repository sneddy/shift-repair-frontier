"""Semisynthetic check of the arbitrary-content extension with fitted MP, SAL and LEACE.

Real content is drawn iid from three fixed pools of whitened EEG features (paired midpoints of the
Neuroscan/Flex recordings of Williams et al., 2020; raw N170, frozen EEGPT N170 and raw MMN features,
reduced to 16 or 64 principal coordinates). Source shift and endpoint noise are synthetic. Every eraser
is fitted to the full paired endpoints; the row space of the fitted matrix gives its exact population
residual, which is compared with the frontier formula computed before any fitting.

Usage: python real_content.py   (writes results/real_content/{protocol.json,summary.csv,validation.json})
"""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from erasers import fit_erasers
from real_content_predictions import (
    exact_rank_one_risk, expected_monte_carlo_se, sample_budget_prediction,
    verify_numerics,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "real_content"
POOL_PATH = HERE / "real_content_pools.npz"
POOLS = ["n170_raw", "n170_eegpt_pca256", "mmn_raw"]
METHODS = ["MP", "SAL", "LEACE"]
CONFIG = {
    "pools": POOLS, "dimensions": [16, 64],
    "contrast_noise_variances": [0.25, 1.0, 1.75],
    "n_over_d": [0.25, 0.5, 1.0, 2.0, 4.0],
    "theta": 1.0, "direction": "first coordinate unit vector",
    "repetitions": 512, "master_seed": 20260921, "methods": METHODS,
    "additional_budget_checks": {
        "s2": 1.0, "targets": [0.5, 0.25], "multipliers": [0.5, 1.0, 2.0],
        "rounding": "ceil(multiplier * exact minimum integer pairs)",
    },
    "seed_recipe": "uint64 SeedSequence([master_seed, pool_index, d, N, int(100*s2), repeat])",
    "independence": "independent seeds across pools and cells; methods share each calibration set",
    "sampling": "iid uniform pool indices with replacement; identical content in both endpoints",
    "construction": "H+/- = sqrt(1-s2/2)*pool[index] +/- theta*u/2 + N(0,s2*I/2)",
    "population": "conditional on frozen pool and whitening; fresh within-source covariance I",
    "loss": "exact population normalized covariance-adjusted squared mean separation, via SVD of fitted M",
    "cost": "E||MV+a-V||^2 = ||M-I||_F^2+||a||^2 for source-neutral Cov(V)=I",
    "predeclared_summaries": ["cell mean and Monte Carlo SE", "exact a priori risk",
                             "RMSE, maximum absolute error, standardized errors and pointwise coverage",
                             "sample-budget checks at half/exact/double budget",
                             "actual-map rank/kernel/residual diagnostics", "median/IQR costs"],
    "scope": "real represented content, synthetic source and noise; one dataset, not three datasets",
    "excluded_interpretations": ["fraction of all source information", "Gaussian KL or Bayes error",
                                 "per-fit guarantee from expected risk", "natural-shift validation"],
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def protocol():
    cells = {}
    budget_predictions = [sample_budget_prediction(d, 1, 1, target, CONFIG["repetitions"])
                          for d in CONFIG["dimensions"] for target in (0.5, 0.25)]
    for pool_index, pool in enumerate(POOLS):
        for d in CONFIG["dimensions"]:
            entries = [(round(d * ratio), s2, "primary_grid")
                       for s2 in CONFIG["contrast_noise_variances"] for ratio in CONFIG["n_over_d"]]
            entries += [(check["n_pairs"], 1.0, "sample_budget")
                        for p in budget_predictions if p["d"] == d for check in p["check_cells"]]
            for n, s2, role in entries:
                key = f"{pool}_d{d}_n{n}_s{int(s2 * 100):03d}"
                if key not in cells:
                    kappa = n * CONFIG["theta"]**2 / s2
                    cells[key] = {
                        "cell_id": key, "pool": pool, "pool_index": pool_index,
                        "d": d, "n_pairs": n, "s2": s2, "kappa": kappa, "roles": [],
                        "predicted_mean_residual": exact_rank_one_risk(d, kappa),
                        "predicted_mc_se": expected_monte_carlo_se(d, kappa, CONFIG["repetitions"]),
                    }
                cells[key]["roles"].append(role)
    return {"config": CONFIG, "cells": list(cells.values()),
            "budget_predictions": budget_predictions,
            "prediction_numerics": verify_numerics()}


def freeze():
    """Write all predictions (protocol.json) before any eraser is fitted."""
    declared = protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": declared,
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "scipy": scipy.__version__}}
    write_json(OUT / "protocol.json", manifest)
    return manifest


def simulate_cell(cell, z):
    n, d, s2 = cell["n_pairs"], cell["d"], cell["s2"]
    u = np.eye(1, d)[0]
    rows = []
    start = time.perf_counter()
    for repeat in range(CONFIG["repetitions"]):
        seed = int(np.random.SeedSequence([CONFIG["master_seed"], cell["pool_index"], d, n,
                                          int(100*s2), repeat]).generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(seed)
        content = z[rng.integers(len(z), size=n)] * np.sqrt(1-s2/2)
        noise_plus = rng.normal(size=(n, d))*np.sqrt(s2/2)
        noise_minus = rng.normal(size=(n, d))*np.sqrt(s2/2)
        plus = content + CONFIG["theta"]*u/2 + noise_plus
        minus = content - CONFIG["theta"]*u/2 + noise_minus
        cancellation_error = float(np.max(np.abs((plus-minus)-(CONFIG["theta"]*u+noise_plus-noise_minus))))
        assert cancellation_error < 1e-10
        contrast = (plus-minus).mean(0)
        w = contrast / np.linalg.norm(contrast)
        analytic = float(1-(w@u)**2)
        fits = fit_erasers(plus, minus)
        midpoint_mean = (plus.mean(0)+minus.mean(0))/2
        for method in METHODS:
            fit = fits[method]
            _, singular, right_t = np.linalg.svd(fit.matrix, full_matrices=False)
            keep = singular > d*np.finfo(float).eps*singular[0]
            rank = int(keep.sum())
            residual = float(np.square(right_t[keep]@u).sum())
            kernel_error = float(np.linalg.norm(fit.matrix@w))
            idempotence = float(np.linalg.norm(fit.matrix@fit.matrix-fit.matrix) /
                                max(1., np.linalg.norm(fit.matrix)))
            residual_error = abs(residual-analytic)
            mean_error = float(np.linalg.norm(fit.transform(midpoint_mean)-midpoint_mean))
            assert rank == d-1, (cell, repeat, method, rank)
            assert kernel_error < 1e-8 and residual_error < 1e-9 and idempotence < 1e-8
            if method in ("MP", "SAL"):
                assert abs(fit.linear_cost-1) < 1e-10
            else:
                assert mean_error < 1e-8
            rows.append({"cell_id": cell["cell_id"], "pool": cell["pool"], "d": d,
                         "n_pairs": n, "s2": s2, "repeat": repeat, "seed": seed, "method": method,
                         "residual": residual, "analytic_residual": analytic,
                         "contrast_cancellation_error": cancellation_error,
                         "linear_cost": fit.linear_cost, "affine_cost": fit.affine_cost,
                         "rank": rank, "kernel_error": kernel_error,
                         "residual_computation_error": residual_error,
                         "relative_idempotence_error": idempotence,
                         "covariance_support_error": fit.covariance_support_error,
                         "empirical_mean_preservation_error": mean_error})
        assert np.linalg.norm(fits["MP"].matrix-fits["SAL"].matrix) < 1e-10
    return rows, time.perf_counter()-start


def summarize_cell(cell, rows, seconds):
    summaries = []
    for method in METHODS:
        values = [r for r in rows if r["method"] == method]
        risk = np.array([r["residual"] for r in values])
        assert len(risk) == CONFIG["repetitions"]
        summary = {k: v for k, v in cell.items() if k not in ("roles", "pool_index")}
        summary.update(method=method, repetitions=len(risk), residual_mean=float(risk.mean()),
                       residual_mc_se=float(risk.std(ddof=1)/np.sqrt(len(risk))),
                       standardized_error=float((risk.mean()-cell["predicted_mean_residual"])/cell["predicted_mc_se"]))
        for name in ("linear_cost", "affine_cost"):
            q = np.quantile([r[name] for r in values], [0.25, 0.5, 0.75])
            summary.update({f"{name}_q25": float(q[0]), f"{name}_median": float(q[1]),
                            f"{name}_q75": float(q[2])})
        for name in ("kernel_error", "residual_computation_error", "relative_idempotence_error", "covariance_support_error", "contrast_cancellation_error"):
            summary[f"max_{name}"] = max(r[name] for r in values)
        summaries.append(summary)
    return {"cell": cell, "elapsed_seconds": seconds, "summaries": summaries}


def run(manifest):
    cells_dir = OUT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    aggregate = []
    with np.load(POOL_PATH, allow_pickle=False) as pools:
        for index, cell in enumerate(manifest["protocol"]["cells"]):
            csv_path = cells_dir / f"{cell['cell_id']}.csv"
            summary_path = csv_path.with_suffix(".json")
            rows, seconds = simulate_cell(cell, pools[f"{cell['pool']}_d{cell['d']}"])
            temporary = csv_path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(csv_path)
            checkpoint = summarize_cell(cell, rows, seconds)
            write_json(summary_path, checkpoint)
            print(f"Completed {index+1}/{len(manifest['protocol']['cells'])}: {cell['cell_id']} ({seconds:.2f}s)", flush=True)
            aggregate += checkpoint["summaries"]
    with (OUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    independent = [r for r in aggregate if r["method"] == "MP"]
    differences = np.array([r["residual_mean"]-r["predicted_mean_residual"] for r in independent])
    standardized = np.array([r["standardized_error"] for r in independent])
    result = {"unique_conditions": len(independent),
              "independent_calibration_sets": len(independent)*CONFIG["repetitions"],
              "fitted_maps": len(aggregate)*CONFIG["repetitions"],
              "shared_risk_counting": "one observation per calibration set; methods not independent",
              "risk_rmse": float(np.sqrt(np.mean(differences**2))),
              "max_absolute_risk_error": float(np.max(np.abs(differences))),
              "max_absolute_standardized_error": float(np.max(np.abs(standardized))),
              "mean_standardized_error": float(standardized.mean()),
              "pointwise_normal_95_coverage": float(np.mean(np.abs(standardized) <= 1.959963984540054)),
              "coverage_note": "diagnostic using exact theoretical MC SE; simultaneous coverage is not asserted",
              "max_residual_computation_error": max(r["max_residual_computation_error"] for r in aggregate),
              "max_covariance_support_error": max(r["max_covariance_support_error"] for r in aggregate),
              }
    write_json(OUT/"validation.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    run(freeze())
