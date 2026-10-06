"""Anisotropic within-source covariance: residual of the empirical-direction projector versus the isotropic formula.

Content C ~ N(0, Sigma_C) with power-law spectrum lambda_j ∝ j^(-alpha) (mean content variance 1 - s^2/2),
isotropic measurement noise N(0, s^2 I/2) per endpoint, shift theta*u.  Editing, cost and loss are taken in
coordinates whitened by the within-source covariance Sigma_V = Sigma_C + s^2 I/2 (Mahalanobis residual,
Sigma_V-weighted Frobenius cost), where the contrast noise has covariance s^2 Sigma_V^{-1}.  The editor
projects out the whitened empirical contrast direction.  The isotropic formula is evaluated at the
whitened signal-to-noise ratio kappa = N ||delta_w||^2 / mean_j(s^2/lambda_V,j) and, alternatively, at
the effective dimension of the whitened noise spectrum.  Usage: python anisotropic.py   (writes anisotropic_results.json)
"""
from __future__ import annotations
import json, sys, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulations"))
from model import exact_risk
D, THETA, REPS, SEED = 64, 1.0, 4000, 20260923
ALPHAS, S2S, DIRECTIONS, N_OVER_D = (1.0, 2.0), (0.1, 1.0), ("random", "top", "bottom"), (.25, .5, 1, 2, 4)

def cell(alpha, s2, direction, rng):
    lam_c = np.arange(1, D + 1, dtype=float) ** (-alpha); lam_c *= (1 - s2 / 2) * D / lam_c.sum()
    lam_v = lam_c + s2 / 2; noise_w = s2 / lam_v
    d_eff = float(noise_w.sum() ** 2 / (noise_w ** 2).sum())
    fixed = None if direction == "random" else np.eye(D)[0 if direction == "top" else -1]
    rows = []
    for nd in N_OVER_D:
        n = int(nd * D); losses = []
        for _ in range(REPS):
            u = fixed if fixed is not None else (lambda v: v / np.linalg.norm(v))(rng.standard_normal(D))
            delta_w = THETA * u / np.sqrt(lam_v); u_w = delta_w / np.linalg.norm(delta_w)
            Dbar = delta_w + rng.standard_normal(D) * np.sqrt(noise_w / n)
            w = Dbar / np.linalg.norm(Dbar); losses.append(1 - (u_w @ w) ** 2)
        losses = np.array(losses)
        if fixed is not None:
            kappa = n * float(delta_w @ delta_w) / noise_w.mean(); kappa_u = n * float(delta_w @ delta_w) / float(noise_w @ u_w ** 2)
        else:
            kappa = n * THETA ** 2 * float(np.mean(1 / lam_v)) / noise_w.mean(); kappa_u = None
        rows.append(dict(n_over_d=nd, N=n, observed=float(losses.mean()), mc_se=float(losses.std(ddof=1) / np.sqrt(REPS)),
                         isotropic=exact_risk(D, kappa, 1), isotropic_d_eff=exact_risk(d_eff, kappa * d_eff / D, 1),
                         kappa=kappa, kappa_along_u=kappa_u))
    return dict(alpha=alpha, s2=s2, direction=direction, whitened_noise_d_eff=d_eff,
                whitened_noise_max_over_mean=float(noise_w.max() / noise_w.mean()), rows=rows)

def main():
    rng = np.random.default_rng(SEED); cells = []
    for s2 in S2S:
        for alpha in ALPHAS:
            for direction in DIRECTIONS:
                c = cell(alpha, s2, direction, rng); cells.append(c)
                worst = max(abs(r["observed"] - r["isotropic"]) for r in c["rows"])
                print(f"s2={s2} alpha={alpha} {direction:6s} d_eff={c['whitened_noise_d_eff']:.1f} max|obs-iso|={worst:.4f}")
    dev = [abs(r["observed"] - r["isotropic"]) for c in cells for r in c["rows"]]
    out = dict(d=D, theta=THETA, repetitions=REPS, seed=SEED, cells=cells,
               max_abs_deviation=max(dev), rms_deviation=float(np.sqrt(np.mean(np.square(dev)))),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / "anisotropic_results.json").write_text(json.dumps(out, indent=1) + "\n")
    print("max abs deviation", out["max_abs_deviation"], "rms", out["rms_deviation"])

if __name__ == "__main__":
    main()
