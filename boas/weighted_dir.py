"""Exploratory: does a noise-weighted direction beat the mean direction on BOAS?

Same participants, splits and calibration draws as demo.py (N2, 0.5 Hz, channel rule). For each
calibration draw (P participants, n epochs each) we delete, in the pool-whitened coordinates,
  * mp:  the mean contrast Dbar (MP / SAL / LEACE kernel), as in the paper;
  * map: the maximum a posteriori direction under a uniform prior on the sphere, with theta^2 and the
         noise covariance C = (Sigma_b + Sigma_w / n) / P estimated on the calibration pool; it has the
         form (I + g C)^{-1} Dbar, with g fixed by the unit-norm condition (trust-region solution);
  * best: (I + g C)^{-1} Dbar with g chosen on the TEST residual from a grid (oracle; an optimistic
         bound on what noise weighting of this form can give);
  * pop: the pool-mean direction (all 43 pool participants, all epochs), a reference for what
         much more calibration would give.
The residual of each is measured on the test participants with the off-diagonal statistic.

Usage:  python3 weighted_dir.py
Writes results/weighted_dir.json.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
from common import RESULTS
from demo import (P_GRID, N_GRID, SPLITS, REPEATS, SEED, load_people, split_once, pool_params,
                  offdiag, draw)


GAMMAS = np.concatenate([[0.0], np.logspace(-3, 3, 19)])


def ridge_dir(z, c, g):
    v = z / (1 + g * c)
    return v / np.linalg.norm(v)


def map_dir(z, c, th2):
    """argmax over the unit sphere of theta u'C^{-1}z - theta^2 u'C^{-1}u / 2 for diagonal C = diag(c):
    u = theta (theta^2 + 2 lam c)^{-1} z with lam the root of ||u|| = 1 on (-theta^2 / (2 c_max), inf)."""
    th = np.sqrt(max(th2, 1e-12)); c = np.maximum(c, 1e-12)
    f = lambda lam: np.sum((th * z / (th2 + 2 * lam * c)) ** 2) - 1
    lo = -th2 / (2 * c.max()) * (1 - 1e-9); hi = 1.0
    while f(hi) > 0:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    v = th * z / (th2 + 2 * hi * c)
    return v / np.linalg.norm(v)


def main() -> None:
    people, dropped = load_people(2, 0.5, True)
    rng = np.random.default_rng(SEED + 1)
    names = ("mp", "map", "best", "pop")
    cells = {(P, n): {k: [] for k in names + ("diff",)} for P in P_GRID for n in N_GRID}
    for s in range(SPLITS):
        W, m_test, Dpool = split_once(people, rng)
        q0 = offdiag(m_test)
        delta, Sb, Sw, Sn = pool_params(Dpool, False)
        u_pop = delta / np.linalg.norm(delta)
        r_pop = offdiag(m_test, u_pop) / q0
        eig = {n: np.linalg.eigh(Sb + Sw / n) for n in N_GRID}
        th2 = float(delta @ delta)
        for P in P_GRID:
            for n in N_GRID:
                obs = {k: [] for k in names}
                for _ in range(REPEATS):
                    who = rng.choice(len(Dpool), P, replace=False)
                    Dbar = np.mean([draw(Dpool[i], n, False, rng) for i in who], 0)
                    w_mp = Dbar / np.linalg.norm(Dbar)
                    c, U = eig[n]; c = np.maximum(c, 0) / P; z = U.T @ Dbar
                    obs["mp"].append(offdiag(m_test, w_mp) / q0)
                    obs["map"].append(offdiag(m_test, U @ map_dir(z, c, th2)) / q0)
                    obs["best_grid"] = obs.get("best_grid", []) + [[offdiag(m_test, U @ ridge_dir(z, c, gm)) / q0
                                                                    for gm in GAMMAS / max(c.mean(), 1e-12)]]
                    obs["pop"].append(r_pop)
                bg = np.array(obs.pop("best_grid")).mean(0)
                obs["best"] = [float(bg.min())]
                cc = cells[(P, n)]
                for k in names:
                    cc[k].append(float(np.mean(obs[k])))
                cc["diff"].append(float(np.mean(obs["map"]) - np.mean(obs["mp"])))
        print(f"split {s + 1}/{SPLITS} done", flush=True)
    rows = []
    for (P, n), c in cells.items():
        row = dict(P=P, n=n)
        for k in names + ("diff",):
            a = np.array(c[k])
            row[k] = float(a.mean()); row[f"{k}_se"] = float(a.std(ddof=1) / np.sqrt(len(a)))
        rows.append(row)
    out = dict(design="demo.py splits; map = MAP direction (I + g C)^{-1} Dbar with pool theta^2 and C; best = oracle g on test", splits=SPLITS,
               repeats=REPEATS, n_people=len(people), rows=rows,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (RESULTS / "weighted_dir.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        if r["n"] in (1, 8, 128) and r["P"] in (1, 4, 8, 32):
            print(f"P={r['P']:>2} n={r['n']:>3}: mp {r['mp']:.3f}  map {r['map']:.3f}  best {r['best']:.3f}  "
                  f"map-mp {r['diff']:+.3f}±{1.96 * r['diff_se']:.3f}  pop-direction {r['pop']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
