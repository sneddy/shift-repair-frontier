"""BOAS: fit the actual MP, centered rank-one SAL and binary LEACE maps to calibration endpoints.

For a few cells of the two-level grid, the three erasers are fitted to the raw (unwhitened) paired
endpoints of P calibration participants with n epochs each, using erasers.py, the
implementation of the paper's controlled experiments. Their kernels are mapped to the whitened coordinates of demo.py and
compared with the whitened mean-difference direction used there; the residual on the test
participants is computed for each fitted map. Writes results/erasers_check.json.

Usage:  python3 erasers_check.py
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

import demo
from common import RESULTS, inv_sqrt
from erasers import fit_erasers

CELLS = [(1, 128), (4, 8), (8, 8), (32, 1), (32, 128)]
SPLITS, REPEATS, SEED = 3, 5, 20261005


def kernel_vector(M: np.ndarray) -> np.ndarray:
    _, s, vt = np.linalg.svd(M)
    return vt[-1]


def main() -> None:
    people, _ = demo.load_people(2, 0.5, True)
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(SPLITS):
        idx = rng.permutation(len(people)); G = len(people) // 2
        test = [people[i] for i in idx[:G]]; pool = [people[i] for i in idx[G:]]
        PP = np.concatenate([p["P"] for p in pool]); HP = np.concatenate([p["H"] for p in pool])
        W = inv_sqrt(0.5 * (np.cov(PP, rowvar=False) + np.cov(HP, rowvar=False)))
        m_test = np.stack([((p["P"] - p["H"]) @ W).mean(0) for p in test])
        q0 = demo.offdiag(m_test)
        for P, n in CELLS:
            for _ in range(REPEATS):
                who = rng.choice(len(pool), P, replace=False)
                sel = [rng.choice(len(pool[i]["P"]), n, replace=False) for i in who]
                hp = np.concatenate([pool[i]["P"][s] for i, s in zip(who, sel)])
                hm = np.concatenate([pool[i]["H"][s] for i, s in zip(who, sel)])
                Dbar_w = (hp - hm).mean(0) @ W
                w_mp = Dbar_w / np.linalg.norm(Dbar_w)
                rec = dict(P=P, n=n, residual_whitened_mp=demo.offdiag(m_test, w_mp) / q0)
                for name, fit in fit_erasers(hp, hm).items():
                    k = kernel_vector(fit.matrix)          # kernel in raw coordinates
                    kw = W @ k                             # kernel of W M W^-1 in whitened coordinates z = W x
                    kw /= np.linalg.norm(kw)
                    rec[f"{name}_sin2_to_mp"] = float(1 - (kw @ w_mp) ** 2)
                    rec[f"{name}_residual"] = demo.offdiag(m_test, kw) / q0
                    rec[f"{name}_linear_cost"] = fit.linear_cost
                rows.append(rec)
    out = dict(cells=CELLS, splits=SPLITS, repeats=REPEATS, rows=rows,
               max_sin2_to_mp={m: max(r[f"{m}_sin2_to_mp"] for r in rows) for m in ("MP", "SAL", "LEACE")},
               max_abs_residual_diff={m: max(abs(r[f"{m}_residual"] - r["residual_whitened_mp"]) for r in rows)
                                      for m in ("MP", "SAL", "LEACE")},
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (RESULTS / "erasers_check.json").write_text(json.dumps(out, indent=1))
    print("max sin^2 to whitened MP direction:", out["max_sin2_to_mp"])
    print("max |residual - whitened MP residual|:", out["max_abs_residual_diff"])


if __name__ == "__main__":
    sys.exit(main())
