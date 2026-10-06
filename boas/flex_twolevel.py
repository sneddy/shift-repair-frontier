"""Neuroscan/Flex N170 pairs (Williams et al. 2020): the BOAS two-level analysis on 16 participants.

Same design as demo.py: participants split at random into a calibration pool and a test set,
whitening by the pool's within-source covariance, prediction from the pool alone, residual of the
population mean contrast measured on test participants. Input: flex_pairs.npz, the 2,291 retained
simultaneous pairs (15 successive 50-ms means of 16 common-average-referenced channels per device).
Coordinates: a fixed orthonormal Helmert basis of the mean-zero channel subspace (225 dimensions).
Writes results/flex_twolevel.json.
Usage:  python3 flex_twolevel.py
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

import demo
from common import HERE, RESULTS

P_GRID, N_GRID = [1, 2, 3, 4, 6, 8], [1, 2, 4, 8, 16]
SPLITS, REPEATS, SEED = 40, 10, 20261006


def helmert_basis() -> np.ndarray:
    """Fixed orthonormal coordinates on the 16-channel mean-zero subspace."""
    basis = np.zeros((15, 16))
    for k in range(15):
        den = np.sqrt((k + 1) * (k + 2))
        basis[k, :k + 1] = 1 / den
        basis[k, k + 1] = -(k + 1) / den
    return basis


def main() -> None:
    basis = helmert_basis()
    people = []
    with np.load(HERE / "flex_pairs.npz") as z:
        raw, who = z["raw_timebins"].astype(np.float64), z["participant"]
        for pid in z["participant_order"]:
            x = np.einsum("kc,...ct->...kt", basis, raw[who == pid].reshape(-1, 2, 16, 15)).reshape(-1, 2, 225)
            people.append(dict(pid=str(pid), P=x[:, 0], H=x[:, 1]))
    demo.N_GRID = N_GRID
    rng = np.random.default_rng(SEED)
    cells = {(P, n): dict(mp=[], sub=[], pred=[]) for P in P_GRID for n in N_GRID}
    info = []
    for _ in range(SPLITS):
        W, m_test, Dpool = demo.split_once(people, rng)
        q0 = demo.offdiag(m_test)
        delta, Sb, Sw, Sn = demo.pool_params(Dpool, blocks=False)
        info.append(dict(q0_test=q0, theta2_pool=float(delta @ delta), tr_Sb=float(np.trace(Sb)), tr_Sw=float(np.trace(Sw)),
                         d=len(delta), kappa_one_participant_cap=float(len(delta) * (delta @ delta) / np.trace(Sb))))
        for P in P_GRID:
            for n in N_GRID:
                pr, _ = demo.predict(delta, Sb, Sn[n], P, rng)
                c = cells[(P, n)]; c["pred"].append(pr)
                mp, sub = [], []
                for _ in range(REPEATS):
                    who = rng.choice(len(Dpool), P, replace=False)
                    Dbar = np.mean([Dpool[i][rng.choice(len(Dpool[i]), n, replace=False)].mean(0) for i in who], 0)
                    mp.append(demo.offdiag(m_test, Dbar / np.linalg.norm(Dbar)) / q0)
                    sub.append(demo.offdiag(m_test, ("subtract", Dbar)) / q0)
                c["mp"].append(float(np.mean(mp))); c["sub"].append(float(np.mean(sub)))
    rows = []
    for (P, n), c in cells.items():
        pm = {k: float(np.mean([p[k] for p in c["pred"]])) for k in ("mp", "iso", "subtract", "kappa_bar")}
        rows.append(dict(P=P, n=n, observed_mp=float(np.mean(c["mp"])), observed_mp_se=float(np.std(c["mp"], ddof=1) / np.sqrt(SPLITS)),
                         observed_mp_split_sd=float(np.std(c["mp"], ddof=1)),
                         observed_subtract=float(np.mean(c["sub"])), predicted_mp=pm["mp"], predicted_iso=pm["iso"],
                         predicted_subtract=pm["subtract"], kappa_bar=pm["kappa_bar"]))
    out = dict(n_people=len(people), splits=SPLITS, repeats=REPEATS, split_info=info, rows=rows,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (RESULTS / "flex_twolevel.json").write_text(json.dumps(out, indent=1))
    print("cap kappa per participant (mean over splits):", np.mean([i["kappa_one_participant_cap"] for i in info]), "d", info[0]["d"])
    for r in rows:
        print(f"P={r['P']} n={r['n']:>2}: MP obs {r['observed_mp']:.3f}±{r['observed_mp_se']:.3f} (split sd {r['observed_mp_split_sd']:.2f}) "
              f"pred {r['predicted_mp']:.3f} iso {r['predicted_iso']:.3f} | sub obs {r['observed_subtract']:.2f} pred {r['predicted_subtract']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
