"""BOAS: how much of the divergence between the two devices' feature distributions is a mean shift.

For the analysed participants (N2 epochs, d = 118), the Jeffreys divergence between Gaussian fits to
the polysomnograph and headband features splits into a mean part and a covariance part; each is
compared with its value under random within-pair swaps of the two devices (the estimation-noise
baseline). Computed pooled over participants and within each participant's night.
Writes results/moments.json.  Usage:  python3 moments.py
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

import demo
from common import RESULTS, jeffreys

SEED = 20261007


def share(P, H, rng, reps=5):
    a, b = jeffreys(P, H)
    null = []
    for _ in range(reps):
        f = rng.random(len(P)) < 0.5
        null.append(jeffreys(np.where(f[:, None], H, P), np.where(f[:, None], P, H)))
    na, nb = np.mean(null, 0)
    return dict(mean_part=a, cov_part=b, null_mean=float(na), null_cov=float(nb),
                mean_share=float((a - na) / ((a - na) + max(b - nb, 0.0))))


def main():
    people, _ = demo.load_people(2, 0.5, True)
    rng = np.random.default_rng(SEED)
    pooled = share(np.concatenate([p["P"] for p in people]), np.concatenate([p["H"] for p in people]), rng)
    within = [share(p["P"], p["H"], rng)["mean_share"] for p in people]
    out = dict(n_people=len(people), pooled=pooled,
               within_night=dict(median=float(np.median(within)), q10=float(np.quantile(within, .1)),
                                 q90=float(np.quantile(within, .9)), values=within),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (RESULTS / "moments.json").write_text(json.dumps(out, indent=1))
    print("pooled mean share %.3f; within-night median %.2f (10-90%%: %.2f-%.2f)" % (
        pooled["mean_share"], out["within_night"]["median"], out["within_night"]["q10"], out["within_night"]["q90"]))


if __name__ == "__main__":
    sys.exit(main())
