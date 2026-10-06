"""BOAS demonstration: two-level calibration of a wearable headband against clinical PSG.

Unit of calibration is the participant. A practitioner records P calibration participants with n
paired 30-s epochs each, removes the mean direction of the pooled contrast (MP; centered rank-one
SAL and binary LEACE share its kernel), and applies the correction to new participants.

  * One night per participant, drawn with the seed in config.json.
  * Participants are split at random into a CALIBRATION pool and a TEST set, repeatedly.
  * Coordinates: whitened by the within-source covariance of the CALIBRATION pool, so the test
    participants enter only through the measured residual.
  * Target: the population mean contrast, estimated on TEST without diagonal bias (off-diagonal
    participant products, as in the paper's real-pair statistic).
  * Prediction (from the calibration pool only): Dbar ~ N(delta, Sigma_b/P + Sigma_n/P), where
    Sigma_b is the between-participant covariance and Sigma_n the covariance of a participant's
    n-epoch mean (Sigma_w/n for random epochs; estimated directly for contiguous blocks). Plug-in
    Monte Carlo, and the isotropic formula at mean noise (an upper bound under anisotropy).
  * Also reported: detection by the radial energy test (null calibrated under the pool-estimated
    noise spectrum), and the device-aware correction that subtracts the estimated mean difference.

Usage:  python3 demo.py [stage] [width] [noqc] [blocks]
Writes results/demo_results_<stage>_w<width>[_noqc][_blocks].json.
"""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from common import CFG, FEAT, RAW, RESULTS, binned, inv_sqrt, exact_risk

SEED = 20261004
MIN_SYNC_CORR = 0.5      # channel-quality rule: median delta-band peak correlation of each channel pair
P_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
N_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
SPLITS, REPEATS, MC = 20, 10, 3000
ALPHA = 0.05


def chosen_nights() -> dict[str, str]:
    """One night per participant (pid), drawn with the selection seed of config.json."""
    rows = list(csv.DictReader((RAW / "participants.tsv").open(), delimiter="\t"))
    rng = np.random.default_rng(CFG["selection"]["seed"])
    by_pid: dict[str, list[str]] = {}
    for r in rows:
        by_pid.setdefault(r["pid"], []).append(r["participant_id"])
    return {p: by_pid[p][rng.integers(len(by_pid[p]))] for p in sorted(by_pid, key=int)}


def load_people(stage: int | None, width: float, qc_rule: bool = True):
    qc = json.loads((RESULTS / "features_qc_full.json").read_text())["report"]
    people, dropped = [], []
    for pid, night in chosen_nights().items():
        corr = [float(np.median(s["peak_corr"])) for s in qc[night]["sync"].values()]
        if qc_rule and min(corr) < MIN_SYNC_CORR:
            dropped.append(dict(pid=pid, night=night, reason="channel quality", median_peak_corr=corr)); continue
        z = np.load(FEAT / f"{night}.npz")
        keep = np.ones(len(z["stage"]), bool) if stage is None else z["stage"] == stage
        if keep.sum() < max(N_GRID):
            dropped.append(dict(pid=pid, night=night, reason=f"fewer than {max(N_GRID)} epochs", n=int(keep.sum()))); continue
        order = np.argsort(z["begsample"][keep])  # time order, for contiguous blocks
        people.append(dict(pid=pid, night=night,
                           P=binned(z["power_psg"][keep], z["freqs"], width)[order],
                           H=binned(z["power_hb"][keep], z["freqs"], width)[order]))
    return people, dropped


def radial_crit(eigs: np.ndarray, rng) -> float:
    return float(np.quantile((rng.standard_normal((100000, len(eigs))) ** 2) @ eigs, 1 - ALPHA))


def offdiag(m: np.ndarray, proj=None) -> float:
    """Unbiased estimate of ||Pi delta||^2 from participant means m (rows); proj = unit vector w
    (residual after deleting w), None for the identity, or ('subtract', vector) for the device-aware correction."""
    G = len(m)
    if isinstance(proj, tuple) and proj[0] == "subtract":
        x = m - proj[1]
        S = x.sum(0)
        return float((S @ S - np.sum(x * x)) / (G * (G - 1)))
    S = m.sum(0)
    tot = S @ S - np.sum(m * m)
    if proj is None:
        return float(tot / (G * (G - 1)))
    a = m @ proj
    return float((tot - (a.sum() ** 2 - np.sum(a * a))) / (G * (G - 1)))


def split_once(people, rng):
    idx = rng.permutation(len(people)); G = len(people) // 2
    test = [people[i] for i in idx[:G]]; pool = [people[i] for i in idx[G:]]
    PP = np.concatenate([p["P"] for p in pool]); HP = np.concatenate([p["H"] for p in pool])
    W = inv_sqrt(0.5 * (np.cov(PP, rowvar=False) + np.cov(HP, rowvar=False)))
    m_test = np.stack([((p["P"] - p["H"]) @ W).mean(0) for p in test])
    Dpool = [(p["P"] - p["H"]) @ W for p in pool]
    return W, m_test, Dpool


def block_means(D: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(np.vstack([np.zeros(D.shape[1]), D]), 0)
    return (c[n:] - c[:-n]) / n


def pool_params(Dpool, blocks: bool):
    """delta: pool mean direction with the squared length estimated without the noise of the mean
    (off-diagonal products of participant means, as for the test target)."""
    means = np.stack([D.mean(0) for D in Dpool])
    raw = means.mean(0)
    theta2 = offdiag(means)
    delta = raw * np.sqrt(max(theta2, 1e-12) / (raw @ raw))
    Sw = np.mean([np.cov(D, rowvar=False) for D in Dpool], 0)
    corr = np.mean([np.cov(D, rowvar=False) / len(D) for D in Dpool], 0)
    Sb = np.cov(means, rowvar=False) - corr
    ev, U = np.linalg.eigh(Sb); Sb = (U * np.maximum(ev, 0)) @ U.T
    Sn = {}
    for n in N_GRID:
        if blocks:  # covariance of a participant's contiguous n-epoch mean around that participant's mean
            Sn[n] = np.mean([np.cov(block_means(D, n) - D.mean(0), rowvar=False) for D in Dpool], 0)
        else:
            Sn[n] = Sw / n
    return delta, Sb, Sw, Sn


def predict(delta, Sb, Sn_n, P, rng):
    C = (Sb + Sn_n) / P
    ev, U = np.linalg.eigh(C); ev = np.maximum(ev, 0); L = U * np.sqrt(ev)
    Y = delta + rng.standard_normal((MC, len(delta))) @ L.T
    u = delta / np.linalg.norm(delta)
    mp = float(np.mean(1 - (Y @ u) ** 2 / np.sum(Y ** 2, 1)))
    crit = radial_crit(ev, rng)
    power = float(np.mean(np.sum(Y ** 2, 1) > crit))
    d = len(delta); th2 = float(delta @ delta); trC = float(ev.sum())
    iso = exact_risk(d, d * th2 / trC)
    return dict(mp=mp, power=power, iso=iso, upper=trC / (trC + th2), subtract=trC / th2,
                kappa_bar=d * th2 / trC), crit


def draw(D: np.ndarray, n: int, blocks: bool, rng) -> np.ndarray:
    if blocks:
        s = rng.integers(len(D) - n + 1)
        return D[s:s + n].mean(0)
    return D[rng.choice(len(D), n, replace=False)].mean(0)


def main() -> None:
    args = sys.argv[1:]
    stage_name = args[0] if args else "N2"
    stage = {"all": None, "W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}[stage_name]
    width = float(args[1]) if len(args) > 1 else CFG["features"]["main_bin_width_hz"]
    qc_rule, blocks = "noqc" not in args, "blocks" in args
    people, dropped = load_people(stage, width, qc_rule)
    rng = np.random.default_rng(SEED)
    keys = ("mp", "sub", "det")
    cells = {(P, n): dict(per_split={k: [] for k in keys}, pred=[]) for P in P_GRID for n in N_GRID}
    split_info = []
    for s in range(SPLITS):
        W, m_test, Dpool = split_once(people, rng)
        q0 = offdiag(m_test)
        delta, Sb, Sw, Sn = pool_params(Dpool, blocks)
        d = len(delta)
        split_info.append(dict(q0_test=q0, theta2_pool=float(delta @ delta), tr_Sb=float(np.trace(Sb)),
                               tr_Sw=float(np.trace(Sw)), d=d, n_pool=len(Dpool), n_test=len(m_test),
                               kappa_one_participant_cap=float(d * (delta @ delta) / np.trace(Sb)),
                               cos2_pool_test=float((delta @ m_test.mean(0)) ** 2 / ((delta @ delta) * q0))))
        for P in P_GRID:
            if P > len(Dpool):
                continue
            for n in N_GRID:
                c = cells[(P, n)]
                pr, crit = predict(delta, Sb, Sn[n], P, rng)
                c["pred"].append(pr)
                obs = {k: [] for k in keys}
                for _ in range(REPEATS):
                    who = rng.choice(len(Dpool), P, replace=False)
                    Dbar = np.mean([draw(Dpool[i], n, blocks, rng) for i in who], 0)
                    w = Dbar / np.linalg.norm(Dbar)
                    obs["mp"].append(offdiag(m_test, w) / q0)
                    obs["sub"].append(offdiag(m_test, ("subtract", Dbar)) / q0)
                    obs["det"].append(float(Dbar @ Dbar > crit))
                for k in keys:
                    c["per_split"][k].append(float(np.mean(obs[k])))
        print(f"split {s + 1}/{SPLITS} done", flush=True)
    rows = []
    for (P, n), c in cells.items():
        if not c["pred"]:
            continue
        ps = {k: np.array(v) for k, v in c["per_split"].items()}
        pm = {k: float(np.mean([p[k] for p in c["pred"]])) for k in c["pred"][0]}
        row = dict(P=P, n=n, predicted_mp=pm["mp"], predicted_iso=pm["iso"], predicted_upper=pm["upper"],
                   predicted_subtract=pm["subtract"], predicted_detect=pm["power"], kappa_bar=pm["kappa_bar"])
        for k, name in (("mp", "mp"), ("sub", "subtract"), ("det", "detect")):
            row[f"observed_{name}"] = float(ps[k].mean())
            row[f"observed_{name}_split_sd"] = float(ps[k].std(ddof=1))
            row[f"observed_{name}_se"] = float(ps[k].std(ddof=1) / np.sqrt(len(ps[k])))
        rows.append(row)
    out = dict(stage=stage_name, width_hz=width, qc_rule=qc_rule, blocks=blocks, n_people=len(people),
               whitening="calibration pool within-source covariance", dropped=dropped,
               splits=SPLITS, repeats=REPEATS, split_info=split_info, rows=rows,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    tag = f"{stage_name}_w{width:g}" + ("" if qc_rule else "_noqc") + ("_blocks" if blocks else "")
    (RESULTS / f"demo_results_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[{tag}] people used {len(people)}, dropped {len(dropped)}")
    for r in rows:
        if r["n"] in (1, 8, 128) and r["P"] in (1, 4, 8, 32):
            print(f"P={r['P']:>2} n={r['n']:>3}: MP obs {r['observed_mp']:.3f}±{r['observed_mp_se']:.3f} pred {r['predicted_mp']:.3f} "
                  f"iso {r['predicted_iso']:.3f} | sub obs {r['observed_subtract']:.2f} pred {r['predicted_subtract']:.2f} | "
                  f"det obs {r['observed_detect']:.2f} pred {r['predicted_detect']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
