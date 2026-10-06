"""Exploratory: can a small pilot plan the calibration? Three scalars, three disjoint sets.

The two-level law needs only theta^2, tr Sigma_b and tr Sigma_w:
    kappa_bar(P, n) = P theta^2 d / (tr Sigma_b + tr Sigma_w / n),
and the predicted eraser residual is the frontier formula at kappa_bar.
Each split divides the 85 participants into three disjoint sets:
  * moments (20): whitening by the within-source covariance, each device separately (no pairs used);
  * calibration (35): pilots and calibration draws come from here;
  * test (30): the residual of the population mean shift is measured here only.
A pilot is P0 calibration participants with their full N2 night; the three scalars come from the
pilot alone (theta^2 by off-diagonal products of participant means). The prediction is compared
with the observed residual of the erasers fitted to P calibration participants with n epochs each.
For reference, the same prediction from all 35 calibration participants (scalars and full plug-in).

Usage:  python3 pilot_plan.py
Writes results/pilot_plan.json.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
from common import RESULTS, inv_sqrt, exact_risk
from demo import P_GRID, N_GRID, SEED, load_people, offdiag, draw, pool_params, predict

SPLITS, REPEATS, PILOTS = 20, 10, 5
N_MOM, N_TEST = 20, 30
P0_GRID = [3, 4, 6, 8]


def scalars(Ds):
    """theta^2, tr Sigma_b, tr Sigma_w from whitened per-participant contrasts (full nights)."""
    means = np.stack([D.mean(0) for D in Ds])
    th2 = max(offdiag(means), 0.0)
    trw = [float(np.trace(np.cov(D, rowvar=False))) for D in Ds]
    trSw = float(np.mean(trw))
    trSb = float(np.trace(np.cov(means, rowvar=False)) - np.mean([t / len(D) for t, D in zip(trw, Ds)]))
    return th2, max(trSb, 1e-6), trSw


def scalar_prediction(th2, trSb, trSw, d):
    out = {}
    for P in P_GRID:
        for n in N_GRID:
            kb = P * th2 * d / (trSb + trSw / n)
            out[(P, n)] = exact_risk(d, kb)
    return out


def main() -> None:
    people, _ = load_people(2, 0.5, True)
    rng = np.random.default_rng(SEED + 2)
    G = len(people)
    obs = {c: [] for c in [(P, n) for P in P_GRID for n in N_GRID]}
    ref_scalar = {c: [] for c in obs}; ref_plugin = {c: [] for c in obs}
    pilot = {P0: [] for P0 in P0_GRID}          # list of dicts: per pilot predictions + scalars
    split_scalars = []
    for s in range(SPLITS):
        idx = rng.permutation(G)
        mom = [people[i] for i in idx[:N_MOM]]
        test = [people[i] for i in idx[N_MOM:N_MOM + N_TEST]]
        cal = [people[i] for i in idx[N_MOM + N_TEST:]]
        PP = np.concatenate([p["P"] for p in mom]); HH = np.concatenate([p["H"] for p in mom])
        W = inv_sqrt(0.5 * (np.cov(PP, rowvar=False) + np.cov(HH, rowvar=False)))
        m_test = np.stack([((p["P"] - p["H"]) @ W).mean(0) for p in test])
        q0 = offdiag(m_test)
        Dcal = [(p["P"] - p["H"]) @ W for p in cal]
        d = Dcal[0].shape[1]
        # observed residuals
        for (P, n) in obs:
            vals = []
            for _ in range(REPEATS):
                who = rng.choice(len(Dcal), P, replace=False)
                Dbar = np.mean([draw(Dcal[i], n, False, rng) for i in who], 0)
                vals.append(offdiag(m_test, Dbar / np.linalg.norm(Dbar)) / q0)
            obs[(P, n)].append(float(np.mean(vals)))
        # reference: all calibration participants
        th2, trSb, trSw = scalars(Dcal)
        split_scalars.append(dict(theta2=th2, trSb=trSb, trSw=trSw, cap=th2 * d / trSb, n_cal=len(Dcal)))
        sp = scalar_prediction(th2, trSb, trSw, d)
        delta, Sb, Sw, Sn = pool_params(Dcal, False)
        for c in obs:
            ref_scalar[c].append(sp[c])
            ref_plugin[c].append(predict(delta, Sb, Sn[c[1]], c[0], rng)[0]["mp"])
        # pilots
        for P0 in P0_GRID:
            for _ in range(PILOTS):
                who = rng.choice(len(Dcal), P0, replace=False)
                th2p, trSbp, trSwp = scalars([Dcal[i] for i in who])
                pp = scalar_prediction(th2p, trSbp, trSwp, d)
                pilot[P0].append(dict(split=s, theta2=th2p, trSb=trSbp, trSw=trSwp, cap=th2p * d / trSbp,
                                      pred={f"{P},{n}": v for (P, n), v in pp.items()}))
        print(f"split {s + 1}/{SPLITS} done", flush=True)

    cells = list(obs)
    o = np.array([[obs[c][s] for c in cells] for s in range(SPLITS)])           # (splits, cells)
    def rms_vs_obs(pred_by_split):
        e = pred_by_split - o
        return float(np.sqrt(np.mean(e.mean(0) ** 2))), float(np.sqrt(np.mean(e ** 2)))
    summary = {}
    rs = np.array([[ref_scalar[c][s] for c in cells] for s in range(SPLITS)])
    rp = np.array([[ref_plugin[c][s] for c in cells] for s in range(SPLITS)])
    summary["calibration_set_scalars"] = dict(zip(("rms_of_mean", "rms_per_split"), rms_vs_obs(rs)))
    summary["calibration_set_plugin"] = dict(zip(("rms_of_mean", "rms_per_split"), rms_vs_obs(rp)))
    for P0 in P0_GRID:
        per = []
        for pl in pilot[P0]:
            pv = np.array([pl["pred"][f"{P},{n}"] for (P, n) in cells])
            per.append(float(np.sqrt(np.mean((pv - o[pl["split"]]) ** 2))))
        caps = [pl["cap"] for pl in pilot[P0]]
        summary[f"pilot_{P0}"] = dict(rms_median=float(np.median(per)), rms_q10=float(np.quantile(per, .1)),
                                      rms_q90=float(np.quantile(per, .9)), cap_median=float(np.median(caps)),
                                      cap_q10=float(np.quantile(caps, .1)), cap_q90=float(np.quantile(caps, .9)))
    rows = []
    for j, (P, n) in enumerate(cells):
        row = dict(P=P, n=n, observed=float(o[:, j].mean()), observed_se=float(o[:, j].std(ddof=1) / np.sqrt(SPLITS)),
                   pred_cal_scalars=float(rs[:, j].mean()), pred_cal_plugin=float(rp[:, j].mean()))
        for P0 in P0_GRID:
            v = np.array([pl["pred"][f"{P},{n}"] for pl in pilot[P0]])
            row[f"pilot{P0}_median"] = float(np.median(v))
            row[f"pilot{P0}_q10"] = float(np.quantile(v, .1)); row[f"pilot{P0}_q90"] = float(np.quantile(v, .9))
        rows.append(row)
    out = dict(design=f"moments {N_MOM} / calibration {G - N_MOM - N_TEST} / test {N_TEST} participants, "
                      f"{SPLITS} splits, {PILOTS} pilots per split and size",
               summary=summary, split_scalars=split_scalars, rows=rows,
               pilots={str(k): [{kk: vv for kk, vv in pl.items() if kk != "pred"} for pl in v] for k, v in pilot.items()},
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (RESULTS / "pilot_plan.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r["n"] in (8, 128) and r["P"] in (1, 4, 8, 32):
            print(f"P={r['P']:>2} n={r['n']:>3}: obs {r['observed']:.3f}  cal-scalars {r['pred_cal_scalars']:.3f}  "
                  + "  ".join(f"pilot{P0} {r[f'pilot{P0}_median']:.3f} [{r[f'pilot{P0}_q10']:.2f},{r[f'pilot{P0}_q90']:.2f}]"
                              for P0 in P0_GRID))


if __name__ == "__main__":
    sys.exit(main())
