#!/usr/bin/env python3
"""Generator for the figures of the paper, from the saved results of the other folders.

Usage: python make_figures.py [fig_concept fig_benchmark ...]   (default: all; writes PDF/PNG here)
Reads ../simulations (frontier, selective, budget and certificate results), ../erasers/results
(INLP / R-LACE and real-content runs) and ../boas/results. Exact curves use ../simulations/model.py.
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP4 = HERE.parent / "simulations"
ERS = HERE.parent / "erasers" / "results"
os.environ["MPLCONFIGDIR"] = str(HERE / ".matplotlib")
sys.path.insert(0, str(EXP4))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import chi2, ncx2, f, ncf
from model import exact_risk                       # exact_risk(d, kappa, k)
from envelope_check import action_power, exact_joint
from budget_check import exact_curves

# ----------------------------------------------------------------- style
TEXTWIDTH = 6.75   # AISTATS \textwidth; figures are drawn at final size
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#7F7F7F"   # theory / ours / baseline
GREEN, ORANGE, SKY, PINK, YELLOW = "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442"
LIGHT = "#BDBDBD"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "legend.frameon": False,
    "legend.handlelength": 1.6, "legend.labelspacing": .25, "legend.borderaxespad": .3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
    "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2, "axes.titlepad": 3,
    "lines.linewidth": 1.3, "lines.markersize": 3.2, "errorbar.capsize": 1.5,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": .15, "grid.linewidth": .5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
EB = dict(lw=.7, capsize=1.5, capthick=.7)

def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight", pad_inches=.02)
    fig.savefig(HERE / f"{name}.png", bbox_inches="tight", pad_inches=.02, dpi=200)
    plt.close(fig)
    print("wrote", name)

def panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left")

def ci_err(values, intervals):
    v = np.asarray(values); iv = np.asarray(intervals)
    return np.vstack([v - iv[:, 0], iv[:, 1] - v])

# ----------------------------------------------------------------- data
def load_json(p): return json.loads(Path(p).read_text())
ISO = load_json(EXP4 / "results/isotropic.json")
THR = {(t["d"], t["nu"]): t for t in load_json(EXP4 / "envelope_results/thresholds.json")}
REPLAY = load_json(EXP4 / "envelope_results/replay.json")
SEL = load_json(EXP4 / "selective_results.json")
BUD = load_json(EXP4 / "budget_results/results.json")
with (ERS / "inlp_rlace/summary.csv").open() as fh:
    CTRL = [{k: (v if k == "method" else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]
RC_PROTO = load_json(ERS / "real_content/protocol.json")["protocol"]
with (ERS / "real_content/summary.csv").open() as fh:
    RC = [r for r in csv.DictReader(fh) if r["method"] == "MP"]
for r in RC:
    for k in ("d", "n_pairs"): r[k] = int(r[k])
    for k in ("s2", "residual_mean", "residual_mc_se"): r[k] = float(r[k])
POOLS = {"n170_raw": ("N170 raw", "o"), "n170_eegpt_pca256": ("N170 EEGPT", "s"), "mmn_raw": ("MMN raw", "^")}
ALPHA, EPS = .05, .5

def iso_rows(d): return sorted([r for r in ISO if r["d"] == d and r["kappa"] > 0], key=lambda r: r["ratio"])
def replay_rows(d): return sorted([r for r in REPLAY if r["kind"] == "known" and r["d"] == d], key=lambda r: r["kappa"])
def rc_rows(pool, d, s2): return sorted([r for r in RC if r["pool"] == pool and r["d"] == d and r["s2"] == s2], key=lambda r: r["n_pairs"])
def log2_ticks(ax, ticks, labels=None):
    ax.set_xscale("log", base=2); ax.set_xticks(ticks, labels or [("½" if t == .5 else "¼" if t == .25 else f"{t:g}") for t in ticks]); ax.minorticks_off()

# ======================================================== main-text figures
def fig_concept():
    """Figure 1 (introduction): the detection--repair wedge and the energy budget behind it."""
    from scipy.optimize import brentq
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.8), layout="constrained", gridspec_kw={"width_ratios": [1.2, 1]})
    boxed = dict(frameon=True, fancybox=False, edgecolor="#222", framealpha=1)   # thin dark frame; grey carries meaning
    # (a) wedge in the (d, kappa) plane
    ax = axes[0]
    ds = np.geomspace(16, 8192, 60)
    det = np.array([brentq(lambda k: ncx2.sf(chi2.ppf(1 - ALPHA, d), d, k) - .5, 1e-6, 50 * d) for d in ds])
    rep = np.array([brentq(lambda k: exact_risk(d, k, 1) - EPS, 1e-6, 50 * d) for d in ds])
    ax.fill_between(ds, det, rep, color=VERM, alpha=.13, lw=0)
    ax.plot(ds, det, color=GREY, label=r"$\kappa$ needed to detect the shift half the time")
    ax.plot(ds, rep, color=BLUE, label=r"$\kappa$ needed to remove half the shift with one deletion")
    on = lambda z, y: np.exp(np.interp(np.log(z), np.log(ds), np.log(y)))   # a curve's value at d = z
    ax.text(3000, on(3000, det) / 1.5, r"$\propto\sqrt{d}$", color=GREY, fontsize=6.5, ha="center", va="top")
    ax.text(2000, on(2000, rep) * 1.5, r"$\propto d$", color=BLUE, fontsize=6.5, ha="center", va="bottom")
    ax.plot([256], [128], marker="*", ms=6, color="#222", ls="none", zorder=5,   # same mark as the bar in (b)
            label=r"$d=256$, $\kappa=128$: detected 99.9%, 67% of shift left")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(16, 8192); ax.set_ylim(8, 6e4)
    ax.set_xticks([16, 64, 256, 1024, 4096], ["16", "64", "256", "1024", "4096"]); ax.minorticks_off()
    ax.set_xlabel("Representation dimension $d$"); ax.set_ylabel(r"Calibration signal-to-noise $\kappa$")
    ax.legend(loc="upper left", fontsize=6.5, **boxed).get_frame().set_linewidth(.5)
    panel_title(ax, "a", "The gap widens with dimension")
    fig.draw_without_rendering()
    mid = lambda z: np.exp(np.interp(np.log(z), np.log(ds), (np.log(det) + np.log(rep)) / 2))   # log-midline of the wedge
    xc = 1700.
    (x0, y0), (x1, y1) = ax.transData.transform([(xc / 2, mid(xc / 2)), (xc * 2, mid(xc * 2))])
    ax.text(xc, mid(xc), "detectable, not repairable", color=VERM, fontsize=6.8, rotation=np.degrees(np.arctan2(y1 - y0, x1 - x0)), rotation_mode="anchor", ha="center", va="center")
    # (b) energy budget at d=256
    ax, d = axes[1], 256
    kappas = [(48, r"$3\sqrt{d}$"), (128, r"$d/2$"), (256, r"$d$"), (768, r"$3d$")]
    x = np.arange(len(kappas)); c_alpha = chi2.ppf(1 - ALPHA, d)
    edge = ["#222" if k == 128 else "none" for k, _ in kappas]   # the case marked in (a)
    ax.bar(x, [d] * len(x), color=LIGHT, width=.6, label=r"noise (mean $d$)", edgecolor=edge, linewidth=1)
    ax.bar(x, [k for k, _ in kappas], bottom=[d] * len(x), color=SKY, width=.6, label=r"signal (mean $\kappa$)", edgecolor=edge, linewidth=1)
    ax.axhline(c_alpha, color=GREY, ls="--", lw=.8, label=r"detection threshold $c_\alpha$")
    for i, (k, lab) in enumerate(kappas):
        r = exact_risk(d, k, 1); pw = ncx2.sf(c_alpha, d, k)
        ax.text(i, d + k + 25, f"{100 * r:.0f}%", ha="center", va="bottom", fontsize=7, color=VERM, fontweight="bold")
        p = ">99.9%" if pw >= .9995 else f"{100 * pw:.1f}%" if pw > .99 else f"{100 * pw:.0f}%"
        ax.text(i, d / 2, f"detected\n{p}", ha="center", va="center", fontsize=6.5, color="#444")
        if k == 128:
            ax.plot([i], [d + k + 225], marker="*", ms=6, color="#222", zorder=5)   # the case marked in (a)
    ax.set_xticks(x, [lab for _, lab in kappas]); ax.set_xlabel(r"Calibration signal-to-noise $\kappa$ ($d=256$)")
    ax.set_ylabel(r"$\|X\|^2$ = noise + signal"); ax.set_ylim(0, 1300); ax.set_yticks([0, 256, 512, 1024], ["0", "$d$", "$2d$", "$4d$"])
    hs, ls = ax.get_legend_handles_labels()
    hs.append(Line2D([], [], ls="none", marker="s", ms=5, color=VERM)); ls.append("shift left after editing (≈ noise share)")
    ax.grid(False); ax.legend(hs, ls, loc="upper left", fontsize=6.5, **boxed).get_frame().set_linewidth(.5)
    panel_title(ax, "b", "Repair keeps the noise share")
    fig.align_xlabels(axes)
    save(fig, "concept")


def draw_method_map(ax):
    d = 32; kappa = 32.
    cell = {(c["method"], int(c["k"])): c for c in CTRL if c["d"] == d and c["n_pairs"] == d}
    bs = np.linspace(.5, 5.2, 600); fr = np.array([exact_risk(d, kappa, int(np.floor(b))) for b in bs])
    ax.fill_between(bs, .3, fr, color=LIGHT, alpha=.35, lw=0, hatch="////", edgecolor="white", step=None)
    ax.plot(bs, fr, color=BLUE, drawstyle="steps-post", label=r"exact frontier $R_d^*(b,\kappa)$", zorder=3)
    box = dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=.85)
    def pt(key, x, mk, col, lab, dx, dy, ha="left", fill=False, z=6, boxed=False):
        c = cell[key]
        ax.errorbar([x], [c["residual_mean"]], yerr=[1.96 * c["residual_se"]], fmt=mk, color=col, mfc=col if fill else "white", ms=5, mew=.9, zorder=z, **EB)
        ax.annotate(lab, (x, c["residual_mean"]), xytext=(dx, dy), textcoords="offset points", fontsize=6, color="#333", ha=ha, va="center",
                    bbox=box if boxed else None, zorder=8)
    pt(("MP", 1), 1, "o", "#222", "MP, SAL,\nR-LACE", 3, -12, boxed=True)
    pt(("LEACE", 1), cell[("LEACE", 1)]["linear_cost_median"], "D", "#222", "LEACE: same kernel,\noblique map", 6, 0)
    pt(("INLP-fixed", 1), 1, "s", "#222", "INLP, 1 step", 7, 0, fill=True)
    pt(("INLP-fixed", 4), 4, "s", "#222", "INLP, 4 steps", 7, 5, fill=True)
    pt(("R-LACE-fixed", 4), 4, "^", "#222", "R-LACE,\n4 deletions", 4, -12, z=7, boxed=True)
    f1 = exact_risk(d, kappa, 1); r_inlp = cell[("INLP-fixed", 1)]["residual_mean"]
    ax.annotate("", xy=(1, r_inlp - .012), xytext=(1, f1 + .004), arrowprops=dict(arrowstyle="<->", color=VERM, lw=.9, shrinkA=0, shrinkB=0))
    ax.text(1.07, (f1 + r_inlp) / 2 + .01, "excess 0.18, predicted\nfrom the fitted kernel", fontsize=6.5, color=VERM, ha="left", va="center")
    ax.annotate("", xy=(cell[("LEACE", 1)]["linear_cost_median"] - .12, cell[("LEACE", 1)]["residual_mean"] + .02), xytext=(1.12, cell[("LEACE", 1)]["residual_mean"] + .02),
                arrowprops=dict(arrowstyle="->", color="#555", lw=.7))
    ax.text(2.55, .37, "infeasible for any\nshared linear editor", fontsize=6, color="#555", ha="center", va="center")
    ax.set_xlim(.8, 5.3); ax.set_ylim(.33, .74); ax.set_xticks([1, 2, 3, 4, 5]); ax.set_yticks([.4, .5, .6, .7])
    ax.set_xlabel(r"Distortion cost $\|M-I\|_F^2$"); ax.set_ylabel("Population residual")
    ax.legend(loc="upper right"); panel_title(ax, "a", r"Fitted erasers at $d=N=32$")


def fig_gate_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.9), layout="constrained")
    d = 256; N = 4; nu = d * (N - 1); t = THR[(d, None)]; tu = THR[(d, nu)]
    rows = replay_rows(d); x = [r["ratio"] for r in rows]
    xs = np.linspace(0, 2.5, 220); c_alpha = chi2.ppf(1 - ALPHA, d)
    spec = [("ball", "ball_threshold", GREY, "Confidence ball"), ("envelope", "threshold", VERM, "Finite-mixture rule")]
    ax = axes[0]
    ax.plot(xs, [action_power(d, d * v, c_alpha) for v in xs], color=GREY, ls=":", lw=1, label="Detection")
    for name, key, color, lab in spec:
        ax.plot(xs, [action_power(d, d * v, t[key]) for v in xs], color=color, label=lab)
        ax.plot(xs, [action_power(d, d * v, tu[key], nu) for v in xs], color=color, ls=(0, (3, 1.5, 1, 1.5)), lw=1, label=f"same, unknown scale ($N={N}$)" if name == "envelope" else None)
        v = [r["policies"][name]["action"]["estimate"] for r in rows]
        ax.errorbar(x, v, yerr=ci_err(v, [r["policies"][name]["action"]["ci95"] for r in rows]), fmt="o", color=color, ms=2.7, **EB)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.3); ax.set_xlim(0, 2.5); ax.set_xticks([0, .5, 1, 1.5, 2, 2.5])
    ax.annotate(r"$0.029\to0.115$", xy=(1, .115), xytext=(.1, .02), fontsize=6.5, color="#333", arrowprops=dict(arrowstyle="-", color="#555", lw=.6))
    ax.set_xlabel(r"Calibration strength $\kappa/d$ ($d=256$)"); ax.set_ylabel("Action probability")
    ax.legend(loc="lower right", fontsize=6.5)
    panel_title(ax, "a", "Acts more often, still after detection")
    ax = axes[1]; peaks = []
    for name, key, color, lab in spec:
        j = np.array([exact_joint(d, d * v, t[key], EPS) for v in xs]); ax.plot(xs, j, color=color, label=lab)
        ju = np.array([exact_joint(d, d * v, tu[key], EPS, nu) for v in xs]); ax.plot(xs, ju, color=color, ls=(0, (3, 1.5, 1, 1.5)), lw=1)
        i = int(j.argmax()); ax.hlines(j[i], .4, xs[i], colors=color, ls=":", lw=.6); peaks.append(j[i])
    d2 = 64; t2 = THR[(d2, None)]
    for name, key, color, lab in spec:
        ax.plot(xs, [exact_joint(d2, d2 * v, t2[key], EPS) for v in xs], color=color, lw=.8, alpha=.45, label=f"same, $d={d2}$" if name == "envelope" else None)
    ax.axhline(ALPHA, color="#222", ls=":", lw=.8); ax.text(1.98, ALPHA + .0015, r"$\alpha=0.05$", fontsize=6.5, ha="right", va="bottom", color="#222")
    ax.set_xlim(.4, 2); ax.set_xticks([.5, 1, 1.5, 2]); ax.set_ylim(0, .056)
    ax.legend(handles=[Line2D([], [], color=VERM, label="known scale, $d=256$"), Line2D([], [], color=VERM, ls=(0, (3, 1.5, 1, 1.5)), lw=1, label="unknown scale, $d=256$"),
                       Line2D([], [], color=VERM, lw=.8, alpha=.45, label="known scale, $d=64$")], loc="upper right", fontsize=6.5, bbox_to_anchor=(1, .93))
    ax.set_yticks([0] + sorted(peaks) + [.05], ["0"] + [f"{v:.3f}" for v in sorted(peaks)] + ["0.05"])
    ax.set_xlabel(r"Calibration strength $\kappa/d$"); ax.set_ylabel("Joint bad-action probability")
    panel_title(ax, "b", "Under the same guarantee")
    save(fig, "gate_comparison")

BOAS = HERE.parent / "boas" / "results"
P_SHOW = (1, 4, 8, 32)
P_COL = {1: "#9ECAE1", 4: "#4292C6", 8: "#2171B5", 32: "#08306B"}


def draw_boas_two_level(ax):
    res = load_json(BOAS / "demo_results_N2_w0.5.json"); rows = res["rows"]; si = res["split_info"]
    d = si[0]["d"]; cap = float(np.mean([s["kappa_one_participant_cap"] for s in si]))
    for P in P_SHOW:
        r = sorted([x for x in rows if x["P"] == P], key=lambda x: x["n"]); n = np.array([x["n"] for x in r])
        ax.plot(n, [x["predicted_mp"] for x in r], color=P_COL[P], lw=1.1)
        ax.errorbar(n, [x["observed_mp"] for x in r], yerr=[1.96 * x["observed_mp_se"] for x in r], fmt="o", color=P_COL[P], ms=2.6, **EB)
        lim = exact_risk(d, P * cap, 1)
        ax.hlines(lim, 150, 230, colors=P_COL[P], ls=":", lw=.9)
        ax.text(250, lim, f"$P={P}$", fontsize=6.5, color=P_COL[P], va="center")
    log2_ticks(ax, [1, 4, 16, 64, 128], ["1", "4", "16", "64", "128"])
    ax.set_xlim(.8, 420); ax.set_ylim(0, 1.02); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlabel("Epochs per calibration participant $n$"); ax.set_ylabel("Shift left in new participants")
    panel_title(ax, "b", f"Wearable vs. clinical EEG, $d={d}$")


def draw_boas_gap(ax, n=8):
    res = load_json(BOAS / "demo_results_N2_w0.5.json")
    r = sorted([x for x in res["rows"] if x["n"] == n], key=lambda x: x["P"]); P = np.array([x["P"] for x in r])
    ax.plot(P, [x["predicted_detect"] for x in r], color=GREY, lw=1)
    ax.plot(P, [x["observed_detect"] for x in r], "^", color=GREY, ms=2.8, label="detected")
    ax.plot(P, [x["predicted_mp"] for x in r], color=BLUE, lw=1.1)
    ax.errorbar(P, [x["observed_mp"] for x in r], yerr=[1.96 * x["observed_mp_se"] for x in r], fmt="o", color=BLUE, ms=2.6, label="MP/SAL/LEACE", **EB)
    i8 = int(np.where(P == 8)[0][0]); top, bot = r[i8]["observed_detect"], r[i8]["observed_mp"]
    ax.annotate("", xy=(8, bot + .035), xytext=(8, top - .02), arrowprops=dict(arrowstyle="<->", color=VERM, lw=.8, shrinkA=0, shrinkB=0))
    ax.text(8.8, (top + bot) / 2, f"detected {int(100 * top)}%,\nhalf of shift left", fontsize=6.5, color=VERM, va="center")
    log2_ticks(ax, [1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlim(.85, 38); ax.set_ylim(0, 1.05); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlabel(f"Calibration participants $P$ ($n={n}$)"); ax.set_ylabel("Probability / shift left")
    ax.legend(loc="lower right", fontsize=6.5); panel_title(ax, "c", "Detected long before repaired")


def fig_benchmark():
    """Section 6: simulated erasers (a), and the BOAS wearable-vs-clinical shift (b, c)."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.0), layout="constrained", gridspec_kw={"width_ratios": [1.1, 1, 1]})
    draw_method_map(axes[0]); draw_boas_two_level(axes[1]); draw_boas_gap(axes[2])
    save(fig, "benchmark")


# ======================================================== appendix figures
def fig_frontier_ranks():
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.85), layout="constrained")
    d = 256; xs = np.geomspace(.03, 16, 200); rows = iso_rows(d); x = [r["ratio"] for r in rows]
    ax = axes[0]; q = chi2.ppf(1 - ALPHA, d)
    ax.plot(xs, ncx2.sf(q, d, d * xs), color=GREY, label="Detection power")
    ax.plot(xs, ncx2.sf(q / EPS, d, d * xs), color=GREY, ls=":", label="Confidence-ball action")
    ax.plot(xs, [exact_risk(d, d * v, 1) for v in xs], color=BLUE, label="Residual, cost 1")
    for key, color, mk in (("detect", GREY, "o"), ("action", GREY, "^")):
        v = [r["policies"][key]["action"]["estimate"] for r in rows]
        ax.errorbar(x, v, yerr=ci_err(v, [r["policies"][key]["action"]["ci95"] for r in rows]), fmt=mk, color=color, ms=2.6, mfc="white" if key == "action" else color, **EB)
    ax.errorbar(x, [r["frontier"][0]["empirical"]["mean"] for r in rows], yerr=[1.96 * r["frontier"][0]["empirical"]["mc_se"] for r in rows], fmt="o", color=BLUE, ms=2.6, **EB)
    ax.set_xscale("log"); ax.set_xlim(.03, 16); ax.set_ylim(-.02, 1.03); ax.set_yticks([0, .5, 1])
    ax.set_xlabel(r"Calibration strength $\kappa/d$ ($d=256$)"); ax.set_ylabel("Probability / residual")
    ax.legend(loc="center left", bbox_to_anchor=(0, .42)); panel_title(ax, "a", "Detection, certification, repair")
    ax = axes[1]
    for k, color, mk in ((1, BLUE, "o"), (64, GREEN, "s"), (128, ORANGE, "^")):
        ax.plot(xs, [exact_risk(d, d * v, k) for v in xs], color=color, label=f"Cost $k={k}$")
        s = [next(z for z in r["frontier"] if z["k"] == k)["empirical"] for r in rows]
        ax.errorbar(x, [z["mean"] for z in s], yerr=[1.96 * z["mc_se"] for z in s], fmt=mk, color=color, ms=2.7, **EB)
    ax.axhline(.5, color=LIGHT, lw=.6, ls=":")
    ax.set_xscale("log"); ax.set_xlim(.03, 16); ax.set_ylim(-.02, 1.03); ax.set_yticks([0, .5, 1])
    ax.set_xlabel(r"Calibration strength $\kappa/d$ ($d=256$)"); ax.set_ylabel("Expected residual")
    ax.legend(loc="upper right"); panel_title(ax, "b", "More deletion, lower risk"); save(fig, "frontier_ranks")


def fig_selective():
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 1.8), layout="constrained")
    handles = []
    for kappa, color in zip(SEL["signals"], (BLUE, GREEN, ORANGE)):
        rows = [r for r in SEL["rows"] if r["kappa"] == kappa]
        h, = ax.plot([r["minimum_action_frequency"] for r in rows], [r["conditional_residual"] for r in rows], color=color, label=rf"$\kappa={int(kappa)}$")
        handles.append(h)
        g = next(r for r in SEL["confidence_gate_points"] if r["kappa"] == kappa)
        ax.plot(g["minimum_action_frequency"], g["conditional_residual"], marker="D", ms=4, mfc=color, mec="white", mew=.6, ls="none", zorder=5, clip_on=False)
        ax.plot(1, rows[-1]["conditional_residual"], marker="o", ms=3.4, mfc="white", mec=color, ls="none", clip_on=False)
    ax.axhline(EPS, color=LIGHT, lw=.7, ls="--", zorder=0); ax.text(1.3e-10, EPS - .025, r"Target $\varepsilon=0.5$", fontsize=6.8, color="#555", va="top")
    handles.append(Line2D([], [], color="#444", marker="D", ls="none", ms=4, label="Confidence-ball rule"))
    ax.legend(handles=handles, loc="lower right", ncol=4)
    ax.set(xscale="log", xlim=(SEL["p_range"][0], 1.2), ylim=(0, 1), xlabel=r"Minimum action frequency $p$", ylabel="Conditional residual")
    ax.set_xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]); ax.set_yticks([0, .25, .5, .75, 1])
    save(fig, "selective_frontier")


def fig_expected_budget():
    cfg, rec, dist = BUD["config"], BUD["records"], BUD["cost_distribution"]
    d, kappa = cfg["dimension"], cfg["kappa"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.8), layout="constrained", gridspec_kw={"width_ratios": [2, 1]})
    grid = np.linspace(0, d, cfg["figure_grid_points"]); ex = [exact_curves(d, kappa, float(b)) for b in grid]
    ax = axes[0]
    ax.plot(grid, [r["hard"] for r in ex], color="#333", drawstyle="steps-post", lw=1, label="Hard budget")
    ax.plot(grid, [r["naive"] for r in ex], color=GREY, ls="--", label="Adjacent-rank mixing")
    ax.plot(grid, [r["average"] for r in ex], color=BLUE, label="Optimal average budget")
    ax.errorbar([r["budget"] for r in rec], [r["risk"]["mean"] for r in rec], yerr=[1.96 * r["risk"]["mc_se"] for r in rec], fmt="o", color=BLUE, ms=2.7, **EB)
    ax.set(xlabel="Allowed cost $b$", ylabel="Expected residual", xlim=(0, d), ylim=(-.02, 1.03), xticks=[0, 4, 8, 12, 16], yticks=[0, .5, 1])
    ax.legend(loc="upper right"); panel_title(ax, "a", rf"$d={d}$, $\kappa={kappa:g}$")
    ax = axes[1]; ranks = [1, d]
    p = [dist[str(k)]["estimate"] for k in ranks]; b = np.array([dist[str(k)]["ci95"] for k in ranks])
    ax.bar([0, 1], [.8, .2], color=[BLUE, ORANGE], alpha=.8, width=.55)
    ax.errorbar([0, 1], p, yerr=np.vstack([np.array(p) - b[:, 0], b[:, 1] - p]), fmt="o", color="#222", ms=2.5, capsize=2.5, lw=.7)
    for xv, lab in ((0, "80%"), (1, "20%")): ax.text(xv, .83 if xv == 0 else .23, lab, ha="center", fontsize=7)
    ax.set(xticks=[0, 1], xticklabels=["Delete 1", f"Delete {d}"], ylim=(0, 1.02), ylabel="Probability", yticks=[0, .5, 1])
    ax.set_axisbelow(True); panel_title(ax, "b", "Mean cost 4")
    save(fig, "expected_budget")


def fig_real_content_risk():
    cfg = RC_PROTO["config"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.9), layout="constrained", sharey=True)
    xx = np.geomspace(min(cfg["n_over_d"]), max(cfg["n_over_d"]), 160)
    colors = dict(zip(cfg["contrast_noise_variances"], (BLUE, GREEN, ORANGE)))
    for ax, d in zip(axes, cfg["dimensions"]):
        for s2 in cfg["contrast_noise_variances"]:
            ax.plot(xx, [exact_risk(d, x * d * cfg["theta"] ** 2 / s2, 1) for x in xx], color=colors[s2], label=rf"$s^2={s2:g}$", zorder=2)
            for pool, (lab, mk) in POOLS.items():
                g = rc_rows(pool, d, s2)
                ax.errorbar([r["n_pairs"] / d for r in g], [r["residual_mean"] for r in g], yerr=[1.96 * r["residual_mc_se"] for r in g],
                            fmt=mk, ms=3, mfc="none", mew=.7, color=colors[s2], ls="none", zorder=3, **EB)
        log2_ticks(ax, [.25, .5, 1, 2, 4]); ax.set_xlim(.22, 4.5); ax.set_ylim(0, .92)
        ax.set_title(f"$d={d}$"); ax.set_xlabel(r"Calibration pairs / dimension, $N/d$")
    axes[0].set_ylabel("Expected residual"); axes[0].set_yticks([0, .4, .8])
    h = axes[0].get_legend_handles_labels()[0] + [Line2D([], [], ls="none", marker=mk, mfc="none", color="#333", ms=3, label=lab) for lab, mk in POOLS.values()]
    axes[1].legend(handles=h, loc="upper right", ncol=2)
    save(fig, "real_content_risk")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            globals()[name]()
        sys.exit()
    for fn in (fig_concept, fig_benchmark, fig_gate_comparison, fig_frontier_ranks, fig_selective,
               fig_expected_budget, fig_real_content_risk):
        fn()
