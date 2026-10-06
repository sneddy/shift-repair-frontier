"""Shared paths, configuration and numerical helpers for the BOAS analyses.

Data location: the directory given by the environment variable BOAS_DATA, or else the first
`data/boas` directory found above this folder. It holds `raw/` (the OpenNeuro mirror written by
download.py) and `features/` (written by features.py).
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())
RESULTS = HERE / "results"


def _data_dir() -> Path:
    if os.environ.get("BOAS_DATA"):
        return Path(os.environ["BOAS_DATA"])
    for parent in HERE.parents:
        if (parent / "data" / "boas").is_dir():
            return parent / "data" / "boas"
    return HERE / "data"


DATA = _data_dir()
RAW, FEAT = DATA / "raw", DATA / "features"


def binned(power: np.ndarray, freqs: np.ndarray, width: float) -> np.ndarray:
    """log10 of Welch power averaged within frequency bins of the given width, channels concatenated."""
    lo, hi = CFG["features"]["range_hz"]
    nb = int(np.floor((hi - lo) / width + 1e-9))
    idx = np.floor((freqs - lo) / width + 1e-9).astype(int)
    out = np.stack([power[..., idx == b].mean(-1) for b in range(nb)], axis=-1)  # (n, ch, nb)
    return np.log10(out).reshape(len(power), -1)


def inv_sqrt(S: np.ndarray) -> np.ndarray:
    ev, U = np.linalg.eigh(S)
    return U @ np.diag(ev ** -0.5) @ U.T


def exact_risk(d: int, kappa: float) -> float:
    """Cost-one frontier (d-1) E[1/(d+2J)], J ~ Poisson(kappa/2)."""
    lam = kappa / 2
    j = np.arange(max(0, int(lam - 12 * np.sqrt(lam + 1) - 30)), int(lam + 12 * np.sqrt(lam + 1) + 60))
    return float((d - 1) * np.sum(stats.poisson.pmf(j, lam) / (d + 2 * j)))


def jeffreys(P: np.ndarray, H: np.ndarray) -> tuple[float, float]:
    """Mean and covariance parts of the Jeffreys divergence between Gaussian fits to P and H."""
    dmu = P.mean(0) - H.mean(0)
    SP, SH = np.cov(P, rowvar=False), np.cov(H, rowvar=False)
    iP, iH = np.linalg.inv(SP), np.linalg.inv(SH)
    d = P.shape[1]
    mean_part = 0.5 * dmu @ (iP + iH) @ dmu
    cov_part = 0.5 * (np.trace(iP @ SH) + np.trace(iH @ SP)) - d
    return float(mean_part), float(cov_part)
