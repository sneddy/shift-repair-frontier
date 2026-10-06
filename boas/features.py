"""Per-night preprocessing for BOAS: synchronization check and paired spectral features.

For each night listed in participants.tsv:
  * reads the paired channels of both devices (config.signals.pairs), in microvolts;
  * synchronization: cross-correlation of the 0.5-4 Hz signals of each channel pair in
    consecutive windows, lag of the peak within +-max_lag seconds;
  * epochs: 30-s epochs from the PSG events file (human consensus stage); drops epochs with
    other stage codes, epochs the headband AI scoring marks as artifact (-2), and epochs failing
    the amplitude checks on the band-passed signal of either device;
  * features: Welch power (4-s windows, 50% overlap, 0.25 Hz resolution) on 0.5-30 Hz for each
    channel of each device, stored linear so that the analysis can bin and log them.

Writes <data>/features/<night>.npz and <night>_qc.json, and results/features_qc_full.json.
Usage:  python3 features.py
"""
from __future__ import annotations
import csv, json, sys
import mne
import numpy as np
from scipy import signal
from common import CFG, RAW, FEAT, RESULTS

SIG, SYNC, QC, FT = CFG["signals"], CFG["sync_check"], CFG["epoch_qc"], CFG["features"]
FS = SIG["sampling_frequency"]


def read(night: str, acq: str, names: list[str]) -> np.ndarray:
    path = RAW / night / "eeg" / f"{night}_task-Sleep_acq-{acq}_eeg.edf"
    raw = mne.io.read_raw_edf(path, include=names, preload=True, verbose="error")
    assert raw.info["sfreq"] == FS, (night, acq, raw.info["sfreq"])
    return raw.get_data(picks=names) * 1e6  # volts -> microvolts


def events(night: str, acq: str) -> list[dict]:
    path = RAW / night / "eeg" / f"{night}_task-Sleep_acq-{acq}_events.tsv"
    return list(csv.DictReader(path.open(), delimiter="\t"))


def sync_lags(a: np.ndarray, b: np.ndarray) -> dict:
    """Lag (s) of b relative to a maximizing the correlation, per window; positive = b later."""
    sos = signal.butter(4, SYNC["band_hz"], btype="band", fs=FS, output="sos")
    a, b = signal.sosfiltfilt(sos, a), signal.sosfiltfilt(sos, b)
    win, maxlag = int(SYNC["window_seconds"] * FS), int(SYNC["max_lag_seconds"] * FS)
    lags, peaks, zero = [], [], []
    for start in range(0, min(len(a), len(b)) - win + 1, win):
        x, y = a[start:start + win], b[start:start + win]
        x, y = (x - x.mean()) / (x.std() + 1e-12), (y - y.mean()) / (y.std() + 1e-12)
        c = signal.correlate(y, x, mode="full", method="fft") / win
        mid = win - 1
        seg = c[mid - maxlag: mid + maxlag + 1]
        k = int(np.argmax(seg))
        lags.append((k - maxlag) / FS); peaks.append(float(seg[k])); zero.append(float(c[mid]))
    return dict(lag_s=lags, peak_corr=peaks, zero_lag_corr=zero)


HB_NAMES = [p[0] for p in SIG["pairs"]]
PSG_NAMES = [p[1] for p in SIG["pairs"]]
SOS_BP = signal.butter(4, SIG["bandpass_hz"], btype="band", fs=FS, output="sos")
NPER = int(FT["welch_window_seconds"] * FS)
FREQS = np.fft.rfftfreq(NPER, 1 / FS)
KEEP = (FREQS >= FT["range_hz"][0]) & (FREQS < FT["range_hz"][1])


def process_night(night: str, pid: str) -> dict:
    """Sync check, epoch QC and Welch features for one night; writes FEAT/<night>.npz."""
    hb, psg = read(night, "headband", HB_NAMES), read(night, "psg", PSG_NAMES)
    T = min(hb.shape[1], psg.shape[1])
    hb, psg = hb[:, :T], psg[:, :T]
    sync = {f"{h}~{p}": sync_lags(psg[i], hb[i]) for i, (h, p) in enumerate(SIG["pairs"])}
    hb_f, psg_f = signal.sosfiltfilt(SOS_BP, hb, axis=1), signal.sosfiltfilt(SOS_BP, psg, axis=1)
    ev_psg, ev_hb = events(night, "psg"), events(night, "headband")
    hb_ai = {int(e["begsample"]): int(e["stage_ai"]) for e in ev_hb}
    rows, drops = [], dict(stage=0, hb_artifact=0, amplitude=0, flat=0, short=0)
    for e in ev_psg:
        b0, b1 = int(e["begsample"]) - 1, int(e["endsample"])
        st = int(e["stage_hum"])
        if b1 > T:
            drops["short"] += 1; continue
        if st not in (0, 1, 2, 3, 4):
            drops["stage"] += 1; continue
        if QC["drop_headband_ai_artifact"] and hb_ai.get(b0 + 1, 0) == -2:
            drops["hb_artifact"] += 1; continue
        seg = np.concatenate([hb_f[:, b0:b1], psg_f[:, b0:b1]])
        if np.abs(seg).max() > QC["max_abs_uv"]:
            drops["amplitude"] += 1; continue
        if seg.std(axis=1).min() < QC["min_std_uv"]:
            drops["flat"] += 1; continue
        rows.append((b0, b1, st))
    if rows:
        X = np.stack([np.concatenate([hb[:, b0:b1], psg[:, b0:b1]]) for b0, b1, _ in rows])
        f, pxx = signal.welch(X, fs=FS, nperseg=NPER, noverlap=int(NPER * FT["welch_overlap"]),
                              detrend="constant", axis=-1)
        assert np.allclose(f, FREQS)
        pxx = pxx[..., KEEP].astype(np.float64)  # (epochs, 4 channels: HB_1, HB_2, F3, F4, freqs)
    else:
        pxx = np.zeros((0, 4, int(KEEP.sum())))
    np.savez_compressed(FEAT / f"{night}.npz", power_hb=pxx[:, :2], power_psg=pxx[:, 2:],
                        freqs=FREQS[KEEP], stage=np.array([r[2] for r in rows], dtype=int),
                        begsample=np.array([r[0] for r in rows], dtype=int), pid=pid, night=night)
    rep = dict(pid=pid, seconds=T / FS, epochs_total=len(ev_psg), epochs_kept=len(rows), drops=drops,
               stage_counts={int(s): int(c) for s, c in zip(*np.unique([r[2] for r in rows], return_counts=True))},
               sync=sync)
    lag = {k: round(float(np.median(v["lag_s"])), 3) for k, v in sync.items()}
    pk = {k: round(float(np.median(v["peak_corr"])), 3) for k, v in sync.items()}
    print(f"{night}: kept {len(rows)}/{len(ev_psg)} epochs, drops {drops}, median lag {lag}, median peak corr {pk}", flush=True)
    return rep


def main() -> None:
    FEAT.mkdir(parents=True, exist_ok=True)
    report = {}
    for r in csv.DictReader((RAW / "participants.tsv").open(), delimiter="\t"):
        night, q = r["participant_id"], FEAT / f"{r['participant_id']}_qc.json"
        rep = json.loads(q.read_text()) if q.exists() else process_night(night, r["pid"])
        q.write_text(json.dumps(rep))
        rep.pop("files", None)
        report[night] = rep
    (RESULTS / "features_qc_full.json").write_text(json.dumps(dict(report=report), indent=1))
    print(f"features for {len(report)} nights")


if __name__ == "__main__":
    sys.exit(main())
