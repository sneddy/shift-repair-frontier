"""Rebuild the Neuroscan/Flex inputs of the paper from the ingested recordings.

Writes out/flex_pairs.npz (input of boas/flex_twolevel.py) and out/real_content_pools_raw.npz (the raw N170
and raw MMN pools of erasers/real_content_pools.npz), then compares them with those files when present.
The frozen-EEGPT N170 pool is not rebuilt here (see README.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "out"
SHIPPED_PAIRS = HERE.parent / "boas" / "flex_pairs.npz"
SHIPPED_POOLS = HERE.parent / "erasers" / "real_content_pools.npz"
# Participant order of the files used in the paper.
ORDER = ("1022", "1023", "1013", "1014", "1016", "1018", "1021", "1026", "1027", "1028",
         "1029", "1030", "1031", "1032", "1033", "1034")
DIMENSIONS = (16, 64)


def raw_timebins(x, times):
    """Fixed 50 ms bins, 0..750 ms, in microvolts; no fitted transform."""
    times = np.asarray(times)
    edges = np.arange(0., .750001, .05)
    blocks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        keep = (times >= lo - 1e-8) & (times < hi - 1e-8)
        if not keep.any():
            raise ValueError("empty raw ERP time bin")
        blocks.append(np.asarray(x)[..., keep].mean(-1))
    return (np.stack(blocks, axis=-1).reshape(*x.shape[:-2], -1) * 1e6).astype(np.float32)


def load_clean(person, paradigm):
    """Clean pairs (neither device above 100 uV) of one recording, or None if it does not exist."""
    path = CACHE / f"{person}_{paradigm}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        assert z["source_order"].tolist() == ["neuroscan", "flex"]
        clean = z["cleanflag"].astype(bool)
        assert np.array_equal(clean, ~z["artifact_by_device"].any(axis=1))
        return raw_timebins(z["x"], z["times"])[clean], z["event_id"][clean], z["channels"]


def collect(paradigm):
    raw, people, events, channels, missing = [], [], [], None, []
    for person in ORDER:
        record = load_clean(person, paradigm)
        if record is None:
            missing.append(person)
            continue
        raw.append(record[0])
        events.append(record[1])
        people.extend([person] * len(record[0]))
        channels = record[2]
    return np.concatenate(raw), np.asarray(people), np.concatenate(events), channels, missing


def whiten_pool(name, midpoint):
    """Center, rotate to principal axes with a fixed sign convention, and whiten the first 64 axes."""
    n = len(midpoint)
    centered = midpoint - midpoint.mean(axis=0)
    _, singular_values, right_t = np.linalg.svd(centered, full_matrices=False)
    eigenvalues, eigenvectors = singular_values ** 2 / n, right_t.T
    for j in range(eigenvectors.shape[1]):
        pivot = np.argmax(np.abs(eigenvectors[:, j]))
        if eigenvectors[pivot, j] < 0:
            eigenvectors[:, j] *= -1
    whitened = centered @ (eigenvectors[:, :max(DIMENSIONS)] / np.sqrt(eigenvalues[:max(DIMENSIONS)]))
    return {f"{name}_d{d}": np.ascontiguousarray(whitened[:, :d]) for d in DIMENSIONS}


def main():
    OUT.mkdir(exist_ok=True)
    raw, people, events, channels, missing = collect("n170")
    assert not missing, missing
    pairs = {"raw_timebins": raw, "participant": people, "event_id": events,
             "participant_order": np.asarray(ORDER), "channels": channels,
             "source_order": np.asarray(["neuroscan", "flex"])}
    np.savez_compressed(OUT / "flex_pairs.npz", **pairs)
    pools = whiten_pool("n170_raw", raw.astype(np.float64).mean(axis=1))
    mmn, _, _, _, mmn_missing = collect("mmn")
    assert mmn_missing == ["1034"], mmn_missing
    pools.update(whiten_pool("mmn_raw", mmn.astype(np.float64).mean(axis=1)))
    np.savez_compressed(OUT / "real_content_pools_raw.npz", **pools)

    report = {"pairs": len(raw), "participants": len(ORDER), "mmn_pairs": len(mmn)}
    if SHIPPED_PAIRS.exists():
        with np.load(SHIPPED_PAIRS, allow_pickle=False) as z:
            report["flex_pairs_identical"] = {k: bool(np.array_equal(pairs[k], z[k])) for k in z.files}
    if SHIPPED_POOLS.exists():
        with np.load(SHIPPED_POOLS, allow_pickle=False) as z:
            report["pools_max_abs_difference"] = {k: float(np.abs(v - z[k]).max()) for k, v in pools.items()}
    (OUT / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
