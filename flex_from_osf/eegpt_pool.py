"""Rebuild the frozen-EEGPT N170 pool of erasers/real_content_pools.npz from the ingested recordings.

1. Encode every paired epoch with the pretrained EEGPT encoder (braindecode/eegpt-pretrained, frozen):
   16 channels upsampled from 128 to 256 Hz, divided by 20 uV, full token grid [7 patches, 4 summary
   tokens, 512] flattened to 14,336 coordinates per view.
2. Fit a 256-component PCA on MMN tokens of four participants (1013, 1014, 1016, 1018): 200 pairs per
   participant drawn without replacement, both device views as rows, randomized solver with seed 17.
3. Project the N170 tokens of all 16 participants, average the two views of each clean pair, and whiten the
   midpoints as in build_npz.py (16 and 64 coordinates).

Writes out/real_content_pools_eegpt.npz and, when out/real_content_pools_raw.npz exists, the complete
out/real_content_pools.npz; compares with ../erasers/real_content_pools.npz when present.
No task labels are used. Extra dependencies: torch, braindecode, safetensors, huggingface_hub.

Usage: python eegpt_pool.py [--device cpu|mps]   (after flex_ingest.py and flex_single_gap_repair.py)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from build_npz import CACHE, ORDER, OUT, SHIPPED_POOLS, whiten_pool

REPO_ID = "braindecode/eegpt-pretrained"
REVISION = "e41cb3ae2ce4fd9eb736862292c91f8128d15618"
CHECKPOINT_SHA256 = "fb34c20609983324679b9534f9d17a2289a232d0276dd2b8b7d5b088f876f621"
PCA_SUBJECTS = ("1013", "1014", "1016", "1018")
PCA_PAIRS_PER_SUBJECT = 200
PCA_COMPONENTS = 256
SEED = 17
BATCH_SIZE = 24


def checkpoint():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO_ID, "model.safetensors", revision=REVISION)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError("EEGPT checkpoint does not match the one used in the paper")
    return path


class Encoder:
    """Frozen EEGPT returning the full encoder token grid for one-second epochs at 256 Hz."""

    def __init__(self, channels, device):
        import mne
        import torch
        from braindecode.models import EEGPT
        from safetensors.torch import load_file
        torch.manual_seed(20260909)
        info = mne.create_info(channels, 256., ch_types="eeg")
        info.set_montage("standard_1005", match_case=False)
        model = EEGPT(n_chans=len(channels), chs_info=info["chs"], n_times=256, sfreq=256.,
                      return_encoder_output=True, chan_proj_type="none")
        state = load_file(checkpoint(), device="cpu")
        own = model.state_dict()
        compatible = {k: v for k, v in state.items() if k != "chans_id" and k in own and v.shape == own[k].shape}
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if sorted(set(state) - set(compatible)) != ["chans_id"] or list(missing) != ["chans_id"] or unexpected:
            raise RuntimeError("EEGPT checkpoint does not fit the encoder")
        self.model, self.device = model.to(device).eval(), device

    def tokens(self, x):
        """[epoch, channel, 128 samples] in volts -> [epoch, 14336]."""
        import torch
        values = resample_poly(np.asarray(x, np.float32), 2, 1, axis=-1, padtype="line").astype(np.float32) / 20e-6
        parts = []
        with torch.inference_mode():
            for start in range(0, len(values), BATCH_SIZE):
                batch = torch.as_tensor(values[start:start + BATCH_SIZE], device=self.device)
                parts.append(self.model(batch).float().cpu().numpy())
        tokens = np.concatenate(parts)
        if tokens.shape[1:] != (7, 4, 512) or not np.isfinite(tokens).all():
            raise FloatingPointError(f"unexpected encoder output {tokens.shape}")
        return tokens.reshape(len(tokens), -1)


def encode(encoder, person, paradigm):
    """Tokens [pair, view, 14336] of all epochs and the clean-pair mask of one recording."""
    with np.load(CACHE / f"{person}_{paradigm}.npz", allow_pickle=False) as z:
        assert z["source_order"].tolist() == ["neuroscan", "flex"] and float(z["sfreq"]) == 128.
        x, clean = z["x"], z["cleanflag"].astype(bool)
    pairs = len(x)
    return encoder.tokens(x.reshape(-1, *x.shape[-2:])).reshape(pairs, 2, -1), clean


def fit_pca(tokens):
    """PCA on 200 pairs per participant; the draw depends only on the pair counts."""
    from sklearn.decomposition import PCA
    rows = []
    for index, person in enumerate(PCA_SUBJECTS):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 0, index]))
        chosen = np.sort(rng.choice(len(tokens[person]), size=PCA_PAIRS_PER_SUBJECT, replace=False))
        rows.append(tokens[person][chosen].reshape(-1, tokens[person].shape[-1]))
    model = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", iterated_power=2, n_oversamples=20,
                random_state=SEED).fit(np.concatenate(rows).astype(np.float32))
    return model.mean_, model.components_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--no-pca", action="store_true", help="whiten midpoints of the full token grid")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    with np.load(CACHE / f"{ORDER[0]}_n170.npz", allow_pickle=False) as z:
        channels = z["channels"].tolist()
    encoder = Encoder(channels, args.device)

    if not args.no_pca:
        center, components = fit_pca({person: encode(encoder, person, "mmn")[0] for person in PCA_SUBJECTS})
    midpoints = []
    for person in ORDER:
        tokens, clean = encode(encoder, person, "n170")
        if not args.no_pca:
            tokens = ((tokens.reshape(-1, tokens.shape[-1]) - center) @ components.T).astype(np.float32)
            tokens = tokens.reshape(len(clean), 2, -1)
        midpoints.append(tokens[clean].astype(np.float64).mean(axis=1))
        print(f"encoded {person}: {int(clean.sum())} clean N170 pairs", flush=True)
    pool = whiten_pool("n170_eegpt_pca256", np.concatenate(midpoints))

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "real_content_pools_eegpt.npz", **pool)
    report = {"device": args.device, "pca": not args.no_pca, "pairs": len(np.concatenate(midpoints))}
    raw = OUT / "real_content_pools_raw.npz"
    if raw.exists():
        with np.load(raw, allow_pickle=False) as z:
            np.savez_compressed(OUT / "real_content_pools.npz", **{k: z[k] for k in z.files}, **pool)
    if SHIPPED_POOLS.exists():
        with np.load(SHIPPED_POOLS, allow_pickle=False) as z:
            report["max_abs_difference"] = {k: float(np.abs(v - z[k]).max()) for k, v in pool.items()}
            report["mean_canonical_correlation"] = {
                k: float(np.linalg.svd(np.linalg.qr(v)[0].T @ np.linalg.qr(z[k])[0], compute_uv=False).mean())
                for k, v in pool.items()}
    (OUT / "check_eegpt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
