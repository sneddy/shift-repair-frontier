# Neuroscan/Flex inputs from the public recordings

Rebuilds `boas/flex_pairs.npz` and the raw N170 and raw MMN pools of `erasers/real_content_pools.npz` from
the original recordings of Williams et al. (2020), *A Validation of Emotiv EPOC Flex Saline for EEG and ERP
Research*, PeerJ 8:e9713, publicly released on OSF (https://osf.io/zj3f5/). The OSF release states no
license, so these derived files are not included in the repository; this folder regenerates them.

Extra dependency: `curryreader==0.1.2` (Neuroscan Curry files), besides NumPy, SciPy and MNE from
`../requirements.txt`.

| Step | Command (from inside this folder) | Output | Time |
|---|---|---|---|
| 1 | `python fetch_selected.py download` | `raw/Saline Raw Data/{N170,oddball_active,oddball_passive}` (383 files, 4.7 GB), checked against the OSF SHA-256 and MD5 hashes in `selected_manifest.json` | download |
| 2 | `python flex_ingest.py` | `cache/<participant>_{n170,mmn}.npz`: paired epochs of 15 participants | ~1 min |
| 3 | `python flex_single_gap_repair.py` | `cache/1034_n170.npz` (one recording with a four-event gap in the Neuroscan triggers) | seconds |
| 4 | `python build_npz.py` | `out/flex_pairs.npz`, `out/real_content_pools_raw.npz`, `out/check.json` | seconds |

`--data-root` points steps 2 and 3 to an existing copy of the recordings (the folder that contains `raw/`).

Preprocessing: 16 common scalp channels, 0.1–30 Hz zero-phase FIR filter, common-average reference,
resampling to 128 Hz, epochs from −0.2 to 0.8 s with prestimulus baseline subtraction, Neuroscan and Flex
epochs matched by behavioural trial number; a pair is dropped when either device exceeds 100 µV. Features
are fifteen 50-ms averages from 0 to 750 ms per channel (240 coordinates).

Copy `out/flex_pairs.npz` to `../boas/` and `out/real_content_pools_raw.npz` to
`../erasers/real_content_pools.npz`. When those files are already in place, `out/check.json` compares the
rebuild with them. For the files used in the paper, `flex_pairs.npz` was bit-identical and the raw pools
agreed to within 1e-12 (floating-point differences of the SVD across BLAS builds).

The frozen-EEGPT N170 pool (`n170_eegpt_pca256_*`) is not rebuilt here: it needs the pretrained
`braindecode/eegpt-pretrained` encoder and a PCA basis fitted on MMN tokens of four participants, and is
not included in this repository.
