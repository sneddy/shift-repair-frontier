# Wearable versus clinical EEG (BOAS) and the Neuroscan/Flex pairs

Code and results for the section "A wearable versus a clinical recording" and the appendix
"Wearable versus clinical EEG (BOAS)" of the paper, including its second device pair.

Data: Bitbrain Open Access Sleep (OpenNeuro ds005555, version 1.1.3, CC0). 128 nights of 100 healthy
adults, recorded simultaneously by a clinical polysomnograph and a forehead EEG headband.
The second device pair (`flex_pairs.npz`) holds the 2,291 retained simultaneous Neuroscan/Flex N170
pairs of Williams et al. (2020). It is not included in this repository: build it with `../flex_from_osf/`
and copy `../flex_from_osf/out/flex_pairs.npz` here before running `flex_twolevel.py`.

Raw data (35.9 GB) and per-night features are not included: `download.py` and `features.py` regenerate
them; the analyses then run from the features. Environment: Python 3.12, NumPy 1.26, SciPy 1.16, MNE 1.11. Data directory: `$BOAS_DATA`, or the
first `data/boas` found above this folder (`raw/` and `features/` inside it).

| Step | Script | Output |
|---|---|---|
| 1 | `download.py [workers]` | mirror of ds005555 in `<data>/raw` (1,418 files, 35.9 GB), `manifest_full.json` |
| 2 | `features.py` | per-night features and QC in `<data>/features`, `results/features_qc_full.json` |
| 3 | `demo.py [stage] [width] [noqc] [blocks]` | `results/demo_results_<stage>_w<width>[_noqc][_blocks].json` |
| 4 | `erasers_check.py` | `results/erasers_check.json` |
| 5 | `moments.py` | `results/moments.json` |
| 6 | `flex_twolevel.py` | `results/flex_twolevel.json` |
| 7 | `pilot_plan.py` | `results/pilot_plan.json` (planning from a pilot of 3 to 8 nights) |
| 8 | `weighted_dir.py` | `results/weighted_dir.json` (noise-weighted direction, appendix on Proposition 7) |

Runs used in the paper: `demo.py N2 0.5` (main), `N2 0.5 noqc`, `N2 0.5 blocks`, `all 0.5`, `N2 2`,
`N2 1`, `N2 0.25`.

- `config.json`: dataset, night-selection seed, channel pairs, epoch checks, feature definition.
- `common.py`: paths and shared helpers (feature binning, whitening, frontier formula, Jeffreys split).
- `features.py`: alignment check (lag of peak cross-correlation of the 0.5–4 Hz signals in 5-min
  windows), epoch checks, Welch power per 30-s epoch for headband HB_1/HB_2 and polysomnograph F3/F4.
- `demo.py`: one night per participant (selection seed), channel-quality rule, participant splits
  into calibration pool and test set, whitening by the pool, two-level prediction from the pool,
  observed residual of the population mean shift on the test participants, detection, subtraction.
- `erasers.py`: the MP, centered rank-one SAL and binary LEACE implementations of the paper's
  controlled experiments; `erasers_check.py` fits them to BOAS endpoints and compares kernels.
- `moments.py`: mean and covariance parts of the Jeffreys divergence between the two devices.
- `flex_twolevel.py`: the same two-level analysis on the 16 Neuroscan/Flex participants.
- `pilot_plan.py`: three disjoint sets per split (20 for whitening without pairs, 35 calibration, 30 test);
  theta^2, tr Sigma_b and tr Sigma_w from a pilot of P0 full nights predict the whole grid.
- `weighted_dir.py`: deletes (I + g C)^{-1} Dbar, with g from the posterior mode or tuned on the test set,
  on separate splits of the main design.

## Preprocessing details

- Devices: Micromed polysomnograph and Bitbrain headband (channels near AF7/AF8), both referenced to the
  left mastoid, 256 Hz. Pairs HB_1~F3 and HB_2~F4.
- Epochs (30 s, consensus human staging): dropped if the stage is missing, if the headband's automatic
  scoring marks an artifact (-2), or if either device exceeds 500 uV or has standard deviation below
  0.5 uV in a 0.3-35 Hz band-passed copy.
- Features: Welch power with 4-s windows and 50% overlap, averaged in bins (0.5 Hz: 59 bins per channel,
  d = 118), log10.
- Alignment: lag of peak cross-correlation of the 0.5-4 Hz signals in 5-min windows; median absolute
  lag 0.008 s over channel pairs and nights. Channel rule: a night is excluded if the median peak
  correlation of either channel pair is below 0.5 (14 of 100 nights).
- Design: participants and epochs drawn without replacement; 10 calibration draws per cell and split;
  3,000 Monte Carlo draws for the plug-in prediction; kernels of MP, SAL and LEACE compared on 75 draws.
  Detection: radial statistic ||Dbar||^2 against the 0.95 quantile of its null law under the
  pool-estimated covariance (100,000 draws).
- Neuroscan/Flex: 16 common channels, 0.1-30 Hz zero-phase filter, common-average reference, 128 Hz,
  1-s epochs from -0.2 s with baseline subtraction, pairs dropped above 100 uV; fifteen 50-ms averages per
  channel (240 coordinates) and an orthonormal Helmert basis removing the reference dependency, d = 225.

Seeds are fixed in the scripts; every result file records the SHA-256 of the script that wrote it.
