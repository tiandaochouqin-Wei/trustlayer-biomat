# Trustworthy AI for biomedical materials: a model-agnostic trust layer

Code, processed data, split indices and results for the BMEMat Perspective
*"Trustworthy AI for Biomedical Materials: A Model-Agnostic Trust Layer for
Calibrated Uncertainty under Distribution Shift"*.

Everything in the paper is produced by these scripts from open data. No number
in the manuscript is typed by hand: tables are generated from the result JSONs
and figures are drawn from the same files.

## Layout

| Path | Contents |
|---|---|
| `code/common.py` | datasets, shift definitions, the split protocol, conformal procedures, metrics, models |
| `code/PROTOCOL.md` | the binding protocol every analysis follows |
| `code/trustlayer.py` | the trust layer as one decision procedure (calibrate, shift check, recalibrate, stratify, decide) |
| `code/exp_*.py` | one script per analysis (see the table below) |
| `figures/` | `style.py` (journal style) and one script per figure |
| `results/` | result JSONs, per-analysis reports (`*.md`) and `results/splits/` with every split's index sets |
| `data/` | small processed datasets and `data/get_data.py` for the ones that must be downloaded |
| `survey/` | the systematic-survey protocol, queries, screening and coding code, and the coded-paper table |

## Analyses

| Script | Section of the paper |
|---|---|
| `exp_R1_core.py` | 5.2 calibration and design-region shift (six tasks, random forest) |
| `exp_R2_models.py` | 5.4 model-agnosticism (eight model families) |
| `exp_R3_uq.py` | 5.5 uncertainty scorecard (seven uncertainty methods, full metric set) |
| `exp_R4_realshift.py` | 5.3 genuine shifts: held-out laboratories, publication years, measurement protocol |
| `exp_R4b_dft_expt.py` | 5.3 computed versus measured formation enthalpy |
| `exp_R5_imaging.py` | 5.3 histology with an external cohort |
| `exp_R6_fusion.py` | 5.7 naturally incomplete characterisation measurements |
| `exp_R7_pipeline.py` | 5.6 the layer as one decision procedure |
| `exp_R8_budget.py` | 5.8 recalibration budget and retrospective replay |
| `exp_R9_*.py` | 5.9 risk control, weighted conformal, local coverage, ESOL |

## Reproducing

```bash
conda env create -f environment.yml && conda activate trustlayer-biomat
python data/get_data.py            # downloads SciGlass (via glasspy), matminer sets, PathMNIST, Kather 2016
cd code && python exp_R1_core.py   # ~10 min on 6 cores; every other exp_*.py is run the same way
cd ../figures && python fig4_design_shift.py
```

Each script writes `results/<tag>.json` (all numbers) and `results/<tag>.md` (a
readable report). Runtimes on a 28-core workstation with one RTX 4000 Ada GPU
are given at the end of each report.

## Data sources

All data are open. `data/get_data.py` fetches them and records the versions:
SciGlass through glasspy; `steel_strength`, `expt_formation_enthalpy` and
`matbench_glass` through matminer; the polymer glass-transition set; ESOL/Delaney;
PathMNIST through medmnist; and the colorectal-cancer texture tiles of
Kather et al. 2016 from Zenodo (DOI 10.5281/zenodo.53169). Each dataset keeps the
licence of its source; see `LICENSE-note.md`.

## Conventions that matter

* Splits are grouped: replicate records of one composition never straddle a split.
* Target-region records are halved into a recalibration pool and a target-test set;
  nothing that touches target-test labels is used for calibration, budgets or thresholds.
* The conformal quantile is the finite-sample one and is `+inf` when the calibration
  set is too small (fewer than 9 records at alpha = 0.10).
* Results are reported as the mean over 20 splits with the 2.5-97.5 percentile range;
  the splits resample one dataset, so this is split-to-split variability, not a
  population confidence interval.
