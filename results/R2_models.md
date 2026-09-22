# R2_models: empirical model-agnosticism (Reviewer 2, comment 5)

Script: `revision/code/exp_R2_models.py`; results: `revision/results/R2_models.json`. Protocol: `revision/code/PROTOCOL.md` / `common.py` (grouped 60/20/20 source split, target pool halves = recalibration pool / target-test, headline m = clip(floor(|T|/3), 10, 30), alpha = 0.1, finite-sample conformal quantile, 20 seeds). Cells are mean [2.5, 97.5 percentile] across the 20 splits; the splits resample one finite dataset each, so the interval describes split-to-split variability, not population sampling error. Hyperparameters are fixed a priori in `common.make_model` (no tuning); nothing is selected with target-test labels. GP = exact GP (isotropic RBF + white noise) fitted on at most 2,500 randomly subsampled training records (`common._SubsampledGP`; affects glass_Tg, glass_Tliq, polymer_Tg); RF/ExtraTrees 300 trees. Compute: 190 worker-minutes over 960 (model, dataset, split) tasks. Cache fingerprint (common.py + the compute section of this script): `a58e66b98067008f`.

Datasets: glass_Tg (source 6592, target 258, m = 30); glass_E (source 3544, target 67, m = 22); glass_HV (source 1473, target 71, m = 23); glass_Tliq (source 7760, target 131, m = 30); steel_yield (source 196, target 41, m = 13); polymer_Tg (source 4332, target 963, m = 30)

## Key question: does the collapse and the recovery hold for every model family?

- In-distribution split-conformal coverage is 0.887-0.936 across all 48 (model, dataset) cells (nominal 0.90), as guaranteed for any predictor.
- Source-calibrated coverage on the design-region target-test half ranges 0.131-0.781; 48/48 cells fall below 0.80.
- Target recalibration with the headline m restores mean coverage to 0.875-0.979; 48/48 cells reach >= 0.85.
- Source->target coverage range across the 8 models, per dataset: glass_Tg 0.37-0.56; glass_E 0.16-0.34; glass_HV 0.16-0.39; glass_Tliq 0.30-0.63; steel_yield 0.13-0.36; polymer_Tg 0.37-0.78.
- The recalibration budget is counted in target RECORDS, not in distinct compositions: on glass_Tg 29.0/30 at m and 8.4/9 at k = 9; glass_Tliq 28.5/30 at m and 8.7/9 at k = 9; polymer_Tg 22.6/30 at m and 6.7/9 at k = 9 distinct compositions on average, so the effective calibration size is smaller than k there (on glass_E, glass_HV, steel_yield k records are k compositions). Drawing the same k from k DISTINCT compositions changes the recalibrated coverage by +0.000 to +0.041 (mean +0.008) on those datasets.

Verdict thresholds (fixed before running, on the mean over splits): collapse = source-calibrated target-test coverage < 0.80; recovery = recalibrated coverage >= 0.85.

Caveats (48/48 cells meet BOTH verdict thresholds; the bullets below are not failed cells, they qualify how the thresholds are met):

- RF / polymer_Tg: collapse on average (source->target 0.763, below the threshold) but not in every split: 15 % of splits have source->target coverage >= 0.80 (upper 97.5 % 0.82)
- ExtraTrees / polymer_Tg: collapse on average (source->target 0.781, below the threshold) but not in every split: 30 % of splits have source->target coverage >= 0.80 (upper 97.5 % 0.82)
- HistGB / polymer_Tg: collapse on average (source->target 0.778, below the threshold) but not in every split: 25 % of splits have source->target coverage >= 0.80 (upper 97.5 % 0.82)
- kNN / steel_yield: in-distribution coverage 0.936 deviates > 0.02 from 0.90. With n_cal = 39 the exact finite-sample expectation is ceil((n_cal+1)(1-alpha))/(n_cal+1) = 0.900 and marginal coverage is guaranteed in [0.900, 0.925]; with only 39 source-test records the per-split SD under this split scheme is 0.042. Over 380 additional in-distribution-only splits (seeds 20-399; no target record is read) the same cell averages 0.903 (SD 0.065, SE 0.003), i.e. not above the guaranteed upper bound 0.925 (within 2 SE of it), so the 20-split value is split-sampling noise, not systematic over-coverage of the procedure
- Ridge / steel_yield: in-distribution coverage 0.928 deviates > 0.02 from 0.90. With n_cal = 39 the exact finite-sample expectation is ceil((n_cal+1)(1-alpha))/(n_cal+1) = 0.900 and marginal coverage is guaranteed in [0.900, 0.925]; with only 39 source-test records the per-split SD under this split scheme is 0.048. Over 380 additional in-distribution-only splits (seeds 20-399; no target record is read) the same cell averages 0.895 (SD 0.069, SE 0.004), i.e. not above the guaranteed upper bound 0.925 (within 2 SE of it), so the 20-split value is split-sampling noise, not systematic over-coverage of the procedure
- MLP / polymer_Tg: collapse on average (source->target 0.682, below the threshold) but not in every split: 25 % of splits have source->target coverage >= 0.80 (upper 97.5 % 0.84)

## How the families differ (generated from the numbers below)

- glass_Tg: source-test RMSE best ExtraTrees 27.1 / worst Ridge 44; target-test RMSE best ExtraTrees 98.9 / worst MLP 136; ID normalised width 0.66 (ExtraTrees)-1.12 (Ridge); recalibrated normalised width 2.48 (Ridge)-3.55 (MLP); least collapse Ridge (0.56), most SVR (0.37).
- glass_E: source-test RMSE best HistGB 5.55 / worst Ridge 8.63; target-test RMSE best kNN 15.7 / worst Ridge 24.9; ID normalised width 0.73 (SVR)-1.32 (GP); recalibrated normalised width 2.84 (GP)-5.33 (MLP); least collapse GP (0.34), most SVR (0.16).
- glass_HV: source-test RMSE best ExtraTrees 0.702 / worst Ridge 0.969; target-test RMSE best GP 1.89 / worst SVR 2.71; ID normalised width 1.57 (ExtraTrees)-2.62 (Ridge); recalibrated normalised width 4.34 (GP)-5.60 (MLP); least collapse kNN (0.39), most SVR (0.16).
- glass_Tliq: source-test RMSE best ExtraTrees 45.2 / worst Ridge 93.1; target-test RMSE best SVR 169 / worst GP 208; ID normalised width 0.71 (ExtraTrees)-1.55 (Ridge); recalibrated normalised width 2.39 (SVR)-3.24 (MLP); least collapse Ridge (0.63), most kNN (0.30).
- steel_yield: source-test RMSE best ExtraTrees 101 / worst MLP 123; target-test RMSE best Ridge 467 / worst HistGB 617; ID normalised width 1.57 (GP)-1.99 (Ridge); recalibrated normalised width 2.42 (Ridge)-3.32 (kNN); least collapse Ridge (0.36), most ExtraTrees (0.13).
- polymer_Tg: source-test RMSE best GP 49 / worst Ridge 59.7; target-test RMSE best HistGB 68.9 / worst GP 143; ID normalised width 1.45 (SVR)-1.83 (Ridge); recalibrated normalised width 1.86 (ExtraTrees)-3.80 (GP); least collapse ExtraTrees (0.78), most GP (0.37).
- Across the 48 cells the severity of the collapse tracks how much the model's error inflates under the shift: Spearman rho(source->target coverage, target/source RMSE ratio) = -0.48 pooled, and -0.95 to -0.40 within the six datasets (8 models each), so the association is not an artefact of pooling datasets of different difficulty. RMSE inflates 1.4-5.6x in every cell. (The corresponding rho with the recalibrated/source conformal quantile ratio, -0.62, is close to tautological - the source-calibrated coverage is largely determined by that ratio - and is reported only for completeness; the quantile inflates 1.4-6.6x.)
- Price of recovery: recalibrated 90 % intervals are 1.86-5.60 target-IQR wide (in-distribution 0.66-2.62 source-IQR); coverage is restored by widening, not by the model becoming accurate in the design region. In absolute terms they span 35-170 % of the FULL observed property range of the target pool (worst model per dataset: glass_Tg 100 %; glass_E 170 %; glass_HV 131 %; glass_Tliq 48 %; steel_yield 156 %; polymer_Tg 71 %), and 23/48 cells exceed 100 %, i.e. the honest interval covers the whole design region and cannot rank candidates.
- Model-agnostic normalised conformal (|r|/(sigma+beta)): in-distribution coverage 0.886-0.933. Without any target label it raises source-calibrated target coverage in 48/48 cells (median +0.30; range 0.38-1.00), because target inputs lie 2.1-7.7x farther from the training data than source-test inputs, but reaches >= 0.85 in only 15/48 cells. Recalibrated on the same m target records it gives 0.881-0.969 (48/48 cells >= 0.85).
- Model-agnostic triage (rank by sigma): RMSE reduction at 50 % retained is positive in 48/48 cells in-distribution (3 to 42 %) and 46/48 on target-test (-1 to 57 %); relative AURC < 1 in 48/48 (ID) and 45/48 (target) cells. Cells with no benefit on target-test (reduction <= 1 %): HistGB/glass_HV (-1 %), SVR/glass_HV (0 %), Ridge/steel_yield (-0 %).
- Native-uncertainty triage as reference (mean over datasets): RF native 43 % ID / 7 % target (worst glass_Tliq -18 %) vs model-agnostic 24 % / 25 %; ExtraTrees native 41 % ID / 8 % target (worst glass_Tliq -18 %) vs model-agnostic 24 % / 22 %; GP native 29 % ID / 28 % target (worst glass_HV 10 %) vs model-agnostic 29 % / 29 %.
- GP native Gaussian 90 % intervals (no conformal step): in-distribution coverage 0.881-0.936, target-test 0.659-0.856. The GP's own variance grows away from the data and so under-covers less than source-calibrated conformal, but still falls below nominal on every dataset; it is not a substitute for target recalibration.

Consistency check: the RF row uses the same splits and model as `exp_R1_core.py`; the maximum absolute difference to `R1_core.json` in ID / source->target / recalibrated coverage (and relative target RMSE) is 0.

## Split-conformal coverage (alpha = 0.10, nominal 0.90)

### (a) In-distribution coverage on source-test

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.90 [0.88, 0.92] | 0.90 [0.87, 0.91] | 0.90 [0.83, 0.95] | 0.90 [0.87, 0.92] | 0.90 [0.73, 0.99] | 0.90 [0.88, 0.93] |
| ExtraTrees (tree ensemble) | 0.90 [0.88, 0.92] | 0.89 [0.86, 0.92] | 0.89 [0.84, 0.96] | 0.90 [0.87, 0.92] | 0.90 [0.76, 0.99] | 0.90 [0.88, 0.93] |
| HistGB (tree ensemble (boosting)) | 0.90 [0.87, 0.92] | 0.90 [0.87, 0.92] | 0.90 [0.86, 0.95] | 0.90 [0.88, 0.91] | 0.89 [0.72, 0.99] | 0.90 [0.88, 0.92] |
| GP (kernel (Bayesian)) | 0.90 [0.88, 0.92] | 0.90 [0.87, 0.93] | 0.90 [0.86, 0.95] | 0.90 [0.87, 0.92] | 0.90 [0.82, 0.97] | 0.90 [0.88, 0.92] |
| SVR (kernel) | 0.90 [0.88, 0.93] | 0.90 [0.87, 0.92] | 0.89 [0.84, 0.95] | 0.90 [0.88, 0.92] | 0.90 [0.85, 0.97] | 0.90 [0.88, 0.92] |
| kNN (instance-based) | 0.90 [0.89, 0.91] | 0.90 [0.87, 0.92] | 0.90 [0.85, 0.95] | 0.90 [0.88, 0.91] | 0.94 [0.85, 0.99] | 0.91 [0.89, 0.92] |
| Ridge (linear) | 0.90 [0.87, 0.92] | 0.89 [0.87, 0.92] | 0.90 [0.86, 0.94] | 0.90 [0.87, 0.91] | 0.93 [0.85, 1.00] | 0.90 [0.87, 0.93] |
| MLP (neural) | 0.90 [0.88, 0.92] | 0.90 [0.88, 0.92] | 0.90 [0.83, 0.95] | 0.90 [0.88, 0.92] | 0.91 [0.74, 1.00] | 0.90 [0.88, 0.93] |

### (b) Source-calibrated coverage on target-test (design shift)

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.40 [0.33, 0.48] | 0.23 [0.12, 0.33] | 0.21 [0.14, 0.29] | 0.41 [0.33, 0.49] | 0.15 [0.02, 0.34] | 0.76 [0.72, 0.82] |
| ExtraTrees (tree ensemble) | 0.40 [0.34, 0.46] | 0.17 [0.07, 0.26] | 0.22 [0.11, 0.30] | 0.46 [0.38, 0.54] | 0.13 [0.02, 0.29] | 0.78 [0.74, 0.82] |
| HistGB (tree ensemble (boosting)) | 0.42 [0.31, 0.50] | 0.20 [0.14, 0.30] | 0.20 [0.13, 0.32] | 0.31 [0.18, 0.50] | 0.14 [0.02, 0.29] | 0.78 [0.74, 0.82] |
| GP (kernel (Bayesian)) | 0.38 [0.30, 0.44] | 0.34 [0.18, 0.50] | 0.26 [0.17, 0.34] | 0.54 [0.41, 0.64] | 0.21 [0.10, 0.43] | 0.37 [0.32, 0.45] |
| SVR (kernel) | 0.37 [0.27, 0.43] | 0.16 [0.11, 0.26] | 0.16 [0.07, 0.21] | 0.56 [0.47, 0.65] | 0.27 [0.10, 0.53] | 0.45 [0.35, 0.60] |
| kNN (instance-based) | 0.43 [0.36, 0.51] | 0.33 [0.24, 0.44] | 0.39 [0.20, 0.57] | 0.30 [0.21, 0.39] | 0.32 [0.17, 0.55] | 0.71 [0.67, 0.74] |
| Ridge (linear) | 0.56 [0.50, 0.64] | 0.28 [0.19, 0.39] | 0.31 [0.23, 0.40] | 0.63 [0.55, 0.69] | 0.36 [0.19, 0.62] | 0.70 [0.63, 0.76] |
| MLP (neural) | 0.39 [0.28, 0.49] | 0.29 [0.15, 0.47] | 0.27 [0.11, 0.59] | 0.36 [0.16, 0.49] | 0.31 [0.09, 0.58] | 0.68 [0.43, 0.84] |

### (c) Coverage after target recalibration with headline m (first m records of the recalibration pool), target-test

Finite-sample expectation under exchangeability of m calibration records: glass_Tg 0.903, glass_E 0.913, glass_HV 0.917, glass_Tliq 0.903, steel_yield 0.929, polymer_Tg 0.903. This expectation assumes the m calibration records are exchangeable with the test records; on glass_Tg the m = 30 records come from 29.0 distinct compositions on average, glass_Tliq the m = 30 records come from 28.5 distinct compositions on average, polymer_Tg the m = 30 records come from 22.6 distinct compositions on average, so the effective calibration size there is smaller than m (see the replicate-structure section).

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.93 [0.84, 0.99] | 0.92 [0.75, 1.00] | 0.93 [0.84, 1.00] | 0.90 [0.75, 0.99] | 0.95 [0.83, 1.00] | 0.90 [0.82, 0.98] |
| ExtraTrees (tree ensemble) | 0.92 [0.83, 0.98] | 0.91 [0.71, 0.97] | 0.93 [0.83, 1.00] | 0.88 [0.73, 0.98] | 0.94 [0.78, 1.00] | 0.88 [0.75, 0.96] |
| HistGB (tree ensemble (boosting)) | 0.92 [0.83, 0.99] | 0.89 [0.68, 0.97] | 0.94 [0.86, 1.00] | 0.89 [0.73, 0.98] | 0.94 [0.81, 1.00] | 0.90 [0.81, 0.97] |
| GP (kernel (Bayesian)) | 0.91 [0.81, 0.97] | 0.90 [0.77, 0.97] | 0.94 [0.84, 1.00] | 0.88 [0.78, 0.95] | 0.97 [0.88, 1.00] | 0.89 [0.76, 0.98] |
| SVR (kernel) | 0.93 [0.84, 0.98] | 0.89 [0.74, 1.00] | 0.92 [0.78, 1.00] | 0.89 [0.80, 0.95] | 0.98 [0.88, 1.00] | 0.88 [0.73, 0.97] |
| kNN (instance-based) | 0.92 [0.81, 0.96] | 0.88 [0.77, 0.96] | 0.92 [0.74, 1.00] | 0.89 [0.78, 0.97] | 0.98 [0.88, 1.00] | 0.88 [0.72, 0.98] |
| Ridge (linear) | 0.93 [0.82, 0.97] | 0.92 [0.83, 1.00] | 0.94 [0.81, 1.00] | 0.90 [0.79, 0.98] | 0.92 [0.64, 1.00] | 0.89 [0.75, 0.98] |
| MLP (neural) | 0.92 [0.79, 0.98] | 0.94 [0.80, 1.00] | 0.94 [0.86, 1.00] | 0.89 [0.75, 0.99] | 0.95 [0.78, 1.00] | 0.88 [0.70, 0.96] |

## Interval width (normalised by the IQR of the evaluated domain's property values)

### In-distribution normalised width (source-test)

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.75 [0.71, 0.82] | 0.80 [0.70, 0.88] | 1.71 [1.36, 2.22] | 0.78 [0.70, 0.85] | 1.71 [1.29, 2.23] | 1.51 [1.43, 1.64] |
| ExtraTrees (tree ensemble) | 0.66 [0.61, 0.74] | 0.73 [0.63, 0.82] | 1.57 [1.30, 2.09] | 0.71 [0.63, 0.78] | 1.61 [1.23, 2.39] | 1.48 [1.38, 1.59] |
| HistGB (tree ensemble (boosting)) | 0.72 [0.65, 0.82] | 0.83 [0.74, 0.95] | 1.73 [1.52, 2.21] | 0.74 [0.67, 0.78] | 1.65 [1.20, 2.43] | 1.47 [1.41, 1.60] |
| GP (kernel (Bayesian)) | 0.81 [0.73, 0.85] | 1.32 [1.07, 1.59] | 1.79 [1.55, 2.23] | 0.86 [0.77, 0.95] | 1.57 [1.01, 2.53] | 1.46 [1.37, 1.54] |
| SVR (kernel) | 0.71 [0.64, 0.76] | 0.73 [0.61, 0.81] | 1.77 [1.58, 2.15] | 0.83 [0.75, 0.90] | 1.70 [1.15, 2.69] | 1.45 [1.38, 1.55] |
| kNN (instance-based) | 0.89 [0.84, 0.94] | 0.93 [0.77, 1.06] | 1.88 [1.58, 2.32] | 0.91 [0.84, 0.95] | 1.97 [1.54, 2.78] | 1.59 [1.52, 1.67] |
| Ridge (linear) | 1.12 [1.07, 1.18] | 1.06 [0.96, 1.15] | 2.62 [2.45, 2.99] | 1.55 [1.45, 1.66] | 1.99 [1.32, 2.81] | 1.83 [1.73, 1.95] |
| MLP (neural) | 0.74 [0.67, 0.89] | 0.90 [0.84, 0.96] | 1.90 [1.56, 2.31] | 0.77 [0.70, 0.83] | 1.91 [1.11, 2.79] | 1.47 [1.36, 1.59] |

### Target-test normalised width: source-calibrated -> recalibrated (m)

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.61 -> 2.72 [2.16, 3.24] | 0.76 -> 3.66 [3.20, 4.50] | 1.40 -> 5.14 [3.15, 7.49] | 0.71 -> 2.83 [1.93, 3.85] | 0.56 -> 3.08 [2.80, 3.39] | 1.33 -> 2.08 [1.52, 3.01] |
| ExtraTrees (tree ensemble) | 0.54 -> 2.49 [2.06, 2.88] | 0.68 -> 3.24 [2.97, 3.45] | 1.28 -> 4.64 [2.86, 7.92] | 0.64 -> 2.68 [1.65, 3.46] | 0.53 -> 2.79 [2.54, 3.13] | 1.31 -> 1.86 [1.22, 2.65] |
| HistGB (tree ensemble (boosting)) | 0.58 -> 2.48 [1.95, 3.46] | 0.79 -> 4.16 [3.13, 4.90] | 1.41 -> 4.69 [3.11, 7.86] | 0.67 -> 3.20 [1.94, 4.34] | 0.54 -> 2.98 [2.52, 3.36] | 1.30 -> 1.92 [1.49, 2.65] |
| GP (kernel (Bayesian)) | 0.65 -> 2.85 [2.44, 3.22] | 1.24 -> 2.84 [2.66, 3.09] | 1.46 -> 4.34 [3.03, 8.11] | 0.79 -> 2.45 [1.70, 3.15] | 0.52 -> 3.22 [3.00, 3.31] | 1.29 -> 3.80 [2.74, 4.98] |
| SVR (kernel) | 0.57 -> 3.17 [2.58, 3.85] | 0.68 -> 3.17 [2.96, 3.43] | 1.44 -> 5.46 [3.84, 9.51] | 0.76 -> 2.39 [1.69, 3.72] | 0.56 -> 3.20 [2.96, 3.38] | 1.28 -> 3.19 [2.23, 4.41] |
| kNN (instance-based) | 0.72 -> 2.77 [2.09, 3.20] | 0.88 -> 3.14 [2.65, 3.76] | 1.53 -> 4.53 [3.54, 7.84] | 0.83 -> 2.51 [2.07, 2.98] | 0.65 -> 3.32 [3.02, 3.62] | 1.40 -> 2.19 [1.44, 3.19] |
| Ridge (linear) | 0.91 -> 2.48 [2.04, 2.93] | 1.00 -> 4.66 [3.81, 5.97] | 2.14 -> 4.47 [3.41, 7.15] | 1.41 -> 2.85 [2.38, 3.38] | 0.65 -> 2.42 [1.80, 3.30] | 1.62 -> 2.84 [1.81, 3.76] |
| MLP (neural) | 0.60 -> 3.55 [2.49, 4.94] | 0.85 -> 5.33 [3.77, 8.84] | 1.55 -> 5.60 [2.39, 8.40] | 0.70 -> 3.24 [2.21, 6.01] | 0.63 -> 2.67 [1.56, 4.10] | 1.30 -> 2.19 [1.39, 3.70] |

### Recalibrated width as a percentage of the FULL observed target-pool property range

An interval wider than the whole observed range of the property in the design region cannot discriminate between candidates, however honest its coverage (Reviewer 2, comment 7).

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 77 % | 117 % | 120 % | 42 % | 145 % | 39 % |
| ExtraTrees (tree ensemble) | 70 % | 103 % | 109 % | 40 % | 131 % | 35 % |
| HistGB (tree ensemble (boosting)) | 70 % | 133 % | 110 % | 47 % | 140 % | 36 % |
| GP (kernel (Bayesian)) | 80 % | 91 % | 101 % | 36 % | 151 % | 71 % |
| SVR (kernel) | 89 % | 101 % | 128 % | 35 % | 150 % | 60 % |
| kNN (instance-based) | 78 % | 100 % | 106 % | 37 % | 156 % | 41 % |
| Ridge (linear) | 70 % | 148 % | 105 % | 42 % | 114 % | 53 % |
| MLP (neural) | 100 % | 170 % | 131 % | 48 % | 125 % | 41 % |

## Point accuracy

### RMSE on source-test (dataset units)

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 30.0 [27.8, 32.9] | 6.0 [5.2, 6.9] | 0.7 [0.6, 0.9] | 49.5 [46.0, 54.4] | 109.1 [82.8, 141.1] | 49.7 [47.3, 52.0] |
| ExtraTrees (tree ensemble) | 27.1 [24.5, 30.0] | 5.6 [4.8, 6.6] | 0.7 [0.5, 0.8] | 45.2 [41.8, 49.1] | 101.3 [78.2, 125.7] | 49.1 [46.2, 51.4] |
| HistGB (tree ensemble (boosting)) | 28.3 [25.8, 31.0] | 5.5 [4.7, 6.4] | 0.7 [0.6, 0.9] | 47.9 [44.3, 51.7] | 111.4 [85.8, 137.7] | 49.2 [47.0, 52.1] |
| GP (kernel (Bayesian)) | 32.4 [29.4, 35.5] | 8.0 [6.9, 9.8] | 0.8 [0.6, 0.9] | 56.1 [53.1, 59.9] | 108.2 [84.9, 141.3] | 49.0 [46.8, 52.0] |
| SVR (kernel) | 30.0 [27.9, 34.1] | 7.3 [6.2, 8.7] | 0.8 [0.6, 0.9] | 52.0 [49.3, 54.4] | 114.9 [85.8, 140.6] | 49.4 [47.2, 52.6] |
| kNN (instance-based) | 33.8 [31.4, 36.5] | 6.4 [5.5, 7.2] | 0.8 [0.6, 1.0] | 56.3 [52.4, 59.3] | 118.2 [84.3, 157.7] | 51.7 [49.9, 53.8] |
| Ridge (linear) | 44.0 [41.8, 47.3] | 8.6 [7.4, 9.9] | 1.0 [0.9, 1.1] | 93.1 [88.5, 97.5] | 121.7 [97.4, 151.7] | 59.7 [57.2, 62.2] |
| MLP (neural) | 30.4 [27.0, 35.7] | 6.8 [5.6, 7.9] | 0.8 [0.6, 0.9] | 49.1 [45.9, 53.5] | 122.7 [99.1, 170.1] | 49.3 [46.9, 53.1] |

### RMSE on target-test (dataset units)

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 107.5 [95.8, 120.4] | 19.7 [16.8, 23.7] | 2.1 [1.9, 2.3] | 190.9 [144.2, 221.9] | 577.5 [468.3, 706.6] | 74.5 [68.2, 81.2] |
| ExtraTrees (tree ensemble) | 98.9 [90.5, 113.4] | 18.2 [16.0, 20.0] | 2.0 [1.6, 2.2] | 184.1 [141.5, 212.7] | 556.6 [474.6, 662.0] | 70.0 [66.2, 75.1] |
| HistGB (tree ensemble (boosting)) | 99.3 [87.1, 119.8] | 20.7 [16.6, 25.8] | 2.0 [1.8, 2.3] | 204.1 [159.1, 236.6] | 616.7 [526.7, 707.7] | 68.9 [64.6, 74.3] |
| GP (kernel (Bayesian)) | 117.9 [106.5, 129.9] | 16.2 [14.8, 17.7] | 1.9 [1.7, 2.2] | 207.7 [146.5, 257.1] | 557.6 [457.8, 669.0] | 142.7 [125.6, 156.8] |
| SVR (kernel) | 123.9 [105.5, 143.5] | 18.1 [16.3, 20.2] | 2.7 [2.3, 3.1] | 169.2 [127.8, 193.7] | 547.5 [441.2, 660.4] | 121.3 [102.7, 140.7] |
| kNN (instance-based) | 106.8 [98.3, 115.8] | 15.7 [13.2, 17.8] | 1.9 [1.5, 2.2] | 196.9 [152.0, 230.8] | 573.0 [496.3, 709.0] | 82.7 [76.8, 88.4] |
| Ridge (linear) | 99.9 [92.6, 113.2] | 24.9 [19.9, 32.6] | 2.1 [1.8, 2.5] | 202.6 [152.1, 235.5] | 467.4 [335.2, 555.0] | 99.4 [91.4, 108.3] |
| MLP (neural) | 136.1 [108.5, 165.9] | 24.8 [15.1, 35.8] | 2.6 [1.3, 3.8] | 206.1 [177.4, 240.1] | 491.8 [327.8, 736.7] | 80.9 [64.0, 113.6] |

### RMSE ratio target-test / source-test

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 3.6 [3.0, 4.0] | 3.3 [2.7, 4.0] | 2.9 [2.4, 3.6] | 3.9 [2.9, 4.6] | 5.4 [4.0, 8.0] | 1.5 [1.4, 1.7] |
| ExtraTrees (tree ensemble) | 3.7 [3.2, 4.1] | 3.3 [2.7, 3.8] | 2.8 [2.1, 3.5] | 4.1 [3.1, 4.9] | 5.6 [4.2, 7.3] | 1.4 [1.3, 1.5] |
| HistGB (tree ensemble (boosting)) | 3.5 [3.0, 4.1] | 3.8 [3.0, 4.5] | 2.7 [2.1, 3.4] | 4.3 [3.3, 5.0] | 5.6 [4.4, 7.7] | 1.4 [1.3, 1.6] |
| GP (kernel (Bayesian)) | 3.6 [3.2, 4.2] | 2.0 [1.7, 2.5] | 2.5 [2.0, 3.0] | 3.7 [2.6, 4.7] | 5.3 [3.8, 7.6] | 2.9 [2.4, 3.2] |
| SVR (kernel) | 4.1 [3.5, 4.9] | 2.5 [2.0, 3.1] | 3.6 [2.7, 4.5] | 3.3 [2.4, 3.7] | 4.9 [3.5, 7.2] | 2.5 [2.1, 2.9] |
| kNN (instance-based) | 3.2 [2.9, 3.6] | 2.5 [2.1, 3.0] | 2.4 [1.9, 3.0] | 3.5 [2.7, 4.3] | 5.0 [3.7, 7.4] | 1.6 [1.5, 1.7] |
| Ridge (linear) | 2.3 [2.0, 2.6] | 2.9 [2.1, 4.0] | 2.2 [1.8, 2.7] | 2.2 [1.6, 2.6] | 3.9 [2.8, 5.1] | 1.7 [1.5, 1.8] |
| MLP (neural) | 4.5 [3.1, 5.7] | 3.7 [1.9, 5.7] | 3.4 [1.4, 5.9] | 4.2 [3.6, 4.9] | 4.1 [2.6, 7.3] | 1.6 [1.3, 2.3] |

## Interval score (Gneiting & Raftery 2007, alpha = 0.10; lower is better; dataset units)

### Interval score: ID / source->target / recalibrated

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 148.5 / 1085.2 / 438.4 | 29.0 / 239.8 / 64.1 | 3.7 / 20.9 / 9.4 | 244.5 / 1671.8 / 924.9 | 518.8 / 6453.3 / 2239.1 | 230.6 / 378.5 / 341.6 |
| ExtraTrees (tree ensemble) | 133.7 / 1009.9 / 400.2 | 26.6 / 225.9 / 57.9 | 3.5 / 18.7 / 9.0 | 224.9 / 1608.7 / 912.9 | 493.5 / 6581.5 / 2030.5 | 228.4 / 352.2 / 324.5 |
| HistGB (tree ensemble (boosting)) | 138.0 / 973.1 / 410.8 | 26.6 / 249.0 / 80.0 | 3.7 / 18.2 / 8.8 | 232.5 / 1963.6 / 972.2 | 537.0 / 7713.3 / 2179.9 | 226.7 / 342.3 / 313.9 |
| GP (kernel (Bayesian)) | 160.3 / 1207.4 / 466.7 | 41.2 / 135.4 / 52.5 | 3.8 / 15.9 / 8.4 | 280.8 / 1549.5 / 1035.9 | 528.2 / 6052.8 / 2302.4 | 226.4 / 1240.0 / 583.1 |
| SVR (kernel) | 147.8 / 1331.7 / 501.4 | 32.3 / 225.0 / 57.6 | 3.9 / 31.4 / 10.1 | 259.0 / 1278.9 / 876.9 | 562.2 / 5603.7 / 2287.4 | 232.3 / 937.5 / 508.1 |
| kNN (instance-based) | 166.7 / 981.6 / 442.8 | 31.3 / 157.6 / 58.7 | 4.0 / 14.7 / 8.5 | 275.6 / 1673.7 / 895.3 | 573.2 / 5637.7 / 2367.8 | 236.9 / 429.3 / 362.7 |
| Ridge (linear) | 201.7 / 778.1 / 409.3 | 39.2 / 292.6 / 77.9 | 4.5 / 14.6 / 8.0 | 423.0 / 1288.8 / 965.2 | 571.7 / 4427.2 / 1950.3 | 263.9 / 537.4 / 448.9 |
| MLP (neural) | 145.8 / 1406.2 / 578.7 | 32.1 / 292.3 / 92.8 | 3.9 / 28.2 / 9.7 | 237.0 / 1949.2 / 1008.2 | 608.9 / 5018.5 / 2015.1 | 227.0 / 458.8 / 354.9 |

### Mean interval calibration error over levels 0.50-0.95: ID / source->target / recalibrated

At level 0.95 the m-record quantile is infinite when m < 19 (steel, m = 13); such intervals count as covering (reported, not dropped).

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.014 / 0.462 / 0.064 | 0.018 / 0.618 / 0.079 | 0.034 / 0.598 / 0.088 | 0.013 / 0.477 / 0.058 | 0.069 / 0.649 / 0.110 | 0.016 / 0.172 / 0.058 |
| ExtraTrees (tree ensemble) | 0.014 / 0.468 / 0.067 | 0.018 / 0.656 / 0.077 | 0.034 / 0.603 / 0.076 | 0.012 / 0.420 / 0.058 | 0.068 / 0.666 / 0.107 | 0.017 / 0.148 / 0.057 |
| HistGB (tree ensemble (boosting)) | 0.015 / 0.462 / 0.067 | 0.017 / 0.627 / 0.085 | 0.036 / 0.601 / 0.071 | 0.013 / 0.551 / 0.060 | 0.072 / 0.656 / 0.106 | 0.016 / 0.137 / 0.060 |
| GP (kernel (Bayesian)) | 0.013 / 0.499 / 0.068 | 0.017 / 0.551 / 0.072 | 0.029 / 0.543 / 0.073 | 0.012 / 0.375 / 0.060 | 0.060 / 0.601 / 0.121 | 0.016 / 0.481 / 0.076 |
| SVR (kernel) | 0.015 / 0.502 / 0.064 | 0.018 / 0.649 / 0.080 | 0.027 / 0.647 / 0.085 | 0.012 / 0.376 / 0.056 | 0.065 / 0.573 / 0.116 | 0.017 / 0.427 / 0.074 |
| kNN (instance-based) | 0.012 / 0.458 / 0.070 | 0.016 / 0.529 / 0.074 | 0.030 / 0.490 / 0.084 | 0.012 / 0.527 / 0.062 | 0.052 / 0.514 / 0.111 | 0.014 / 0.220 / 0.070 |
| Ridge (linear) | 0.015 / 0.304 / 0.066 | 0.020 / 0.529 / 0.073 | 0.032 / 0.476 / 0.079 | 0.013 / 0.212 / 0.057 | 0.063 / 0.486 / 0.107 | 0.018 / 0.216 / 0.080 |
| MLP (neural) | 0.013 / 0.470 / 0.063 | 0.015 / 0.570 / 0.072 | 0.033 / 0.566 / 0.066 | 0.011 / 0.500 / 0.066 | 0.080 / 0.530 / 0.097 | 0.019 / 0.234 / 0.065 |

## k-sweep (coverage on target-test; k = 0 is the source-calibrated interval)

Ranges of the per-cell mean and of the per-cell 2.5th percentile over the 20 splits - k = 9: mean 0.84-0.95, 2.5th percentile across cells 0.557-0.847 (48 cells); k = 10: mean 0.84-0.95, 2.5th percentile across cells 0.557-0.893 (48 cells); k = 20: mean 0.86-0.93, 2.5th percentile across cells 0.619-0.864 (48 cells); k = 30: mean 0.87-0.93, 2.5th percentile across cells 0.665-0.842 (40 cells). A small budget restores coverage on average while single splits can still under-cover badly.

### k = 0

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.40 [0.33, 0.48] | 0.23 [0.12, 0.33] | 0.21 [0.14, 0.29] | 0.41 [0.33, 0.49] | 0.15 [0.02, 0.34] | 0.76 [0.72, 0.82] |
| ExtraTrees (tree ensemble) | 0.40 [0.34, 0.46] | 0.17 [0.07, 0.26] | 0.22 [0.11, 0.30] | 0.46 [0.38, 0.54] | 0.13 [0.02, 0.29] | 0.78 [0.74, 0.82] |
| HistGB (tree ensemble (boosting)) | 0.42 [0.31, 0.50] | 0.20 [0.14, 0.30] | 0.20 [0.13, 0.32] | 0.31 [0.18, 0.50] | 0.14 [0.02, 0.29] | 0.78 [0.74, 0.82] |
| GP (kernel (Bayesian)) | 0.38 [0.30, 0.44] | 0.34 [0.18, 0.50] | 0.26 [0.17, 0.34] | 0.54 [0.41, 0.64] | 0.21 [0.10, 0.43] | 0.37 [0.32, 0.45] |
| SVR (kernel) | 0.37 [0.27, 0.43] | 0.16 [0.11, 0.26] | 0.16 [0.07, 0.21] | 0.56 [0.47, 0.65] | 0.27 [0.10, 0.53] | 0.45 [0.35, 0.60] |
| kNN (instance-based) | 0.43 [0.36, 0.51] | 0.33 [0.24, 0.44] | 0.39 [0.20, 0.57] | 0.30 [0.21, 0.39] | 0.32 [0.17, 0.55] | 0.71 [0.67, 0.74] |
| Ridge (linear) | 0.56 [0.50, 0.64] | 0.28 [0.19, 0.39] | 0.31 [0.23, 0.40] | 0.63 [0.55, 0.69] | 0.36 [0.19, 0.62] | 0.70 [0.63, 0.76] |
| MLP (neural) | 0.39 [0.28, 0.49] | 0.29 [0.15, 0.47] | 0.27 [0.11, 0.59] | 0.36 [0.16, 0.49] | 0.31 [0.09, 0.58] | 0.68 [0.43, 0.84] |

### k = 9

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.89 [0.71, 1.00] | 0.92 [0.74, 1.00] | 0.92 [0.80, 1.00] | 0.89 [0.71, 1.00] | 0.93 [0.78, 1.00] | 0.85 [0.65, 0.99] |
| ExtraTrees (tree ensemble) | 0.88 [0.70, 1.00] | 0.92 [0.82, 1.00] | 0.91 [0.77, 1.00] | 0.89 [0.70, 1.00] | 0.92 [0.76, 1.00] | 0.87 [0.62, 0.99] |
| HistGB (tree ensemble (boosting)) | 0.87 [0.64, 1.00] | 0.88 [0.68, 1.00] | 0.89 [0.76, 1.00] | 0.90 [0.70, 1.00] | 0.92 [0.81, 1.00] | 0.88 [0.65, 0.99] |
| GP (kernel (Bayesian)) | 0.89 [0.72, 1.00] | 0.92 [0.76, 1.00] | 0.91 [0.69, 1.00] | 0.89 [0.67, 1.00] | 0.93 [0.78, 1.00] | 0.84 [0.56, 0.99] |
| SVR (kernel) | 0.89 [0.70, 1.00] | 0.91 [0.79, 1.00] | 0.88 [0.67, 1.00] | 0.89 [0.74, 1.00] | 0.94 [0.78, 1.00] | 0.86 [0.67, 0.99] |
| kNN (instance-based) | 0.90 [0.69, 1.00] | 0.90 [0.73, 1.00] | 0.90 [0.71, 1.00] | 0.91 [0.74, 1.00] | 0.93 [0.78, 1.00] | 0.84 [0.58, 0.99] |
| Ridge (linear) | 0.89 [0.68, 1.00] | 0.92 [0.80, 1.00] | 0.89 [0.66, 1.00] | 0.91 [0.75, 1.00] | 0.90 [0.59, 1.00] | 0.87 [0.60, 1.00] |
| MLP (neural) | 0.89 [0.64, 1.00] | 0.94 [0.85, 1.00] | 0.92 [0.77, 1.00] | 0.90 [0.70, 1.00] | 0.95 [0.76, 1.00] | 0.86 [0.60, 0.99] |

### k = 10

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.90 [0.71, 1.00] | 0.93 [0.77, 1.00] | 0.93 [0.83, 1.00] | 0.91 [0.71, 1.00] | 0.93 [0.78, 1.00] | 0.86 [0.72, 0.99] |
| ExtraTrees (tree ensemble) | 0.89 [0.70, 1.00] | 0.93 [0.82, 1.00] | 0.91 [0.77, 1.00] | 0.91 [0.70, 1.00] | 0.92 [0.76, 1.00] | 0.88 [0.68, 0.99] |
| HistGB (tree ensemble (boosting)) | 0.89 [0.66, 1.00] | 0.92 [0.71, 1.00] | 0.90 [0.76, 1.00] | 0.91 [0.72, 1.00] | 0.92 [0.81, 1.00] | 0.89 [0.66, 0.99] |
| GP (kernel (Bayesian)) | 0.89 [0.72, 1.00] | 0.93 [0.82, 1.00] | 0.91 [0.69, 1.00] | 0.91 [0.72, 1.00] | 0.94 [0.78, 1.00] | 0.84 [0.56, 0.99] |
| SVR (kernel) | 0.90 [0.70, 1.00] | 0.91 [0.79, 1.00] | 0.89 [0.67, 1.00] | 0.90 [0.76, 1.00] | 0.94 [0.78, 1.00] | 0.86 [0.68, 0.99] |
| kNN (instance-based) | 0.90 [0.69, 1.00] | 0.90 [0.73, 1.00] | 0.90 [0.71, 1.00] | 0.92 [0.75, 1.00] | 0.94 [0.78, 1.00] | 0.86 [0.65, 0.99] |
| Ridge (linear) | 0.89 [0.68, 1.00] | 0.92 [0.80, 1.00] | 0.90 [0.70, 1.00] | 0.92 [0.76, 1.00] | 0.92 [0.64, 1.00] | 0.88 [0.67, 1.00] |
| MLP (neural) | 0.90 [0.64, 1.00] | 0.95 [0.89, 1.00] | 0.92 [0.77, 1.00] | 0.91 [0.71, 1.00] | 0.95 [0.76, 1.00] | 0.86 [0.60, 0.99] |

### k = 20

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.92 [0.84, 0.98] | 0.90 [0.73, 1.00] | 0.92 [0.80, 1.00] | 0.89 [0.72, 1.00] | 0.92 [0.76, 1.00] | 0.89 [0.77, 0.97] |
| ExtraTrees (tree ensemble) | 0.91 [0.82, 0.98] | 0.89 [0.69, 0.97] | 0.91 [0.79, 1.00] | 0.89 [0.69, 1.00] | 0.91 [0.74, 1.00] | 0.86 [0.74, 0.96] |
| HistGB (tree ensemble (boosting)) | 0.92 [0.83, 0.99] | 0.89 [0.68, 0.97] | 0.93 [0.86, 1.00] | 0.89 [0.71, 1.00] | 0.90 [0.73, 1.00] | 0.89 [0.73, 0.99] |
| GP (kernel (Bayesian)) | 0.92 [0.80, 0.98] | 0.90 [0.77, 0.97] | 0.92 [0.75, 1.00] | 0.88 [0.76, 1.00] | 0.93 [0.74, 1.00] | 0.89 [0.77, 0.98] |
| SVR (kernel) | 0.93 [0.86, 0.98] | 0.88 [0.74, 0.99] | 0.89 [0.68, 1.00] | 0.89 [0.76, 1.00] | 0.93 [0.74, 1.00] | 0.87 [0.71, 0.98] |
| kNN (instance-based) | 0.93 [0.78, 0.98] | 0.86 [0.69, 0.96] | 0.89 [0.69, 1.00] | 0.88 [0.76, 1.00] | 0.92 [0.74, 1.00] | 0.88 [0.64, 0.98] |
| Ridge (linear) | 0.93 [0.81, 0.99] | 0.92 [0.82, 1.00] | 0.92 [0.73, 1.00] | 0.90 [0.74, 1.00] | 0.89 [0.62, 1.00] | 0.89 [0.75, 0.98] |
| MLP (neural) | 0.92 [0.79, 0.99] | 0.93 [0.80, 1.00] | 0.92 [0.78, 1.00] | 0.88 [0.70, 0.99] | 0.92 [0.73, 1.00] | 0.88 [0.73, 0.99] |

### k = 30

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.93 [0.84, 0.99] | 0.91 [0.75, 1.00] | 0.91 [0.79, 0.99] | 0.90 [0.75, 0.99] | n/a | 0.90 [0.82, 0.98] |
| ExtraTrees (tree ensemble) | 0.92 [0.83, 0.98] | 0.88 [0.71, 0.97] | 0.91 [0.79, 1.00] | 0.88 [0.73, 0.98] | n/a | 0.88 [0.75, 0.96] |
| HistGB (tree ensemble (boosting)) | 0.92 [0.83, 0.99] | 0.89 [0.68, 0.99] | 0.91 [0.84, 0.97] | 0.89 [0.73, 0.98] | n/a | 0.90 [0.81, 0.97] |
| GP (kernel (Bayesian)) | 0.91 [0.81, 0.97] | 0.90 [0.75, 0.97] | 0.92 [0.81, 0.99] | 0.88 [0.78, 0.95] | n/a | 0.89 [0.76, 0.98] |
| SVR (kernel) | 0.93 [0.84, 0.98] | 0.87 [0.67, 1.00] | 0.91 [0.76, 0.99] | 0.89 [0.80, 0.95] | n/a | 0.88 [0.73, 0.97] |
| kNN (instance-based) | 0.92 [0.81, 0.96] | 0.87 [0.68, 0.94] | 0.89 [0.73, 1.00] | 0.89 [0.78, 0.97] | n/a | 0.88 [0.72, 0.98] |
| Ridge (linear) | 0.93 [0.82, 0.97] | 0.90 [0.76, 0.99] | 0.91 [0.74, 1.00] | 0.90 [0.79, 0.98] | n/a | 0.89 [0.75, 0.98] |
| MLP (neural) | 0.92 [0.79, 0.98] | 0.91 [0.79, 0.99] | 0.92 [0.84, 0.99] | 0.89 [0.75, 0.99] | n/a | 0.88 [0.70, 0.96] |

## Replicate structure of the recalibration budget (Reviewer 2, comments 10-11)

`common.design_split` permutes composition GROUPS, so a composition never sits on both sides of a split, but the protocol's recalibration set ("the first k records of the pool") can contain several replicate measurements of the same composition. Replicates share a descriptor vector and therefore a prediction, so their residuals are strongly correlated and the EFFECTIVE calibration size is smaller than k. The counts below are measured per split (mean [2.5, 97.5 percentile] over the 20 splits); they depend only on the split, not on the model.

| dataset | records m | distinct compositions in the m records | k = 9 | k = 10 | k = 20 | k = 30 |
|---|---|---|---|---|---|---|
| glass_Tg | 30 | 29.0 [26.5, 30.0] | 8.4 [6.5, 9.0] | 9.4 [7.5, 10.0] | 19.1 [16.5, 20.0] | 29.0 [26.5, 30.0] |
| glass_E | 22 | 22.0 [22.0, 22.0] | 9.0 [9.0, 9.0] | 10.0 [10.0, 10.0] | 20.0 [20.0, 20.0] | 30.0 [30.0, 30.0] |
| glass_HV | 23 | 23.0 [23.0, 23.0] | 9.0 [9.0, 9.0] | 10.0 [10.0, 10.0] | 20.0 [20.0, 20.0] | 30.0 [30.0, 30.0] |
| glass_Tliq | 30 | 28.5 [26.5, 30.0] | 8.7 [7.5, 9.0] | 9.6 [8.0, 10.0] | 19.4 [17.5, 20.0] | 28.5 [26.5, 30.0] |
| steel_yield | 13 | 13.0 [13.0, 13.0] | 9.0 [9.0, 9.0] | 10.0 [10.0, 10.0] | 20.0 [20.0, 20.0] | n/a |
| polymer_Tg | 30 | 22.6 [19.5, 25.5] | 6.7 [3.5, 9.0] | 7.3 [4.5, 9.5] | 14.7 [11.0, 18.0] | 22.6 [19.5, 25.5] |

- Datasets with replicates inside the budget: glass_Tg 29.0/30, glass_Tliq 28.5/30, polymer_Tg 22.6/30 distinct compositions in the headline-m set. On glass_E, glass_HV, steel_yield k records are k distinct compositions, so k is the effective size there.

The paired comparison below recalibrates on k records drawn from k DISTINCT compositions (the first record of each of the first k compositions of the same pool, same splits, same target-test half). It is a supplementary diagnostic; the headline numbers everywhere else follow the protocol definition (first k records).

### Recalibrated coverage on target-test: first m records -> m distinct compositions

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.930 -> 0.930 | 0.921 -> 0.921 | 0.933 -> 0.933 | 0.898 -> 0.899 | 0.952 -> 0.952 | 0.902 -> 0.914 |
| ExtraTrees (tree ensemble) | 0.921 -> 0.921 | 0.906 -> 0.906 | 0.929 -> 0.929 | 0.882 -> 0.882 | 0.938 -> 0.938 | 0.879 -> 0.911 |
| HistGB (tree ensemble (boosting)) | 0.921 -> 0.921 | 0.891 -> 0.891 | 0.937 -> 0.937 | 0.890 -> 0.893 | 0.938 -> 0.938 | 0.899 -> 0.911 |
| GP (kernel (Bayesian)) | 0.909 -> 0.909 | 0.903 -> 0.903 | 0.943 -> 0.943 | 0.881 -> 0.885 | 0.969 -> 0.969 | 0.887 -> 0.904 |
| SVR (kernel) | 0.925 -> 0.926 | 0.889 -> 0.889 | 0.919 -> 0.919 | 0.892 -> 0.899 | 0.979 -> 0.979 | 0.875 -> 0.904 |
| kNN (instance-based) | 0.922 -> 0.922 | 0.885 -> 0.885 | 0.916 -> 0.916 | 0.888 -> 0.888 | 0.976 -> 0.976 | 0.881 -> 0.922 |
| Ridge (linear) | 0.933 -> 0.934 | 0.924 -> 0.924 | 0.936 -> 0.936 | 0.897 -> 0.897 | 0.919 -> 0.919 | 0.892 -> 0.904 |
| MLP (neural) | 0.917 -> 0.922 | 0.936 -> 0.936 | 0.941 -> 0.941 | 0.895 -> 0.895 | 0.952 -> 0.952 | 0.880 -> 0.899 |

### k = 9: coverage with the first k records -> with k distinct compositions

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.894 -> 0.903 | 0.920 -> 0.920 | 0.921 -> 0.921 | 0.888 -> 0.898 | 0.926 -> 0.926 | 0.850 -> 0.876 |
| ExtraTrees (tree ensemble) | 0.884 -> 0.891 | 0.924 -> 0.924 | 0.906 -> 0.906 | 0.891 -> 0.898 | 0.919 -> 0.919 | 0.867 -> 0.871 |
| HistGB (tree ensemble (boosting)) | 0.874 -> 0.893 | 0.885 -> 0.885 | 0.894 -> 0.894 | 0.901 -> 0.906 | 0.919 -> 0.919 | 0.879 -> 0.895 |
| GP (kernel (Bayesian)) | 0.888 -> 0.888 | 0.918 -> 0.918 | 0.909 -> 0.909 | 0.886 -> 0.891 | 0.931 -> 0.931 | 0.837 -> 0.835 |
| SVR (kernel) | 0.895 -> 0.909 | 0.908 -> 0.908 | 0.881 -> 0.881 | 0.890 -> 0.890 | 0.936 -> 0.936 | 0.857 -> 0.856 |
| kNN (instance-based) | 0.903 -> 0.903 | 0.897 -> 0.897 | 0.904 -> 0.904 | 0.915 -> 0.918 | 0.931 -> 0.931 | 0.845 -> 0.889 |
| Ridge (linear) | 0.887 -> 0.902 | 0.918 -> 0.918 | 0.893 -> 0.893 | 0.908 -> 0.911 | 0.905 -> 0.905 | 0.872 -> 0.877 |
| MLP (neural) | 0.891 -> 0.898 | 0.944 -> 0.944 | 0.916 -> 0.916 | 0.898 -> 0.898 | 0.945 -> 0.945 | 0.861 -> 0.871 |

### k = 10: coverage with the first k records -> with k distinct compositions

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.897 -> 0.912 | 0.930 -> 0.930 | 0.926 -> 0.926 | 0.914 -> 0.914 | 0.929 -> 0.929 | 0.865 -> 0.899 |
| ExtraTrees (tree ensemble) | 0.891 -> 0.909 | 0.926 -> 0.926 | 0.909 -> 0.909 | 0.914 -> 0.914 | 0.919 -> 0.919 | 0.876 -> 0.893 |
| HistGB (tree ensemble (boosting)) | 0.887 -> 0.914 | 0.920 -> 0.920 | 0.896 -> 0.896 | 0.913 -> 0.913 | 0.919 -> 0.919 | 0.888 -> 0.906 |
| GP (kernel (Bayesian)) | 0.891 -> 0.897 | 0.927 -> 0.927 | 0.909 -> 0.909 | 0.911 -> 0.911 | 0.936 -> 0.936 | 0.837 -> 0.883 |
| SVR (kernel) | 0.899 -> 0.921 | 0.909 -> 0.909 | 0.893 -> 0.893 | 0.905 -> 0.905 | 0.943 -> 0.943 | 0.860 -> 0.889 |
| kNN (instance-based) | 0.904 -> 0.911 | 0.898 -> 0.898 | 0.904 -> 0.904 | 0.925 -> 0.925 | 0.938 -> 0.938 | 0.860 -> 0.898 |
| Ridge (linear) | 0.887 -> 0.912 | 0.918 -> 0.918 | 0.900 -> 0.900 | 0.920 -> 0.920 | 0.917 -> 0.917 | 0.878 -> 0.903 |
| MLP (neural) | 0.897 -> 0.909 | 0.953 -> 0.953 | 0.919 -> 0.919 | 0.905 -> 0.905 | 0.945 -> 0.945 | 0.861 -> 0.886 |

## (e) Model-agnostic normalised conformal and triage

Difficulty sigma(x) = mean Euclidean distance of the standardised input to its 10 nearest training inputs (depends on the input only, so it applies to every model); score |r| / (sigma + beta), beta = median sigma on source-cal. Triage ranks candidates by sigma; RMSE reduction on the 50 % most confident vs no abstention, and AURC relative to the no-abstention RMSE (< 1 = ranking helps).

### Normalised conformal, in-distribution coverage

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.90 [0.88, 0.92] | 0.90 [0.87, 0.92] | 0.89 [0.85, 0.94] | 0.90 [0.88, 0.92] | 0.92 [0.79, 1.00] | 0.90 [0.88, 0.92] |
| ExtraTrees (tree ensemble) | 0.90 [0.87, 0.92] | 0.89 [0.87, 0.91] | 0.89 [0.83, 0.94] | 0.90 [0.87, 0.92] | 0.92 [0.81, 0.99] | 0.90 [0.88, 0.92] |
| HistGB (tree ensemble (boosting)) | 0.90 [0.87, 0.92] | 0.89 [0.87, 0.92] | 0.90 [0.83, 0.94] | 0.90 [0.88, 0.92] | 0.89 [0.73, 0.98] | 0.90 [0.88, 0.93] |
| GP (kernel (Bayesian)) | 0.90 [0.88, 0.92] | 0.89 [0.86, 0.92] | 0.89 [0.85, 0.94] | 0.90 [0.87, 0.92] | 0.89 [0.77, 0.97] | 0.90 [0.88, 0.92] |
| SVR (kernel) | 0.90 [0.87, 0.93] | 0.89 [0.86, 0.92] | 0.89 [0.84, 0.95] | 0.90 [0.88, 0.92] | 0.90 [0.81, 0.99] | 0.90 [0.88, 0.92] |
| kNN (instance-based) | 0.90 [0.87, 0.92] | 0.90 [0.87, 0.92] | 0.89 [0.84, 0.94] | 0.90 [0.87, 0.92] | 0.93 [0.82, 1.00] | 0.90 [0.88, 0.92] |
| Ridge (linear) | 0.90 [0.88, 0.92] | 0.90 [0.86, 0.93] | 0.90 [0.84, 0.96] | 0.90 [0.88, 0.92] | 0.92 [0.82, 0.99] | 0.90 [0.87, 0.93] |
| MLP (neural) | 0.90 [0.87, 0.92] | 0.90 [0.86, 0.93] | 0.90 [0.85, 0.95] | 0.90 [0.88, 0.92] | 0.90 [0.81, 0.97] | 0.90 [0.87, 0.92] |

### Normalised conformal, source-calibrated coverage on target-test

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.59 [0.52, 0.68] | 0.45 [0.30, 0.61] | 0.85 [0.70, 0.96] | 0.60 [0.51, 0.67] | 0.75 [0.52, 0.93] | 0.99 [0.99, 1.00] |
| ExtraTrees (tree ensemble) | 0.56 [0.50, 0.63] | 0.42 [0.27, 0.58] | 0.91 [0.81, 0.97] | 0.63 [0.51, 0.69] | 0.71 [0.48, 0.95] | 1.00 [0.99, 1.00] |
| HistGB (tree ensemble (boosting)) | 0.63 [0.54, 0.72] | 0.60 [0.42, 0.71] | 0.89 [0.75, 0.99] | 0.53 [0.37, 0.67] | 0.66 [0.38, 1.00] | 0.99 [0.99, 1.00] |
| GP (kernel (Bayesian)) | 0.55 [0.49, 0.62] | 0.79 [0.70, 0.91] | 0.97 [0.94, 1.00] | 0.63 [0.56, 0.68] | 0.73 [0.50, 0.91] | 0.96 [0.94, 0.98] |
| SVR (kernel) | 0.54 [0.47, 0.59] | 0.38 [0.26, 0.52] | 0.74 [0.49, 0.94] | 0.65 [0.57, 0.74] | 0.78 [0.59, 0.93] | 0.98 [0.97, 0.99] |
| kNN (instance-based) | 0.63 [0.57, 0.71] | 0.77 [0.67, 0.85] | 0.96 [0.88, 1.00] | 0.45 [0.34, 0.57] | 0.81 [0.59, 0.95] | 0.99 [0.99, 1.00] |
| Ridge (linear) | 0.90 [0.84, 0.95] | 0.80 [0.59, 0.92] | 0.97 [0.94, 1.00] | 0.81 [0.75, 0.87] | 0.80 [0.67, 0.95] | 0.99 [0.97, 0.99] |
| MLP (neural) | 0.61 [0.51, 0.67] | 0.51 [0.25, 0.80] | 0.73 [0.24, 1.00] | 0.52 [0.26, 0.65] | 0.81 [0.64, 1.00] | 0.99 [0.98, 1.00] |

### Normalised conformal, recalibrated on the same m target records, target-test coverage

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.93 [0.82, 0.99] | 0.90 [0.76, 0.97] | 0.94 [0.84, 1.00] | 0.91 [0.70, 0.98] | 0.94 [0.76, 1.00] | 0.91 [0.84, 0.97] |
| ExtraTrees (tree ensemble) | 0.93 [0.79, 0.98] | 0.90 [0.79, 0.99] | 0.93 [0.83, 1.00] | 0.90 [0.74, 1.00] | 0.91 [0.67, 1.00] | 0.89 [0.79, 0.97] |
| HistGB (tree ensemble (boosting)) | 0.92 [0.80, 0.98] | 0.89 [0.77, 0.97] | 0.93 [0.84, 1.00] | 0.90 [0.69, 1.00] | 0.94 [0.76, 1.00] | 0.91 [0.82, 0.98] |
| GP (kernel (Bayesian)) | 0.91 [0.82, 0.96] | 0.91 [0.77, 1.00] | 0.95 [0.87, 1.00] | 0.88 [0.74, 0.98] | 0.97 [0.81, 1.00] | 0.90 [0.78, 0.98] |
| SVR (kernel) | 0.91 [0.84, 0.97] | 0.90 [0.71, 1.00] | 0.92 [0.81, 1.00] | 0.89 [0.75, 0.97] | 0.97 [0.81, 1.00] | 0.88 [0.71, 0.98] |
| kNN (instance-based) | 0.92 [0.85, 0.97] | 0.89 [0.80, 0.97] | 0.92 [0.74, 1.00] | 0.90 [0.76, 0.99] | 0.97 [0.83, 1.00] | 0.89 [0.77, 0.99] |
| Ridge (linear) | 0.93 [0.86, 0.98] | 0.90 [0.77, 0.99] | 0.95 [0.86, 1.00] | 0.91 [0.79, 1.00] | 0.94 [0.78, 1.00] | 0.90 [0.78, 0.98] |
| MLP (neural) | 0.90 [0.83, 0.97] | 0.91 [0.77, 0.97] | 0.94 [0.84, 1.00] | 0.89 [0.71, 0.98] | 0.93 [0.76, 1.00] | 0.88 [0.73, 0.97] |

### Normalised conformal normalised width: ID / source->target / recalibrated

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.77 / 1.34 / 2.66 | 0.80 / 2.08 / 4.74 | 1.90 / 3.33 / 5.96 | 0.76 / 1.15 / 3.85 | 1.63 / 2.04 / 2.84 | 1.70 / 4.17 / 2.07 |
| ExtraTrees (tree ensemble) | 0.68 / 1.17 / 2.45 | 0.74 / 1.92 / 4.21 | 1.80 / 3.16 / 5.57 | 0.70 / 1.05 / 3.72 | 1.51 / 1.92 / 2.63 | 1.66 / 4.08 / 1.94 |
| HistGB (tree ensemble (boosting)) | 0.76 / 1.31 / 2.40 | 0.88 / 2.29 / 4.88 | 2.08 / 3.65 / 6.09 | 0.75 / 1.13 / 4.26 | 1.59 / 2.00 / 2.88 | 1.66 / 4.09 / 2.13 |
| GP (kernel (Bayesian)) | 0.80 / 1.38 / 2.72 | 1.24 / 3.22 / 4.51 | 2.19 / 3.85 / 5.32 | 0.82 / 1.23 / 3.43 | 1.39 / 1.77 / 2.96 | 1.61 / 3.97 / 3.41 |
| SVR (kernel) | 0.72 / 1.24 / 2.93 | 0.72 / 1.87 / 4.75 | 2.22 / 3.90 / 7.28 | 0.84 / 1.26 / 3.06 | 1.53 / 1.93 / 2.95 | 1.60 / 3.95 / 2.94 |
| kNN (instance-based) | 0.84 / 1.45 / 2.70 | 0.91 / 2.38 / 3.41 | 2.13 / 3.73 / 5.38 | 0.90 / 1.35 / 3.49 | 1.63 / 2.06 / 3.04 | 1.75 / 4.31 / 2.45 |
| Ridge (linear) | 1.24 / 2.14 / 2.56 | 1.43 / 3.72 / 4.41 | 3.49 / 6.15 / 5.16 | 1.89 / 2.83 / 3.95 | 1.74 / 2.19 / 3.63 | 2.01 / 4.95 / 3.07 |
| MLP (neural) | 0.79 / 1.36 / 3.14 | 0.99 / 2.57 / 5.40 | 2.27 / 3.99 / 6.58 | 0.77 / 1.16 / 3.75 | 1.63 / 2.08 / 3.07 | 1.63 / 4.00 / 2.18 |

### Triage in-distribution: RMSE reduction (%) at 50 % retained

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 35 [26, 43] | 25 [1, 45] | 16 [-1, 38] | 37 [30, 44] | 27 [7, 45] | 4 [-0, 9] |
| ExtraTrees (tree ensemble) | 33 [24, 43] | 29 [6, 45] | 14 [-6, 36] | 35 [27, 44] | 30 [6, 48] | 3 [-2, 8] |
| HistGB (tree ensemble (boosting)) | 30 [23, 38] | 30 [10, 43] | 15 [-2, 36] | 34 [27, 40] | 25 [3, 47] | 4 [-2, 10] |
| GP (kernel (Bayesian)) | 37 [29, 43] | 42 [24, 58] | 14 [-2, 33] | 40 [33, 46] | 35 [12, 55] | 5 [-1, 10] |
| SVR (kernel) | 32 [23, 38] | 5 [-19, 37] | 12 [-5, 29] | 28 [20, 36] | 35 [10, 59] | 5 [-1, 10] |
| kNN (instance-based) | 39 [30, 47] | 23 [-1, 44] | 13 [-4, 30] | 36 [28, 42] | 37 [17, 60] | 6 [1, 12] |
| Ridge (linear) | 25 [18, 35] | 8 [-13, 34] | 9 [-1, 26] | 16 [10, 21] | 32 [15, 53] | 7 [3, 11] |
| MLP (neural) | 31 [24, 40] | 12 [-14, 50] | 15 [2, 33] | 32 [25, 41] | 35 [19, 53] | 5 [-1, 11] |

### Triage on target-test: RMSE reduction (%) at 50 % retained

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 43 [28, 57] | 28 [11, 46] | 9 [-6, 26] | 12 [-0, 29] | 40 [23, 59] | 17 [8, 25] |
| ExtraTrees (tree ensemble) | 41 [27, 54] | 26 [14, 42] | 8 [-8, 26] | 14 [-4, 32] | 32 [19, 50] | 13 [6, 21] |
| HistGB (tree ensemble (boosting)) | 42 [29, 58] | 28 [7, 46] | -1 [-14, 15] | 8 [-6, 22] | 31 [20, 49] | 8 [-0, 16] |
| GP (kernel (Bayesian)) | 41 [29, 55] | 18 [6, 40] | 10 [-8, 36] | 26 [-5, 44] | 48 [26, 66] | 32 [27, 36] |
| SVR (kernel) | 50 [34, 62] | 24 [14, 37] | 0 [-10, 11] | 23 [2, 38] | 51 [29, 67] | 27 [19, 32] |
| kNN (instance-based) | 41 [29, 52] | 23 [7, 40] | 14 [-5, 39] | 22 [8, 35] | 53 [28, 72] | 6 [-1, 10] |
| Ridge (linear) | 44 [30, 58] | 47 [31, 67] | 18 [3, 37] | 20 [-2, 34] | -0 [-26, 22] | 10 [2, 15] |
| MLP (neural) | 57 [44, 68] | 37 [-8, 60] | 10 [-11, 40] | 18 [2, 31] | 14 [-11, 54] | 15 [7, 34] |

### Relative AURC: in-distribution / target-test

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 0.71 / 0.68 | 0.76 / 0.78 | 0.85 / 0.97 | 0.66 / 0.84 | 0.71 / 0.74 | 0.96 / 0.86 |
| ExtraTrees (tree ensemble) | 0.73 / 0.70 | 0.73 / 0.79 | 0.87 / 1.00 | 0.67 / 0.82 | 0.71 / 0.78 | 0.98 / 0.89 |
| HistGB (tree ensemble (boosting)) | 0.75 / 0.70 | 0.73 / 0.77 | 0.87 / 1.05 | 0.69 / 0.87 | 0.76 / 0.77 | 0.96 / 0.94 |
| GP (kernel (Bayesian)) | 0.69 / 0.71 | 0.63 / 0.85 | 0.88 / 1.00 | 0.63 / 0.75 | 0.67 / 0.70 | 0.95 / 0.73 |
| SVR (kernel) | 0.73 / 0.66 | 0.88 / 0.80 | 0.89 / 1.00 | 0.72 / 0.78 | 0.66 / 0.68 | 0.95 / 0.77 |
| kNN (instance-based) | 0.68 / 0.72 | 0.77 / 0.76 | 0.88 / 0.97 | 0.66 / 0.79 | 0.66 / 0.63 | 0.95 / 0.96 |
| Ridge (linear) | 0.77 / 0.69 | 0.88 / 0.65 | 0.90 / 0.90 | 0.87 / 0.80 | 0.71 / 1.03 | 0.94 / 0.91 |
| MLP (neural) | 0.75 / 0.61 | 0.85 / 0.70 | 0.86 / 0.94 | 0.71 / 0.79 | 0.68 / 0.92 | 0.95 / 0.88 |

### Reference: native-uncertainty triage (RF/ExtraTrees tree spread, GP sd), RMSE reduction (%) at 50 %: ID / target-test

| model (family) | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| RF (tree ensemble) | 50 / 19 | 65 / 24 | 49 / 1 | 49 / -18 | 20 / 10 | 24 / 9 |
| ExtraTrees (tree ensemble) | 47 / 26 | 65 / 24 | 46 / 15 | 48 / -18 | 16 / -6 | 22 / 10 |
| HistGB (tree ensemble (boosting)) | - | - | - | - | - | - |
| GP (kernel (Bayesian)) | 34 / 40 | 43 / 18 | 10 / 10 | 42 / 22 | 37 / 48 | 9 / 31 |
| SVR (kernel) | - | - | - | - | - | - |
| kNN (instance-based) | - | - | - | - | - | - |
| Ridge (linear) | - | - | - | - | - | - |
| MLP (neural) | - | - | - | - | - | - |

## (f) GP native Gaussian 90 % intervals (not conformal), reference

| dataset | ID coverage | ID norm width | target-test coverage | target norm width | ID CE | target CE |
|---|---|---|---|---|---|---|
| glass_Tg | 0.93 [0.91, 0.94] | 0.87 | 0.66 [0.59, 0.76] | 1.55 | 0.093 | 0.240 |
| glass_E | 0.94 [0.90, 0.96] | 1.57 | 0.86 [0.76, 0.94] | 2.61 | 0.131 | 0.178 |
| glass_HV | 0.92 [0.88, 0.95] | 2.02 | 0.69 [0.56, 0.82] | 2.65 | 0.090 | 0.275 |
| glass_Tliq | 0.91 [0.89, 0.92] | 0.85 | 0.71 [0.64, 0.79] | 1.52 | 0.058 | 0.138 |
| steel_yield | 0.88 [0.74, 0.96] | 1.29 | 0.70 [0.50, 0.86] | 1.29 | 0.063 | 0.266 |
| polymer_Tg | 0.91 [0.89, 0.93] | 1.49 | 0.77 [0.67, 0.86] | 2.69 | 0.047 | 0.179 |

## Family-level summary (mean over the six datasets of the per-cell means)

| model | family | ID cov | src->tgt cov | recal cov | ID norm width | recal norm width | RMSE ratio tgt/src | triage ID % | triage tgt % |
|---|---|---|---|---|---|---|---|---|---|
| RF | tree ensemble | 0.900 | 0.361 | 0.923 | 1.21 | 3.25 | 3.4 | 24 | 25 |
| ExtraTrees | tree ensemble | 0.898 | 0.360 | 0.909 | 1.13 | 2.95 | 3.5 | 24 | 22 |
| HistGB | tree ensemble (boosting) | 0.899 | 0.342 | 0.913 | 1.19 | 3.24 | 3.6 | 23 | 19 |
| GP | kernel (Bayesian) | 0.899 | 0.347 | 0.915 | 1.30 | 3.25 | 3.4 | 29 | 29 |
| SVR | kernel | 0.899 | 0.327 | 0.913 | 1.20 | 3.43 | 3.5 | 20 | 29 |
| kNN | instance-based | 0.905 | 0.414 | 0.911 | 1.36 | 3.08 | 3.0 | 26 | 27 |
| Ridge | linear | 0.904 | 0.476 | 0.917 | 1.70 | 3.29 | 2.5 | 16 | 23 |
| MLP | neural | 0.901 | 0.384 | 0.920 | 1.28 | 3.76 | 3.6 | 22 | 25 |

Mean fit time per split (s): RF 4.8, ExtraTrees 2.5, HistGB 1.9, GP 77.0, SVR 0.9, kNN 0.0, Ridge 0.0, MLP 1.2

## Reporting caveats and diagnostics

- Worst-cluster (k-means, k <= 8) coverage is only defined where the evaluated set has at least 40 records (`common.interval_metrics`). It is therefore absent for: glass_E target-test (33), glass_HV target-test (35), steel_yield source-test (39), steel_yield target-test (21). Any figure built from `R2_models.json` must show these cells as missing rather than dropping them silently.
- Width-stratified coverage (`wsc_width_strat`) is degenerate for the absolute-residual blocks (`id`, `src_to_tgt`, `recal`): those intervals have a single constant width, so the statistic equals the overall coverage by construction. It is informative only for the normalised blocks (`norm_*`), where the width varies with sigma(x).
- The width normaliser is the IQR of the property over the whole source pool (in-distribution) or the whole target pool (target-test). It uses target labels for REPORTING only: no interval, threshold, budget or model depends on it. The same normaliser is used in `exp_R1_core.py`.

### In-distribution over-coverage: extra-seed diagnostic

For every cell whose 20-split in-distribution coverage deviates by more than 0.02 from 0.90, the script re-ran the IN-DISTRIBUTION part only (fit on train, calibrate on source-cal, evaluate on source-test) over additional seeds. No target-pool record is read in that diagnostic, so it cannot leak design-region information; the headline remains the 20 protocol splits.

| cell | 20-split mean | per-split SD | exact expectation | guaranteed band | extra seeds | extended mean | extended SD | SE |
|---|---|---|---|---|---|---|---|---|
| kNN / steel_yield | 0.936 | 0.042 | 0.900 | [0.900, 0.925] | 380 | 0.903 | 0.065 | 0.003 |
| Ridge / steel_yield | 0.928 | 0.048 | 0.900 | [0.900, 0.925] | 380 | 0.895 | 0.069 | 0.004 |

## Verifier issues not adopted

- None. All five verifier issues were implemented: (1) the distinct-composition counts and the paired k-distinct recalibration, (2) the extra-seed in-distribution diagnostic replacing the self-contradictory SE argument, (3) the exception / caveat split in this markdown, (4) the k-sweep percentile ranges (and the corrected figures in the returned summary), (5) the cache fingerprint. The four additional points the verifier raised as 'not errors but worth adding' (absolute width vs the target-pool range, the degenerate width-stratified statistic for constant-width blocks, the missing worst-cluster cells, and the near-tautological quantile-ratio correlation) are also addressed above.
