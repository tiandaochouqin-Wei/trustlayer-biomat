# R3 - comprehensive uncertainty metrics (Reviewer 2, comment 12)

Script `revision/code/exp_R3_uq.py`; results `revision/results/R3_uq.json`. Compute: 145 worker-minutes over 140 (dataset, split) tasks with 6 single-threaded workers (24 min wall clock on the shared machine); per-task results are cached in `revision/results/R3_uq_parts_v2/`, so the tables can be rebuilt in seconds (this rebuild: 0.04 min). `meta.runtime_min` in the JSON is the compute wall clock, not the rebuild time.
Protocol: common.py / PROTOCOL.md (grouped 60/20/20 source split, target pool halved into recalibration pool / target-test, headline m, alpha = 0.10, finite-sample conformal quantile, 20 resampling splits; every value is computed per split and summarised as mean [2.5-97.5 percentile across splits]). The splits resample one finite dataset each, so the percentile interval describes split-to-split variability, not population sampling error.

Methods: SC split conformal (RF, |r|); NSC normalised split conformal (|r|/tree-sd); CQR conformalised HistGB quantile regression; GP native Gaussian intervals; BOOT 10 x 100-tree bootstrap RF ensemble, Gaussian intervals; QR raw HistGB quantiles; RFH RF tree-spread Gaussian heuristic; X+conf = split-conformal wrapper (score max(lo-y, y-hi)) around method X's own interval (QR+conf is identical to CQR). All fitted on train only; calibration on source-cal (ID, SHIFT_SRC) or on the first m records of the target recalibration pool (SHIFT_RECAL).

Columns: cov = coverage at nominal 0.90; nW = mean 90 % width / IQR of the evaluated domain; CE = interval calibration error, mean (max) |empirical - nominal| over levels 0.50-0.95; IS/IQR and WIS/IQR = interval score (alpha 0.1) and weighted interval score (alpha 0.1-0.5 + median) divided by the domain IQR; wClu / wStr = worst k-means-cluster coverage (evaluated sets >= 40 records only) and worst width-quartile coverage, each followed by wClu0 / wStr0, the mean of the EXACT reference distribution min_b Binom(n_b, 0.90)/n_b for the realised bin sizes - a worst-of-k statistic sits below 0.90 even under exact conditional validity, so wClu must be read against wClu0 and not against 0.90 (* constant width: no stratification, wStr equals marginal coverage); AURC = relative area under the risk-coverage curve ranked by the method's own 90 % width (1 = no better than no ranking; lower is better); R50 = % RMSE reduction on the 50 % narrowest intervals; Ab1/Ab2 = abstention rate with the rule 'abstain if the 90 % interval is wider than 1 x / 2 x IQR'. The abstention IQR uses only labels available at decision time and never more than the setting's own budget: train + source-cal for ID and for SHIFT_SRC (in that scenario no target label exists), the m recalibration labels for SHIFT_RECAL. `abstain_*_ref` in the JSON repeats the rule on the oracle scale that normalises nW (source pool / whole target pool).

## In-distribution (ID): calibrate on source-cal, evaluate source-test (7 datasets incl. ESOL)

Median across datasets of the per-dataset split means (range across datasets in the per-dataset tables / JSON).

| method | cov | nW | CE | CEmax | IS/IQR | WIS/IQR | wClu | wClu0 | wStr | wStr0 | AURC | R50 % | Ab1 | Ab2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SC | 0.901 | 1.18 | 0.018 | 0.038 | 1.89 | 0.167 | 0.76 | 0.84 | 0.90* | n/a | n/a | n/a | 0.95 | 0.00 |
| NSC | 0.901 | 1.20 | 0.021 | 0.042 | 1.57 | 0.165 | 0.82 | 0.84 | 0.87 | 0.88 | 0.56 | 49 | 0.59 | 0.07 |
| CQR | 0.900 | 1.17 | 0.018 | 0.042 | 1.53 | 0.162 | 0.81 | 0.84 | 0.86 | 0.88 | 0.63 | 41 | 0.63 | 0.06 |
| GP | 0.908 | 1.29 | 0.063 | 0.127 | 1.82 | 0.204 | 0.85 | 0.84 | 0.88 | 0.88 | 0.71 | 34 | 0.63 | 0.06 |
| BOOT | 0.482 | 0.38 | 0.385 | 0.439 | 2.82 | 0.201 | 0.38 | 0.84 | 0.40 | 0.88 | 0.60 | 43 | 0.04 | 0.00 |
| QR | 0.745 | 0.73 | 0.156 | 0.191 | 1.88 | 0.173 | 0.65 | 0.84 | 0.66 | 0.88 | 0.63 | 41 | 0.23 | 0.03 |
| RFH | 0.924 | 1.02 | 0.093 | 0.137 | 1.67 | 0.167 | 0.88 | 0.84 | 0.88 | 0.88 | 0.56 | 49 | 0.38 | 0.06 |
| GP+conf | 0.896 | 1.23 | 0.020 | 0.035 | 1.82 | 0.192 | 0.81 | 0.84 | 0.85 | 0.88 | 0.71 | 34 | 0.65 | 0.06 |
| BOOT+conf | 0.899 | 1.12 | 0.019 | 0.038 | 1.53 | 0.164 | 0.77 | 0.84 | 0.82 | 0.88 | 0.60 | 43 | 0.64 | 0.01 |
| RFH+conf | 0.901 | 1.14 | 0.022 | 0.038 | 1.52 | 0.164 | 0.84 | 0.84 | 0.83 | 0.88 | 0.56 | 49 | 0.59 | 0.11 |

Per-dataset coverage at 0.90 (mean [2.5-97.5 %] over 20 splits), ID:

| method | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg | esol_logS |
|---|---|---|---|---|---|---|---|
| SC | 0.90 [0.88, 0.92] | 0.90 [0.87, 0.91] | 0.90 [0.83, 0.95] | 0.90 [0.87, 0.92] | 0.90 [0.73, 0.99] | 0.90 [0.88, 0.93] | 0.90 [0.83, 0.95] |
| NSC | 0.90 [0.88, 0.92] | 0.90 [0.87, 0.93] | 0.90 [0.85, 0.95] | 0.90 [0.88, 0.92] | 0.93 [0.85, 1.00] | 0.90 [0.88, 0.92] | 0.90 [0.84, 0.94] |
| CQR | 0.90 [0.87, 0.92] | 0.90 [0.88, 0.92] | 0.90 [0.85, 0.95] | 0.90 [0.88, 0.92] | 0.90 [0.77, 0.99] | 0.90 [0.88, 0.92] | 0.92 [0.87, 0.95] |
| GP | 0.93 [0.91, 0.94] | 0.94 [0.90, 0.96] | 0.92 [0.88, 0.95] | 0.91 [0.89, 0.92] | 0.88 [0.74, 0.96] | 0.91 [0.89, 0.93] | 0.89 [0.85, 0.92] |
| BOOT | 0.52 [0.49, 0.54] | 0.58 [0.55, 0.61] | 0.48 [0.43, 0.54] | 0.48 [0.46, 0.50] | 0.48 [0.37, 0.59] | 0.43 [0.39, 0.48] | 0.42 [0.36, 0.50] |
| QR | 0.78 [0.75, 0.80] | 0.74 [0.71, 0.78] | 0.69 [0.63, 0.76] | 0.79 [0.77, 0.81] | 0.69 [0.56, 0.85] | 0.76 [0.73, 0.79] | 0.67 [0.57, 0.74] |
| RFH | 0.95 [0.94, 0.96] | 0.96 [0.95, 0.96] | 0.92 [0.89, 0.95] | 0.94 [0.93, 0.96] | 0.88 [0.78, 0.95] | 0.85 [0.82, 0.87] | 0.78 [0.74, 0.83] |
| GP+conf | 0.90 [0.87, 0.92] | 0.89 [0.86, 0.92] | 0.90 [0.86, 0.95] | 0.90 [0.87, 0.91] | 0.89 [0.77, 1.00] | 0.90 [0.88, 0.93] | 0.91 [0.87, 0.96] |
| BOOT+conf | 0.90 [0.88, 0.92] | 0.89 [0.86, 0.91] | 0.90 [0.85, 0.94] | 0.90 [0.88, 0.92] | 0.91 [0.74, 0.97] | 0.90 [0.88, 0.92] | 0.90 [0.85, 0.94] |
| RFH+conf | 0.90 [0.88, 0.93] | 0.90 [0.87, 0.93] | 0.90 [0.86, 0.95] | 0.90 [0.88, 0.92] | 0.92 [0.79, 1.00] | 0.90 [0.88, 0.92] | 0.90 [0.87, 0.93] |

Per-dataset normalised interval score IS/IQR (mean over splits; lower is better), ID:

| method | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg | esol_logS |
|---|---|---|---|---|---|---|---|
| SC | 1.30 | 1.89 | 3.09 | 1.29 | 2.23 | 2.13 | 1.61 |
| NSC | 1.02 | 1.28 | 2.39 | 1.01 | 2.18 | 1.97 | 1.57 |
| CQR | 1.21 | 1.53 | 2.60 | 1.24 | 2.33 | 1.96 | 1.49 |
| GP | 1.26 | 2.32 | 3.02 | 1.21 | 1.82 | 2.07 | 1.45 |
| BOOT | 1.64 | 1.88 | 3.85 | 1.77 | 3.59 | 3.60 | 2.82 |
| QR | 1.29 | 1.63 | 2.96 | 1.31 | 2.79 | 2.17 | 1.88 |
| RFH | 1.06 | 1.34 | 2.37 | 1.03 | 2.06 | 2.00 | 1.67 |
| GP+conf | 1.24 | 2.26 | 3.00 | 1.21 | 1.82 | 2.07 | 1.45 |
| BOOT+conf | 1.13 | 1.53 | 2.63 | 1.16 | 2.18 | 2.02 | 1.53 |
| RFH+conf | 1.03 | 1.31 | 2.36 | 1.01 | 2.12 | 1.96 | 1.52 |

Per-dataset relative AURC (width-ranked selective prediction; mean over splits), ID:

| method | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg | esol_logS |
|---|---|---|---|---|---|---|---|
| SC | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| NSC | 0.54 | 0.40 | 0.54 | 0.56 | 0.76 | 0.78 | 0.81 |
| CQR | 0.57 | 0.44 | 0.63 | 0.62 | 0.79 | 0.79 | 0.82 |
| GP | 0.71 | 0.60 | 0.89 | 0.62 | 0.67 | 0.95 | 0.90 |
| BOOT | 0.57 | 0.41 | 0.57 | 0.60 | 0.75 | 0.83 | 0.85 |
| QR | 0.57 | 0.44 | 0.63 | 0.62 | 0.79 | 0.79 | 0.82 |
| RFH | 0.54 | 0.40 | 0.54 | 0.56 | 0.76 | 0.78 | 0.81 |
| GP+conf | 0.71 | 0.60 | 0.89 | 0.62 | 0.67 | 0.95 | 0.90 |
| BOOT+conf | 0.57 | 0.41 | 0.57 | 0.60 | 0.75 | 0.83 | 0.85 |
| RFH+conf | 0.54 | 0.40 | 0.54 | 0.56 | 0.76 | 0.78 | 0.81 |

Conditional coverage vs the exact binomial reference, ID (obs = worst-bin coverage, mean over splits; ref = mean of min_b Binom(n_b, 0.90)/n_b for the realised bin sizes; p05 = its 5th percentile; k/20 = splits with p = P(min_null <= obs) <= 0.05):

| dataset | stat | SC | NSC | CQR |
|---|---|---|---|---|
| glass_Tg | k-means cluster | 0.72 / 0.85 (0.78) 12/20 | 0.81 / 0.85 (0.78) 4/20 | 0.81 / 0.85 (0.78) 5/20 |
| glass_Tg | width quartile | const | 0.88 / 0.88 (0.86) 4/20 | 0.85 / 0.88 (0.86) 11/20 |
| glass_E | k-means cluster | 0.70 / 0.84 (0.74) 11/20 | 0.80 / 0.84 (0.74) 4/20 | 0.76 / 0.84 (0.74) 6/20 |
| glass_E | width quartile | const | 0.87 / 0.88 (0.85) 3/20 | 0.85 / 0.88 (0.85) 9/20 |
| glass_HV | k-means cluster | 0.79 / 0.84 (0.75) 3/20 | 0.84 / 0.84 (0.75) 1/20 | 0.81 / 0.84 (0.75) 4/20 |
| glass_HV | width quartile | const | 0.85 / 0.86 (0.81) 3/20 | 0.86 / 0.86 (0.81) 2/20 |
| glass_Tliq | k-means cluster | 0.75 / 0.85 (0.79) 14/20 | 0.83 / 0.85 (0.79) 4/20 | 0.85 / 0.85 (0.79) 3/20 |
| glass_Tliq | width quartile | const | 0.87 / 0.88 (0.86) 8/20 | 0.86 / 0.88 (0.86) 13/20 |
| steel_yield | k-means cluster | n/a | n/a | n/a |
| steel_yield | width quartile | const | 0.84 / 0.80 (0.67) 2/20 | 0.79 / 0.80 (0.66) 2/20 |
| polymer_Tg | k-means cluster | 0.83 / 0.85 (0.79) 3/20 | 0.85 / 0.85 (0.79) 0/20 | 0.83 / 0.85 (0.79) 2/20 |
| polymer_Tg | width quartile | const | 0.88 / 0.88 (0.85) 2/20 | 0.88 / 0.88 (0.85) 1/20 |
| esol_logS | k-means cluster | 0.76 / 0.81 (0.71) 6/20 | 0.78 / 0.81 (0.71) 2/20 | 0.80 / 0.81 (0.71) 3/20 |
| esol_logS | width quartile | const | 0.81 / 0.86 (0.80) 5/20 | 0.86 / 0.86 (0.80) 3/20 |

## Design shift, source-calibrated (SHIFT_SRC): evaluate target-test (6 datasets)

Median across datasets of the per-dataset split means (range across datasets in the per-dataset tables / JSON).

| method | cov | nW | CE | CEmax | IS/IQR | WIS/IQR | wClu | wClu0 | wStr | wStr0 | AURC | R50 % | Ab1 | Ab2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SC | 0.317 | 0.74 | 0.537 | 0.636 | 8.57 | 0.561 | 0.31 | 0.84 | 0.32* | n/a | n/a | n/a | 0.50 | 0.00 |
| NSC | 0.502 | 1.55 | 0.394 | 0.474 | 6.15 | 0.486 | 0.48 | 0.84 | 0.28 | 0.81 | 0.92 | 9 | 0.94 | 0.42 |
| CQR | 0.581 | 1.29 | 0.308 | 0.409 | 6.58 | 0.506 | 0.42 | 0.84 | 0.31 | 0.81 | 0.93 | 9 | 0.91 | 0.28 |
| GP | 0.703 | 2.08 | 0.210 | 0.292 | 4.80 | 0.467 | 0.47 | 0.84 | 0.47 | 0.81 | 0.74 | 27 | 1.00 | 0.98 |
| BOOT | 0.257 | 0.55 | 0.555 | 0.663 | 9.42 | 0.559 | 0.10 | 0.84 | 0.06 | 0.81 | 0.94 | 7 | 0.25 | 0.02 |
| QR | 0.532 | 1.04 | 0.339 | 0.440 | 7.06 | 0.528 | 0.25 | 0.84 | 0.25 | 0.81 | 0.93 | 9 | 0.76 | 0.12 |
| RFH | 0.635 | 1.58 | 0.252 | 0.335 | 5.37 | 0.485 | 0.43 | 0.84 | 0.41 | 0.81 | 0.92 | 9 | 0.97 | 0.59 |
| GP+conf | 0.683 | 1.90 | 0.273 | 0.358 | 4.77 | 0.487 | 0.46 | 0.84 | 0.41 | 0.81 | 0.74 | 27 | 1.00 | 0.96 |
| BOOT+conf | 0.386 | 1.14 | 0.462 | 0.542 | 7.28 | 0.508 | 0.41 | 0.84 | 0.20 | 0.81 | 0.94 | 7 | 0.88 | 0.09 |
| RFH+conf | 0.606 | 1.64 | 0.283 | 0.361 | 5.53 | 0.483 | 0.49 | 0.84 | 0.39 | 0.81 | 0.92 | 9 | 0.98 | 0.56 |

Per-dataset coverage at 0.90 (mean [2.5-97.5 %] over 20 splits), SHIFT_SRC:

| method | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| SC | 0.40 [0.33, 0.48] | 0.23 [0.12, 0.33] | 0.21 [0.14, 0.29] | 0.41 [0.33, 0.49] | 0.15 [0.02, 0.34] | 0.76 [0.72, 0.82] |
| NSC | 0.51 [0.45, 0.59] | 0.41 [0.29, 0.50] | 0.34 [0.17, 0.65] | 0.66 [0.59, 0.74] | 0.49 [0.21, 0.74] | 0.85 [0.79, 0.89] |
| CQR | 0.54 [0.47, 0.61] | 0.62 [0.33, 0.76] | 0.25 [0.14, 0.36] | 0.62 [0.53, 0.68] | 0.30 [0.14, 0.55] | 0.77 [0.72, 0.82] |
| GP | 0.66 [0.59, 0.76] | 0.86 [0.76, 0.94] | 0.69 [0.56, 0.82] | 0.71 [0.64, 0.79] | 0.70 [0.50, 0.86] | 0.77 [0.67, 0.86] |
| BOOT | 0.31 [0.26, 0.38] | 0.21 [0.09, 0.35] | 0.09 [0.03, 0.16] | 0.31 [0.18, 0.41] | 0.13 [0.02, 0.24] | 0.38 [0.32, 0.44] |
| QR | 0.49 [0.41, 0.54] | 0.60 [0.32, 0.76] | 0.21 [0.13, 0.31] | 0.58 [0.50, 0.62] | 0.19 [0.07, 0.34] | 0.64 [0.57, 0.71] |
| RFH | 0.62 [0.53, 0.70] | 0.65 [0.45, 0.85] | 0.42 [0.20, 0.72] | 0.73 [0.68, 0.79] | 0.35 [0.12, 0.52] | 0.78 [0.73, 0.84] |
| GP+conf | 0.61 [0.53, 0.70] | 0.67 [0.48, 0.82] | 0.63 [0.53, 0.74] | 0.70 [0.64, 0.77] | 0.70 [0.55, 0.86] | 0.76 [0.66, 0.84] |
| BOOT+conf | 0.46 [0.39, 0.50] | 0.30 [0.18, 0.49] | 0.20 [0.11, 0.27] | 0.54 [0.47, 0.60] | 0.31 [0.12, 0.52] | 0.78 [0.74, 0.82] |
| RFH+conf | 0.59 [0.50, 0.66] | 0.62 [0.45, 0.81] | 0.39 [0.20, 0.67] | 0.72 [0.66, 0.78] | 0.42 [0.16, 0.62] | 0.83 [0.77, 0.88] |

Per-dataset normalised interval score IS/IQR (mean over splits; lower is better), SHIFT_SRC:

| method | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| SC | 7.72 | 14.76 | 14.13 | 8.02 | 9.13 | 3.08 |
| NSC | 5.41 | 8.10 | 9.17 | 6.14 | 6.15 | 2.76 |
| CQR | 5.33 | 6.77 | 10.59 | 6.40 | 8.02 | 2.94 |
| GP | 3.85 | 3.23 | 5.31 | 6.00 | 5.05 | 4.56 |
| BOOT | 8.13 | 13.20 | 18.58 | 8.89 | 9.96 | 5.39 |
| QR | 5.83 | 7.26 | 13.45 | 6.86 | 9.51 | 3.59 |
| RFH | 4.26 | 5.12 | 7.81 | 5.62 | 7.09 | 2.95 |
| GP+conf | 4.09 | 3.71 | 5.82 | 6.06 | 4.95 | 4.59 |
| BOOT+conf | 6.41 | 11.03 | 12.88 | 6.91 | 7.66 | 2.91 |
| RFH+conf | 4.50 | 5.32 | 8.23 | 5.74 | 6.67 | 2.78 |

Per-dataset relative AURC (width-ranked selective prediction; mean over splits), SHIFT_SRC:

| method | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| SC | n/a | n/a | n/a | n/a | n/a | n/a |
| NSC | 0.82 | 0.80 | 1.00 | 1.17 | 0.92 | 0.93 |
| CQR | 0.93 | 0.86 | 0.97 | 0.71 | 1.02 | 0.92 |
| GP | 0.72 | 0.85 | 1.00 | 0.76 | 0.71 | 0.72 |
| BOOT | 0.88 | 0.82 | 0.95 | 1.03 | 1.01 | 0.93 |
| QR | 0.93 | 0.86 | 0.97 | 0.71 | 1.02 | 0.92 |
| RFH | 0.82 | 0.80 | 1.00 | 1.17 | 0.92 | 0.93 |
| GP+conf | 0.72 | 0.85 | 1.00 | 0.76 | 0.71 | 0.72 |
| BOOT+conf | 0.88 | 0.82 | 0.95 | 1.03 | 1.01 | 0.93 |
| RFH+conf | 0.82 | 0.80 | 1.00 | 1.17 | 0.92 | 0.93 |

Conditional coverage vs the exact binomial reference, SHIFT_SRC (obs = worst-bin coverage, mean over splits; ref = mean of min_b Binom(n_b, 0.90)/n_b for the realised bin sizes; p05 = its 5th percentile; k/20 = splits with p = P(min_null <= obs) <= 0.05):

| dataset | stat | SC | NSC | CQR |
|---|---|---|---|---|
| glass_Tg | k-means cluster | 0.05 / 0.84 (0.74) 20/20 | 0.08 / 0.84 (0.74) 20/20 | 0.14 / 0.84 (0.74) 20/20 |
| glass_Tg | width quartile | const | 0.37 / 0.84 (0.78) 20/20 | 0.38 / 0.84 (0.78) 20/20 |
| glass_E | k-means cluster | n/a | n/a | n/a |
| glass_E | width quartile | const | 0.15 / 0.79 (0.62) 20/20 | 0.23 / 0.79 (0.62) 19/20 |
| glass_HV | k-means cluster | n/a | n/a | n/a |
| glass_HV | width quartile | const | 0.10 / 0.79 (0.62) 19/20 | 0.08 / 0.79 (0.62) 20/20 |
| glass_Tliq | k-means cluster | 0.31 / 0.86 (0.74) 20/20 | 0.48 / 0.86 (0.74) 20/20 | 0.50 / 0.86 (0.74) 20/20 |
| glass_Tliq | width quartile | const | 0.42 / 0.82 (0.69) 20/20 | 0.47 / 0.82 (0.69) 20/20 |
| steel_yield | k-means cluster | n/a | n/a | n/a |
| steel_yield | width quartile | const | 0.19 / 0.76 (0.59) 20/20 | 0.09 / 0.76 (0.57) 20/20 |
| polymer_Tg | k-means cluster | 0.36 / 0.83 (0.75) 20/20 | 0.51 / 0.83 (0.75) 20/20 | 0.42 / 0.83 (0.75) 20/20 |
| polymer_Tg | width quartile | const | 0.75 / 0.87 (0.83) 20/20 | 0.70 / 0.87 (0.83) 20/20 |

## Design shift, recalibrated on m target records (SHIFT_RECAL): evaluate the same target-test half (6 datasets)

Median across datasets of the per-dataset split means (range across datasets in the per-dataset tables / JSON).

| method | cov | nW | CE | CEmax | IS/IQR | WIS/IQR | wClu | wClu0 | wStr | wStr0 | AURC | R50 % | Ab1 | Ab2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SC | 0.926 | 2.95 | 0.072 | 0.142 | 3.56 | 0.420 | 0.81 | 0.84 | 0.93* | n/a | n/a | n/a | 1.00 | 0.97 |
| NSC | 0.901 | 3.55 | 0.073 | 0.142 | 4.04 | 0.438 | 0.71 | 0.84 | 0.80 | 0.81 | 0.92 | 9 | 0.99 | 0.87 |
| CQR | 0.916 | 3.04 | 0.070 | 0.137 | 3.93 | 0.425 | 0.80 | 0.84 | 0.83 | 0.81 | 0.93 | 9 | 1.00 | 0.90 |
| GP+conf | 0.906 | 2.98 | 0.073 | 0.138 | 3.74 | 0.445 | 0.73 | 0.84 | 0.80 | 0.81 | 0.74 | 27 | 1.00 | 0.96 |
| BOOT+conf | 0.912 | 3.00 | 0.070 | 0.136 | 3.55 | 0.417 | 0.80 | 0.84 | 0.83 | 0.81 | 0.94 | 7 | 1.00 | 0.92 |
| RFH+conf | 0.908 | 3.07 | 0.072 | 0.140 | 3.71 | 0.426 | 0.73 | 0.84 | 0.83 | 0.81 | 0.92 | 9 | 1.00 | 0.91 |

Raw GP/BOOT/QR/RFH cannot use the m target labels; their target-test numbers are those of SHIFT_SRC.

Per-dataset coverage at 0.90 (mean [2.5-97.5 %] over 20 splits), SHIFT_RECAL:

| method | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| SC | 0.93 [0.84, 0.99] | 0.92 [0.75, 1.00] | 0.93 [0.84, 1.00] | 0.90 [0.75, 0.99] | 0.95 [0.83, 1.00] | 0.90 [0.82, 0.98] |
| NSC | 0.90 [0.73, 0.97] | 0.89 [0.70, 0.96] | 0.92 [0.76, 1.00] | 0.90 [0.82, 0.99] | 0.98 [0.88, 1.00] | 0.90 [0.80, 0.94] |
| CQR | 0.92 [0.86, 0.97] | 0.92 [0.76, 1.00] | 0.94 [0.83, 1.00] | 0.89 [0.78, 0.96] | 0.94 [0.78, 1.00] | 0.90 [0.83, 0.97] |
| GP+conf | 0.91 [0.80, 0.96] | 0.90 [0.77, 0.97] | 0.94 [0.84, 1.00] | 0.86 [0.67, 0.95] | 0.97 [0.88, 1.00] | 0.90 [0.78, 0.98] |
| BOOT+conf | 0.91 [0.79, 0.99] | 0.91 [0.74, 1.00] | 0.93 [0.81, 1.00] | 0.90 [0.78, 0.99] | 0.95 [0.83, 1.00] | 0.90 [0.77, 0.98] |
| RFH+conf | 0.91 [0.74, 0.98] | 0.89 [0.71, 0.97] | 0.92 [0.74, 1.00] | 0.90 [0.79, 0.99] | 0.95 [0.85, 1.00] | 0.90 [0.81, 0.96] |

Per-dataset normalised interval score IS/IQR (mean over splits; lower is better), SHIFT_RECAL:

| method | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| SC | 3.12 | 3.94 | 6.35 | 4.44 | 3.17 | 2.78 |
| NSC | 3.37 | 4.56 | 7.03 | 5.68 | 3.51 | 2.75 |
| CQR | 3.02 | 4.97 | 5.48 | 4.52 | 3.35 | 2.68 |
| GP+conf | 3.05 | 3.24 | 5.92 | 4.87 | 3.22 | 4.24 |
| BOOT+conf | 3.07 | 3.90 | 6.25 | 4.49 | 3.20 | 2.71 |
| RFH+conf | 3.14 | 4.23 | 6.47 | 4.88 | 3.18 | 2.71 |

Per-dataset relative AURC (width-ranked selective prediction; mean over splits), SHIFT_RECAL:

| method | glass_Tg | glass_E | glass_HV | glass_Tliq | steel_yield | polymer_Tg |
|---|---|---|---|---|---|---|
| SC | n/a | n/a | n/a | n/a | n/a | n/a |
| NSC | 0.82 | 0.80 | 1.00 | 1.17 | 0.92 | 0.93 |
| CQR | 0.93 | 0.86 | 0.97 | 0.71 | 1.02 | 0.92 |
| GP+conf | 0.72 | 0.85 | 1.00 | 0.76 | 0.71 | 0.72 |
| BOOT+conf | 0.88 | 0.82 | 0.95 | 1.03 | 1.01 | 0.93 |
| RFH+conf | 0.82 | 0.80 | 1.00 | 1.17 | 0.92 | 0.93 |

Conditional coverage vs the exact binomial reference, SHIFT_RECAL (obs = worst-bin coverage, mean over splits; ref = mean of min_b Binom(n_b, 0.90)/n_b for the realised bin sizes; p05 = its 5th percentile; k/20 = splits with p = P(min_null <= obs) <= 0.05):

| dataset | stat | SC | NSC | CQR |
|---|---|---|---|---|
| glass_Tg | k-means cluster | 0.82 / 0.84 (0.74) 2/20 | 0.76 / 0.84 (0.74) 6/20 | 0.80 / 0.84 (0.74) 4/20 |
| glass_Tg | width quartile | const | 0.81 / 0.84 (0.78) 5/20 | 0.84 / 0.84 (0.78) 2/20 |
| glass_E | k-means cluster | n/a | n/a | n/a |
| glass_E | width quartile | const | 0.75 / 0.79 (0.62) 1/20 | 0.80 / 0.79 (0.62) 1/20 |
| glass_HV | k-means cluster | n/a | n/a | n/a |
| glass_HV | width quartile | const | 0.79 / 0.79 (0.62) 2/20 | 0.83 / 0.79 (0.62) 0/20 |
| glass_Tliq | k-means cluster | 0.81 / 0.86 (0.74) 2/20 | 0.71 / 0.86 (0.74) 10/20 | 0.81 / 0.86 (0.74) 3/20 |
| glass_Tliq | width quartile | const | 0.71 / 0.82 (0.69) 5/20 | 0.79 / 0.82 (0.69) 1/20 |
| steel_yield | k-means cluster | n/a | n/a | n/a |
| steel_yield | width quartile | const | 0.91 / 0.76 (0.59) 0/20 | 0.82 / 0.76 (0.57) 2/20 |
| polymer_Tg | k-means cluster | 0.62 / 0.83 (0.75) 14/20 | 0.64 / 0.83 (0.75) 13/20 | 0.68 / 0.83 (0.75) 13/20 |
| polymer_Tg | width quartile | const | 0.82 / 0.87 (0.83) 11/20 | 0.85 / 0.87 (0.83) 7/20 |

## Interpretation

- Calibration in-distribution. The three conformal procedures and the three conformalised wrappers are calibrated (median coverage 0.901 SC, 0.901 NSC, 0.900 CQR, 0.896-0.901 for the wrappers; median calibration error 0.018-0.022 over the seven levels). The raw methods are not: BOOT 0.48 [0.42-0.58 across the seven datasets], QR 0.74 [0.67-0.79], GP 0.91 [0.88-0.94], RFH 0.92 [0.78-0.96]. The bootstrap ensemble's failure is structural: the spread of ten resampled forests measures model variance only and omits the noise term, so it cannot reach predictive coverage. The direction of the tree-spread heuristic's error depends on the dataset - it over-covers on the four glass properties (0.95 on glass_Tg, 0.96 on glass_E) and under-covers on polymer_Tg (0.85), steel_yield (0.88) and ESOL (0.78), so 'the heuristic under-covers' and 'the heuristic over-covers' are both true only of particular datasets.
- Efficiency in-distribution: coverage alone does not rank methods, the interval score does, and constant-width split conformal is the least efficient calibrated method (median IS/IQR 1.89 versus 1.57 NSC, 1.53 CQR, 1.52 RFH+conf, 1.53 BOOT+conf; best RFH+conf). Split conformal is also not the widest method: its median normalised width is 1.18 against 1.29 (GP), 1.17 (CQR) and 1.20 (NSC). On glass_Tg the 90 % SC interval averages 86 K, the 5th widest of the ten methods: narrower than GP 100 K (cov 0.93), CQR 96 K, RFH 96 K (cov 0.95), GP+conf 87 K, and wider than RFH+conf 85 K, QR 78 K (cov 0.78), BOOT+conf 77 K, NSC 74 K, BOOT 29 K (cov 0.52) - coverage is annotated only where it is not within 0.02 of nominal. Crucially three of the five methods narrower than SC - RFH+conf 85 K at 0.90 coverage, BOOT+conf 77 K at 0.90 coverage, NSC 74 K at 0.90 coverage - are calibrated at the nominal 0.90, so SC is not even the widest calibrated interval: the adaptive conformal variants achieve the same guarantee with a smaller average width. The honest statement is that conformal intervals are wider than intervals that under-cover, not that a guarantee costs width relative to other calibrated methods.
- Conditional coverage in-distribution, read against the right reference. A minimum over k noisy bins is downward biased even under exact conditional validity, so every worst-bin number is compared with the exact distribution of min_b Binom(n_b, 0.90)/n_b for the realised bin sizes (the bins depend on X and on the interval widths, never on the evaluated labels, so this is the correct null). Worst k-means-cluster coverage for SC is 0.76 against a reference of 0.84 (medians over the datasets with >= 40 evaluation records): per dataset glass_Tg 0.72 vs 0.85 (12/20 splits p<=0.05); glass_E 0.70 vs 0.84 (11/20 splits p<=0.05); glass_HV 0.79 vs 0.84 (3/20 splits p<=0.05); glass_Tliq 0.75 vs 0.85 (14/20 splits p<=0.05); polymer_Tg 0.83 vs 0.85 (3/20 splits p<=0.05); esol_logS 0.76 vs 0.81 (6/20 splits p<=0.05). SC's worst-cluster coverage is below the 5th percentile of the reference on 3 of 6 datasets (glass_Tg, glass_E, glass_Tliq), i.e. the shortfall there is real, but it is 0.10-0.14 below the REFERENCE, not the 0.15-0.20 below 0.90 that a naive reading would report - about 28 % of the apparent gap is the bias of a worst-of-k statistic. The width-quartile statistics are a different story: NSC covers 0.87 in its worst quartile against a reference of 0.88 and CQR 0.86 against 0.88, i.e. at or barely below the null, so they are NOT evidence of conditional miscoverage and should not be quoted as such. The defensible claim for Reviewer 2 comment 2 is the cluster result for constant-width split conformal: a 90 % marginal guarantee does not license a 90 % statement about a particular region of composition space, and in-distribution the adaptive methods (NSC 0.82, CQR 0.81, RFH+conf 0.84) sit closer to the reference 0.84 than SC's 0.76 does. That ordering reverses in the design region: after recalibration the adaptive methods are the ones that fail locally (NSC 0.71, RFH+conf 0.73, GP+conf 0.73 against a reference of 0.84, versus SC 0.81), because their width model is still the source-domain one - adaptivity does not buy local validity where the data are new.
- Design shift, source calibration: every method under-covers on the target-test half; median coverage SC 0.32, NSC 0.50, CQR 0.58, GP 0.70, BOOT 0.26, QR 0.53, RFH 0.64. The distance-aware GP degrades least (0.66-0.86 across datasets) and the conformal procedures are not protected at all - the guarantee is only an exchangeability guarantee. Worse, the miscalibration is invisible to a width-based abstention rule, and for split conformal this is exact rather than approximate: SC's interval has constant width 2q, the same calibration set gives the same q, so the 90 % interval offered for a target candidate is literally the same width as the one offered in-distribution and a width trigger returns bit-identical decisions in the two regimes (verified: the SC abstention rate is identical in ID and SHIFT_SRC on all six datasets), while true coverage falls from 0.90 to 0.15-0.41 on five of the six datasets (polymer_Tg retains 0.76, the mildest of the six shifts). Concretely, on glass_Tg, glass_E, glass_Tliq the source-calibrated conformal interval is narrower than 1 x the source-domain IQR for 100 % of target candidates (abstention 0.00) while its true coverage is only 0.23-0.41. A width trigger therefore carries no information about this shift; detecting it needs a distance-aware score (the GP spread does widen on target inputs) or an explicit shift diagnostic, which is an argument for the trust layer but contradicts any reading in which 'the interval widens when the model is out of its depth'.
- Design shift, recalibration on m target records: all six conformal/conformalised methods return to nominal or slightly above (median coverage SC 0.93, NSC 0.90, CQR 0.92, GP+conf 0.91, BOOT+conf 0.91, RFH+conf 0.91), so the conformal wrapper makes any UQ method - including a badly miscalibrated bootstrap ensemble - recoverable with the same handful of target labels; this is the clearest evidence in this study for the model-agnostic claim. Three honest qualifications: (i) the excess is the finite-sample conservatism of the split-conformal quantile, not a bonus - at m = 13 (steel_yield), 22 (glass_E), 23 (glass_HV), 30 (glass_Tg), 30 (glass_Tliq), 30 (polymer_Tg) the guaranteed lower bound ceil((m+1)0.9)/(m+1) is 0.903-0.929 (steel_yield m = 13 forces >= 0.929), which is why SC averages 0.926 and why the recalibrated intervals are wider than they would be with a larger budget; the residual calibration error over the seven levels is correspondingly 0.072 mean / 0.142 max, against 0.018 / 0.038 in-distribution. (ii) The efficiency advantage of adaptive intervals disappears under shift (median IS/IQR SC 3.56, NSC 4.04, CQR 3.93, GP+conf 3.74, BOOT+conf 3.55, RFH+conf 3.71 - NSC and CQR are no longer better than plain SC, because a spread learned on source data does not order target errors). (iii) Coverage is restored by intervals of median width 3.0 x the target-domain IQR (2.1-5.1), so under the rule 'abstain if the interval is wider than 2 x IQR' 0.97 of target candidates (SC; 0.87-0.97 across the six methods) are abstained. The recalibrated intervals are valid but, on five of the six datasets, too wide to decide much: honest uncertainty, not free accuracy.
- Selective prediction: ranking by interval width is invariant to any additive or multiplicative conformal correction, so NSC, RFH and RFH+conf share one ranking (the RF tree spread), GP/GP+conf, BOOT/BOOT+conf and QR/CQR likewise, and their risk-coverage curves are identical by construction. Conformal calibration sets the level of the interval, not the order of the candidates - 'conformal-ranked triage' is heuristic-spread-ranked triage with a calibrated width. In-distribution the ranking works (median relative AURC 0.56 for tree spread, 49 % RMSE reduction on the most-confident half, 19-65 % across datasets). Under the design shift it largely fails: after recalibration the same ranking gives a median relative AURC of 0.92 and only 9 % error reduction (-18 % to +24 % across datasets, negative on glass_Tliq). The GP spread is the only uncertainty that still orders target errors (AURC 0.74, 27 % median reduction), which argues for a distance-aware score under shift rather than tree spread.
- Abstention: the threshold matters more than the method, and a median across datasets is misleading for constant-width SC, whose ID abstention is all-or-nothing by construction (a 1 x IQR rule rejects every candidate or none): glass_Tg, glass_E, glass_Tliq give 0.00 while glass_HV, steel_yield, polymer_Tg, esol_logS give 0.95-1.00. The adaptive methods abstain selectively and are better summarised by a median: 0.59 (NSC), 0.63 (CQR) at 1 x IQR - a 90 % interval is inevitably of the order of the property's own IQR unless the model is very accurate. At 2 x IQR, ID abstention is 0.00-0.11 but rises to 0.87-0.97 after recalibration under shift. Under SHIFT_RECAL the threshold is the IQR of exactly the m recalibration labels, so this number costs no data beyond the recalibration budget itself.

## Notes and deviations

- Nothing is selected with target-test labels: models are fitted on train, conformal quantiles on source-cal or on the first m recalibration-pool records, abstention thresholds on train+cal labels (ID, SHIFT_SRC) or on exactly the m recalibration labels (SHIFT_RECAL). The IQR used to normalise widths / interval scores (source pool for ID, whole target pool for SHIFT, as in R1_core) is a reporting normaliser only; `abstain_*_ref` in the JSON repeats the abstention rule on that oracle scale, and the qualitative conclusions are the same under either scale.
- FIX ROUND vs the previous version of this script. (1) The SHIFT_SRC abstention threshold previously used the IQR of the target recalibration pool, which does not exist in the SHIFT_SRC scenario (no target labels); it is now the source-domain IQR (train + source-cal). (2) The SHIFT_RECAL threshold previously used the IQR of the whole recalibration pool (|rpool| = glass_Tg 129, glass_E 34, glass_HV 36, glass_Tliq 66, steel_yield 20, polymer_Tg 482) although the narrative is about m = glass_Tg 30, glass_E 22, glass_HV 23, glass_Tliq 30, steel_yield 13, polymer_Tg 30 labels; it is now the IQR of exactly those m labels, so no number in the abstention narrative uses more target data than the recalibration budget. (3) Every worst-bin conditional-coverage statistic now carries an exact binomial reference. (4) meta.runtime_min is the compute wall clock, not the cache-rebuild time. Everything else is unchanged and was verified to reproduce bit-exactly against the previous cache (54740 values over 140 parts, 0 mismatches, max |diff| 0.000e+00).
- Conditional coverage reference. A worst-of-k-bins coverage is downward biased even when conditional coverage is exactly 0.90: for every evaluated set we compute the EXACT distribution of min_b Binom(n_b, 0.90)/n_b over the realised bin sizes (no simulation), and report its mean (wClu0 / wStr0), its 5th percentile and the per-split p-value P(min_null <= observed). The k-means bins depend only on the evaluated inputs X and the width bins only on the interval widths, never on the evaluated labels, so the binomial reference is the appropriate null; it does assume independence across records within a bin. Comparing a worst-bin number with 0.90 rather than with this reference overstates conditional miscoverage, which is why the width-quartile results are reported as null findings here.
- ESOL/Delaney is in-distribution only; grouped random 60/20/20 per seed (identical descriptor vectors share a group); split indices in results/splits/esol_logS_splits.json.
- BOOT (10 x 100-tree RFs, min_samples_leaf 2, sd with ddof 1) and the HistGB quantile models (max_iter 400, learning rate 0.05, no early stopping, i.e. make_model('HistGB') with loss='quantile') are defined in this script with a-priori hyperparameters (not in make_model); nothing was tuned. Note for the rebuttal that this BOOT is not the same estimator as the 'bootstrap ensemble' of demo/figS2_uq_baselines.py, which is 8 HistGradientBoosting models: both under-cover for the same structural reason, but the two coverage numbers are not a like-for-like comparison.
- Calibration error uses all seven levels 0.50-0.95 for every method (CQR/QR: 14 quantile models + median per split). Crossed raw quantiles are sorted; empty conformalised intervals (negative correction larger than half the raw width) are counted (n_empty90_total) and scored as zero-width.
- Finite-sample conformal quantiles are +inf when the calibration set is too small for a level (level 0.95 with m < 19, i.e. steel_yield m = 13 in SHIFT_RECAL): those intervals are kept as infinite (coverage 1, counted in splits_with_infinite_level) and enter the calibration error; the 0.90 interval and all WIS levels (0.5-0.9) are finite for every m >= 9. The same finite-m effect makes the recalibrated 0.90 intervals conservative: the guarantee is ceil((m+1)0.9)/(m+1), i.e. 0.929 at m = 13 and 0.903 at m = 30, which accounts for part of the recalibrated over-coverage and of the recalibrated width.
- Worst-cluster coverage is computed only when the evaluated set has >= 40 records (the rule in common.interval_metrics): n/a for steel_yield source-test (39) and for the target-test halves of glass_E (33), glass_HV (35) and steel_yield (21). Width-quartile and risk-coverage values on target-test halves of 21-35 records are noisy (5-9 records per quartile) - the binomial reference makes that noise explicit (wStr0 falls well below 0.90 for those sets).
- Risk-coverage ranks by the method's own 90 % width; errors are those of the method's own point predictor (RF mean, GP mean, bootstrap mean, QR median). SC has constant width, so its ranking is undefined (n/a); width-stratified coverage for SC equals marginal coverage and has no reference.
- Raw GP/BOOT/QR/RFH are not repeated under SHIFT_RECAL because they cannot use target labels; their conformalised versions are (QR+conf is identical to CQR).
- The additive conformal wrapper around an over-covering method can return an empty interval (negative correction): RFH+conf does so for 0.07 % (glass_HV), 0.76 % (glass_Tg), 0.83 % (glass_Tliq), 1.26 % (glass_E) of in-distribution test points and never under shift. A multiplicative wrapper (NSC) cannot produce empty intervals; raw quantile crossing (QR) occurs for < 1 % of points and is fixed by sorting (n_cross90_total).
- ESOL grouping is conservative: 1,128 distinct SMILES fall into 919 distinct descriptor vectors, and records sharing a descriptor vector are kept on the same side of the split.
- GP+conf on glass_Tliq averages 0.86 coverage after recalibration, the only recalibrated cell visibly below nominal; the split interval [0.67, 0.95] contains 0.90, and exchangeability between the recalibration pool and the target-test half holds at composition-group, not record, level.
- Cross-check against exp_R1_core.py: SC coverage (ID, SHIFT_SRC, SHIFT_RECAL) and the NSC recalibrated coverage and 50 %-triage numbers reproduce R1_core.json split by split.
- Scope note for the rebuttal: the glass_Tg widths quoted here are measured on the SOURCE pool's held-out test block (P <= q0.70(P), 6,592 of the 7,623 glasses; source-pool y IQR 114 K vs 115 K for all records), not on the full dataset. demo/figS2_uq_baselines.py measured 155.5 K on a 1,000-record ungrouped subsample of all 7,623 records (600 training rows, 6 seeds, no confidence intervals). Same underlying dataset and nearly the same property spread, but the comparison should be stated as 'source-pool held-out test under the unified protocol' rather than 'the full dataset'.

## Verifier issues not adopted

None. All ten verifier issues were adopted: the two substantive ones (the glass_Tg width sentence, now generated from the width ranking in the table rather than asserted; the conditional-coverage claim, now stated against an exact binomial reference and restricted to the cluster result for split conformal) and the eight minor ones (rounding consistency in the abstention sentence, meta.runtime_min, the empty-interval share lower bound now computed from n_empty90_total / evaluated records, the bimodal SC abstention median now reported per dataset, the SHIFT_SRC abstention scale now source-domain, the SHIFT_RECAL abstention scale now exactly the m recalibration labels with |rpool| also disclosed, the rebuttal-precision notes on 'full dataset' and on the two different bootstrap estimators, and the missing discussion of the recalibrated calibration error and finite-m conservatism).
