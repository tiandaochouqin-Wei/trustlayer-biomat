# R4b - genuine computation-to-measurement shift (DFT -> calorimetry)

**Scope.** Inorganic compounds (intermetallics, silicides, borides, carbides, germanides) from Kim et al. (2017, Sci. Data 4, 170162) with matched Materials Project / OQMD DFT formation enthalpies (matminer `expt_formation_enthalpy`). These are **not biomedical materials**; the analysis is a *genuine computation-to-measurement analogue* of the in-silico -> in-vitro gap (a model trained and calibrated on computed labels is asked to predict a measured property), not that gap itself.

**Data.** 1276 records; magpie featurisation failures: 0 (none); 132 composition features. MP: 643 records with both experimental and MP values (607 reduced formulas, 36 replicate/polymorph records); OQMD: 685 records with both experimental and OQMD values (648 reduced formulas, 37 replicate/polymorph records). Units eV/atom. Experimental IQR (normaliser): MP 0.459, OQMD 0.477.

**Protocol.** 20 grouped (reduced-formula) splits: train 50 % / calibration 20 % / target pool 30 %, pool halves = recalibration pool / target-test. alpha = 0.1, finite-sample conformal quantile, |y - yhat| score; headline m = clip(floor(|pool|/3), 10, 30) = MP [30], OQMD [30]. Mean [2.5-97.5 percentile] across the 20 splits, which resample one dataset (split-to-split variability, not population sampling error). Nothing is selected on target-test labels; the bias-shrinkage pseudo-count (lambda = 5) and chemistry classes were fixed a priori.


## In-silico source: MP

**(iv) DFT - experiment discrepancy** (all matched records; grouped bootstrap 95 % CI):

| subset | n records | mean signed (DFT - expt) | MAE | RMSE | frac DFT more negative |
|---|---|---|---|---|---|
| all | 643 | +0.006 [-0.003, +0.015] | 0.075 [0.069, 0.082] | 0.112 | 0.50 |
| f-block (Ln/An) compounds | 246 | -0.006 [-0.019, +0.006] | 0.067 [0.059, 0.075] | 0.091 | 0.54 |
| B/C/Si/Ge compounds | 115 | +0.005 [-0.011, +0.022] | 0.065 [0.055, 0.076] | 0.087 | 0.50 |
| sp-metal intermetallics | 158 | +0.049 [+0.028, +0.075] | 0.090 [0.072, 0.111] | 0.149 | 0.35 |
| transition-metal-only intermetallics | 124 | -0.025 [-0.045, -0.004] | 0.083 [0.069, 0.098] | 0.116 | 0.62 |

Inter-laboratory reproducibility floor: 34 formulas with replicate experimental records, pooled within-formula SD 0.043 eV/atom, mean |pairwise difference| 0.047 eV/atom.


### MP / RF

Sizes (seed 0): {'train': 322, 'cal': 128, 'rpool': 97, 'ttest': 96}; m = [30].

| setting (target-test, 20 splits) | coverage | norm. width (/IQR expt) | interval score (eV/atom) | WIS | cal. error (mean) | worst-cluster cov |
|---|---|---|---|---|---|---|
| (i) in-silico calibrated, vs **DFT** labels (sanity) | 0.907 [0.839, 0.969] | 1.142 [0.878, 1.382] | 0.641 [0.548, 0.750] | 0.064 [0.055, 0.074] | 0.042 [0.019, 0.080] | 0.759 [0.583, 0.887] |
| (i) in-silico calibrated, vs **experimental** labels | 0.897 [0.840, 0.948] | 1.055 [0.811, 1.276] | 0.652 [0.566, 0.750] | 0.068 [0.058, 0.078] | 0.044 [0.020, 0.106] | 0.760 [0.577, 0.867] |
| (ii) recalibrated with m = 30 measurements | 0.901 [0.818, 0.980] | 1.090 [0.790, 1.531] | 0.678 [0.593, 0.787] | 0.070 [0.059, 0.080] | 0.071 [0.036, 0.148] | 0.772 [0.522, 0.966] |
| (iv) + global signed-bias correction (m) | 0.906 [0.823, 0.979] | 1.130 [0.823, 1.534] | 0.689 [0.593, 0.777] | 0.072 [0.059, 0.083] | 0.076 [0.035, 0.144] | 0.773 [0.580, 0.983] |
| (iv) + class-wise bias correction (m) | 0.904 [0.825, 0.985] | 1.152 [0.827, 1.514] | 0.687 [0.605, 0.790] | 0.073 [0.062, 0.084] | 0.078 [0.030, 0.133] | 0.773 [0.603, 0.970] |
| (ii) sigma-normalised recalibration (m) | 0.902 [0.736, 0.985] | 1.019 [0.653, 1.325] | 0.601 [0.487, 0.742] | n/a | n/a | 0.800 [0.476, 0.943] |
| (iii-a) reference: trained + calibrated on experimental labels | 0.910 [0.861, 0.964] | 1.050 [0.870, 1.276] | 0.631 [0.539, 0.731] | 0.065 [0.057, 0.072] | 0.046 [0.022, 0.073] | 0.779 [0.656, 0.901] |
| (iii-b) reference: DFT value itself + m measurements | 0.853 [0.704, 0.976] | 0.644 [0.427, 1.134] | 0.544 [0.386, 0.747] | 0.050 [0.039, 0.062] | 0.077 [0.031, 0.189] | 0.712 [0.399, 0.921] |

(sanity row: width normalised by the IQR of DFT values.)

RMSE on target-test (eV/atom): DFT-trained model vs DFT 0.136 [0.113, 0.157]; vs experiment 0.143 [0.123, 0.163]; experiment-trained model vs experiment 0.136 [0.117, 0.152]; DFT look-up vs experiment 0.107 [0.081, 0.130]. Correlation(model error on DFT, DFT-expt discrepancy) -0.31 [-0.52, -0.13]. Mean signed (prediction - experiment) on test -0.000 [-0.027, 0.026]; bias estimated from m recal records (expt - prediction) 0.001 [-0.043, 0.064].

Paired coverage shortfall (DFT-label coverage - experimental-label coverage, same intervals, same compounds): 0.010 [-0.047, 0.067].

| chemistry class (target-test) | n/split | in-silico cov vs DFT | in-silico cov vs expt | expt-trained ref cov |
|---|---|---|---|---|
| f-block (Ln/An) compounds | 37 | 0.976 [0.888, 1.000] | 0.962 [0.913, 1.000] | 0.978 [0.931, 1.000] |
| B/C/Si/Ge compounds | 17 | 0.891 [0.761, 0.978] | 0.840 [0.605, 1.000] | 0.881 [0.624, 0.976] |
| sp-metal intermetallics | 23 | 0.839 [0.721, 0.951] | 0.847 [0.680, 0.979] | 0.869 [0.695, 0.958] |
| transition-metal-only intermetallics | 19 | 0.875 [0.714, 1.000] | 0.874 [0.759, 0.976] | 0.858 [0.720, 1.000] |

Coverage of experimental labels stratified (evaluation only) by the compound's own |DFT - experiment| gap:

| stratum (eV/atom) | n/split | in-silico calibrated | recalibrated (m) | expt-trained ref |
|---|---|---|---|---|
| |d|<0.05 | 45.6 | 0.915 [0.821, 0.979] | 0.921 [0.810, 1.000] | 0.913 [0.814, 0.980] |
| 0.05<=|d|<0.15 | 38.8 | 0.923 [0.823, 1.000] | 0.918 [0.790, 1.000] | 0.923 [0.880, 0.975] |
| |d|>=0.15 | 12.5 | 0.735 [0.387, 0.957] | 0.770 [0.395, 1.000] | 0.850 [0.527, 1.000] |

Abstention (share of target candidates whose recalibrated interval is wider than the in-silico interval): constant-width 0.500, sigma-normalised 0.428. Triage (RMSE reduction on the 50 % most confident by tree spread): vs DFT 40.1 [21.5, 57.6] %, vs experiment 28.2 [10.9, 45.7] %; relative AURC vs experiment 0.736 [0.665, 0.838].


k-sweep (coverage of experimental labels / normalised width; k = 0 is the in-silico calibration):

| k | plain recal cov | plain nw | +global bias cov | +global bias nw | d nw (global - plain) | +class bias cov | +class bias nw | DFT look-up cov | DFT look-up nw | Beta theory 5/50/95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.897 | 1.055 | - | - | - | - | - | - | - | - |
| 5 | 1.000 | inf | 1.000 | inf | - | 1.000 | inf | 1.000 | inf | - |
| 9 | 0.923 | 1.301 | 0.916 | 1.369 | 0.068 [-0.366, 0.478] | 0.911 | 1.371 | 0.851 | 0.813 | 0.72, 0.93, 0.99 |
| 10 | 0.923 | 1.302 | 0.911 | 1.372 | 0.070 [-0.405, 0.437] | 0.908 | 1.369 | 0.859 | 0.829 | 0.74, 0.93, 0.99 |
| 20 | 0.904 | 1.140 | 0.898 | 1.177 | 0.037 [-0.178, 0.249] | 0.900 | 1.214 | 0.866 | 0.741 | 0.78, 0.92, 0.98 |
| 30 | 0.901 | 1.090 | 0.906 | 1.130 | 0.040 [-0.095, 0.212] | 0.904 | 1.152 | 0.853 | 0.644 | 0.80, 0.91, 0.97 |
| 50 | 0.895 | 1.071 | 0.897 | 1.087 | 0.016 [-0.110, 0.179] | 0.901 | 1.111 | 0.884 | 0.727 | 0.83, 0.91, 0.96 |
| 100 | n/a (pool < k) |||||||||| 
| all (=96) | 0.912 | 1.132 | 0.913 | 1.143 | 0.011 [-0.105, 0.092] | 0.905 | 1.114 | 0.900 | 0.750 | 0.85, 0.91, 0.95 |

k < 9: the finite-sample conformal quantile is infinite at alpha = 0.1 (coverage 1 trivially, infinite interval). 'n/a (pool < k)': the recalibration pool of every split is smaller than k.


Semi-synthetic surrogate-accuracy sweep (sensitivity analysis; real DFT-experiment discrepancy, surrogate error vs DFT rescaled by lambda; prediction = DFT + lambda x (fitted-model error); lambda = 1 is the fitted model, lambda = 0 a perfect DFT surrogate):

| lambda | surrogate RMSE vs DFT | in-silico cov vs DFT | in-silico cov vs expt | in-silico nw | recal (m) cov vs expt | recal (m) nw |
|---|---|---|---|---|---|---|
| 1.0 | 0.136 | 0.907 | 0.897 [0.840, 0.948] | 1.055 | 0.901 [0.818, 0.980] | 1.090 |
| 0.75 | 0.102 | 0.907 | 0.860 [0.804, 0.923] | 0.791 | 0.889 [0.771, 0.990] | 0.906 |
| 0.5 | 0.068 | 0.907 | 0.772 [0.701, 0.857] | 0.528 | 0.887 [0.755, 0.985] | 0.767 |
| 0.35 | 0.048 | 0.907 | 0.664 [0.522, 0.779] | 0.369 | 0.893 [0.766, 0.980] | 0.727 |
| 0.25 | 0.034 | 0.907 | 0.532 [0.396, 0.671] | 0.264 | 0.890 [0.776, 0.985] | 0.703 |
| 0.15 | 0.020 | 0.907 | 0.343 [0.210, 0.413] | 0.158 | 0.877 [0.751, 0.976] | 0.671 |
| 0.1 | 0.014 | 0.907 | 0.249 [0.170, 0.347] | 0.106 | 0.867 [0.730, 0.966] | 0.659 |
| 0.0 | 0.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000 | 0.853 [0.704, 0.976] | 0.644 |

(lambda = 0: zero-width in-silico interval, so DFT-label coverage is trivially 1 and experimental-label coverage is the share of exact DFT = experiment ties.)


### MP / HistGB

Sizes (seed 0): {'train': 322, 'cal': 128, 'rpool': 97, 'ttest': 96}; m = [30].

| setting (target-test, 20 splits) | coverage | norm. width (/IQR expt) | interval score (eV/atom) | WIS | cal. error (mean) | worst-cluster cov |
|---|---|---|---|---|---|---|
| (i) in-silico calibrated, vs **DFT** labels (sanity) | 0.907 [0.845, 0.964] | 0.965 [0.721, 1.168] | 0.565 [0.468, 0.698] | 0.054 [0.046, 0.063] | 0.041 [0.020, 0.082] | 0.774 [0.579, 0.913] |
| (i) in-silico calibrated, vs **experimental** labels | 0.890 [0.804, 0.938] | 0.891 [0.666, 1.079] | 0.594 [0.485, 0.711] | 0.062 [0.051, 0.072] | 0.066 [0.028, 0.139] | 0.762 [0.612, 0.875] |
| (ii) recalibrated with m = 30 measurements | 0.906 [0.788, 0.979] | 0.985 [0.705, 1.255] | 0.607 [0.522, 0.695] | 0.062 [0.053, 0.074] | 0.076 [0.030, 0.137] | 0.779 [0.497, 0.926] |
| (iv) + global signed-bias correction (m) | 0.914 [0.824, 0.969] | 1.010 [0.734, 1.261] | 0.611 [0.524, 0.691] | 0.064 [0.054, 0.074] | 0.076 [0.031, 0.140] | 0.783 [0.579, 0.909] |
| (iv) + class-wise bias correction (m) | 0.921 [0.829, 0.974] | 1.086 [0.743, 1.378] | 0.619 [0.517, 0.716] | 0.064 [0.055, 0.075] | 0.073 [0.036, 0.131] | 0.800 [0.559, 0.942] |
| (ii) sigma-normalised recalibration (m) | 0.908 [0.804, 0.979] | 1.102 [0.734, 2.079] | 0.651 [0.514, 1.094] | n/a | n/a | 0.778 [0.559, 0.920] |
| (iii-a) reference: trained + calibrated on experimental labels | 0.909 [0.828, 0.979] | 0.880 [0.717, 1.097] | 0.539 [0.415, 0.642] | 0.055 [0.046, 0.061] | 0.045 [0.017, 0.085] | 0.788 [0.645, 0.939] |
| (iii-b) reference: DFT value itself + m measurements | 0.853 [0.704, 0.976] | 0.644 [0.427, 1.134] | 0.544 [0.386, 0.747] | 0.050 [0.039, 0.062] | 0.077 [0.031, 0.189] | 0.712 [0.399, 0.921] |

(sanity row: width normalised by the IQR of DFT values.)

RMSE on target-test (eV/atom): DFT-trained model vs DFT 0.115 [0.097, 0.140]; vs experiment 0.129 [0.110, 0.151]; experiment-trained model vs experiment 0.116 [0.094, 0.130]; DFT look-up vs experiment 0.107 [0.081, 0.130]. Correlation(model error on DFT, DFT-expt discrepancy) -0.31 [-0.58, -0.12]. Mean signed (prediction - experiment) on test 0.001 [-0.023, 0.024]; bias estimated from m recal records (expt - prediction) 0.004 [-0.026, 0.060].

Paired coverage shortfall (DFT-label coverage - experimental-label coverage, same intervals, same compounds): 0.018 [-0.037, 0.082].

| chemistry class (target-test) | n/split | in-silico cov vs DFT | in-silico cov vs expt | expt-trained ref cov |
|---|---|---|---|---|
| f-block (Ln/An) compounds | 37 | 0.970 [0.896, 1.000] | 0.951 [0.899, 1.000] | 0.974 [0.879, 1.000] |
| B/C/Si/Ge compounds | 17 | 0.914 [0.857, 1.000] | 0.860 [0.594, 0.972] | 0.877 [0.737, 0.978] |
| sp-metal intermetallics | 23 | 0.820 [0.687, 0.927] | 0.833 [0.643, 0.983] | 0.855 [0.627, 0.956] |
| transition-metal-only intermetallics | 19 | 0.887 [0.769, 1.000] | 0.867 [0.737, 0.982] | 0.883 [0.703, 1.000] |

Coverage of experimental labels stratified (evaluation only) by the compound's own |DFT - experiment| gap:

| stratum (eV/atom) | n/split | in-silico calibrated | recalibrated (m) | expt-trained ref |
|---|---|---|---|---|
| |d|<0.05 | 45.6 | 0.930 [0.854, 0.979] | 0.948 [0.853, 1.000] | 0.921 [0.799, 0.989] |
| 0.05<=|d|<0.15 | 38.8 | 0.909 [0.816, 0.976] | 0.918 [0.759, 1.000] | 0.921 [0.827, 0.988] |
| |d|>=0.15 | 12.5 | 0.686 [0.412, 0.957] | 0.711 [0.348, 1.000] | 0.823 [0.591, 1.000] |

Abstention (share of target candidates whose recalibrated interval is wider than the in-silico interval): constant-width 0.600, sigma-normalised 0.578. Triage (RMSE reduction on the 50 % most confident by kNN distance): vs DFT 20.1 [-1.9, 38.8] %, vs experiment 15.0 [1.4, 34.9] %; relative AURC vs experiment 0.858 [0.740, 0.940].


k-sweep (coverage of experimental labels / normalised width; k = 0 is the in-silico calibration):

| k | plain recal cov | plain nw | +global bias cov | +global bias nw | d nw (global - plain) | +class bias cov | +class bias nw | DFT look-up cov | DFT look-up nw | Beta theory 5/50/95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.890 | 0.891 | - | - | - | - | - | - | - | - |
| 5 | 1.000 | inf | 1.000 | inf | - | 1.000 | inf | 1.000 | inf | - |
| 9 | 0.923 | 1.208 | 0.912 | 1.275 | 0.067 [-0.303, 0.344] | 0.922 | 1.306 | 0.851 | 0.813 | 0.72, 0.93, 0.99 |
| 10 | 0.930 | 1.214 | 0.915 | 1.275 | 0.061 [-0.331, 0.326] | 0.918 | 1.284 | 0.859 | 0.829 | 0.74, 0.93, 0.99 |
| 20 | 0.898 | 1.015 | 0.895 | 1.006 | -0.009 [-0.164, 0.175] | 0.889 | 1.006 | 0.866 | 0.741 | 0.78, 0.92, 0.98 |
| 30 | 0.906 | 0.985 | 0.914 | 1.010 | 0.025 [-0.130, 0.184] | 0.921 | 1.086 | 0.853 | 0.644 | 0.80, 0.91, 0.97 |
| 50 | 0.904 | 0.970 | 0.899 | 0.988 | 0.017 [-0.079, 0.120] | 0.921 | 1.046 | 0.884 | 0.727 | 0.83, 0.91, 0.96 |
| 100 | n/a (pool < k) |||||||||| 
| all (=96) | 0.920 | 1.021 | 0.917 | 1.011 | -0.010 [-0.076, 0.060] | 0.921 | 1.038 | 0.900 | 0.750 | 0.85, 0.91, 0.95 |

k < 9: the finite-sample conformal quantile is infinite at alpha = 0.1 (coverage 1 trivially, infinite interval). 'n/a (pool < k)': the recalibration pool of every split is smaller than k.


Semi-synthetic surrogate-accuracy sweep (sensitivity analysis; real DFT-experiment discrepancy, surrogate error vs DFT rescaled by lambda; prediction = DFT + lambda x (fitted-model error); lambda = 1 is the fitted model, lambda = 0 a perfect DFT surrogate):

| lambda | surrogate RMSE vs DFT | in-silico cov vs DFT | in-silico cov vs expt | in-silico nw | recal (m) cov vs expt | recal (m) nw |
|---|---|---|---|---|---|---|
| 1.0 | 0.115 | 0.907 | 0.890 [0.804, 0.938] | 0.891 | 0.906 [0.788, 0.979] | 0.985 |
| 0.75 | 0.087 | 0.907 | 0.845 [0.762, 0.918] | 0.668 | 0.905 [0.771, 0.974] | 0.860 |
| 0.5 | 0.058 | 0.907 | 0.731 [0.623, 0.857] | 0.445 | 0.897 [0.766, 0.979] | 0.747 |
| 0.35 | 0.040 | 0.907 | 0.595 [0.390, 0.774] | 0.312 | 0.885 [0.745, 0.980] | 0.695 |
| 0.25 | 0.029 | 0.907 | 0.466 [0.303, 0.624] | 0.223 | 0.875 [0.736, 0.976] | 0.669 |
| 0.15 | 0.017 | 0.907 | 0.296 [0.175, 0.371] | 0.134 | 0.861 [0.746, 0.966] | 0.649 |
| 0.1 | 0.012 | 0.907 | 0.221 [0.132, 0.273] | 0.089 | 0.857 [0.732, 0.966] | 0.645 |
| 0.0 | 0.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000 | 0.853 [0.704, 0.976] | 0.644 |

(lambda = 0: zero-width in-silico interval, so DFT-label coverage is trivially 1 and experimental-label coverage is the share of exact DFT = experiment ties.)


## In-silico source: OQMD

**(iv) DFT - experiment discrepancy** (all matched records; grouped bootstrap 95 % CI):

| subset | n records | mean signed (DFT - expt) | MAE | RMSE | frac DFT more negative |
|---|---|---|---|---|---|
| all | 685 | +0.015 [+0.006, +0.025] | 0.084 [0.077, 0.091] | 0.123 | 0.50 |
| f-block (Ln/An) compounds | 293 | +0.026 [+0.010, +0.042] | 0.092 [0.081, 0.104] | 0.132 | 0.48 |
| B/C/Si/Ge compounds | 107 | +0.006 [-0.010, +0.023] | 0.066 [0.055, 0.078] | 0.090 | 0.51 |
| sp-metal intermetallics | 158 | +0.033 [+0.012, +0.054] | 0.087 [0.071, 0.104] | 0.140 | 0.42 |
| transition-metal-only intermetallics | 127 | -0.026 [-0.043, -0.009] | 0.076 [0.064, 0.089] | 0.102 | 0.62 |

Inter-laboratory reproducibility floor: 35 formulas with replicate experimental records, pooled within-formula SD 0.053 eV/atom, mean |pairwise difference| 0.052 eV/atom.


### OQMD / RF

Sizes (seed 0): {'train': 342, 'cal': 137, 'rpool': 103, 'ttest': 103}; m = [30].

| setting (target-test, 20 splits) | coverage | norm. width (/IQR expt) | interval score (eV/atom) | WIS | cal. error (mean) | worst-cluster cov |
|---|---|---|---|---|---|---|
| (i) in-silico calibrated, vs **DFT** labels (sanity) | 0.912 [0.844, 0.971] | 1.139 [0.914, 1.366] | 0.652 [0.566, 0.797] | 0.068 [0.059, 0.083] | 0.041 [0.015, 0.088] | 0.783 [0.625, 0.917] |
| (i) in-silico calibrated, vs **experimental** labels | 0.903 [0.849, 0.961] | 1.062 [0.853, 1.274] | 0.658 [0.568, 0.758] | 0.070 [0.059, 0.080] | 0.047 [0.020, 0.076] | 0.782 [0.615, 0.906] |
| (ii) recalibrated with m = 30 measurements | 0.889 [0.761, 0.986] | 1.068 [0.768, 1.578] | 0.686 [0.577, 0.838] | 0.071 [0.062, 0.083] | 0.066 [0.030, 0.155] | 0.750 [0.431, 0.974] |
| (iv) + global signed-bias correction (m) | 0.897 [0.717, 0.991] | 1.103 [0.740, 1.765] | 0.694 [0.586, 0.913] | 0.072 [0.065, 0.082] | 0.066 [0.033, 0.167] | 0.763 [0.333, 0.980] |
| (iv) + class-wise bias correction (m) | 0.900 [0.742, 0.991] | 1.115 [0.724, 1.573] | 0.688 [0.597, 0.842] | 0.072 [0.064, 0.082] | 0.072 [0.024, 0.160] | 0.780 [0.410, 0.980] |
| (ii) sigma-normalised recalibration (m) | 0.889 [0.780, 0.986] | 0.998 [0.794, 1.464] | 0.609 [0.490, 0.730] | n/a | n/a | 0.753 [0.459, 0.974] |
| (iii-a) reference: trained + calibrated on experimental labels | 0.917 [0.820, 0.971] | 1.033 [0.833, 1.311] | 0.618 [0.521, 0.724] | 0.063 [0.056, 0.076] | 0.048 [0.020, 0.112] | 0.799 [0.532, 0.913] |
| (iii-b) reference: DFT value itself + m measurements | 0.911 [0.741, 0.990] | 0.939 [0.443, 1.615] | 0.641 [0.499, 0.887] | 0.057 [0.049, 0.066] | 0.082 [0.039, 0.167] | 0.822 [0.611, 0.974] |

(sanity row: width normalised by the IQR of DFT values.)

RMSE on target-test (eV/atom): DFT-trained model vs DFT 0.143 [0.123, 0.174]; vs experiment 0.147 [0.124, 0.164]; experiment-trained model vs experiment 0.132 [0.116, 0.157]; DFT look-up vs experiment 0.122 [0.097, 0.143]. Correlation(model error on DFT, DFT-expt discrepancy) -0.38 [-0.57, -0.17]. Mean signed (prediction - experiment) on test 0.012 [-0.016, 0.044]; bias estimated from m recal records (expt - prediction) -0.008 [-0.040, 0.029].

Paired coverage shortfall (DFT-label coverage - experimental-label coverage, same intervals, same compounds): 0.008 [-0.034, 0.044].

| chemistry class (target-test) | n/split | in-silico cov vs DFT | in-silico cov vs expt | expt-trained ref cov |
|---|---|---|---|---|
| f-block (Ln/An) compounds | 43 | 0.956 [0.895, 1.000] | 0.958 [0.901, 1.000] | 0.987 [0.957, 1.000] |
| B/C/Si/Ge compounds | 17 | 0.874 [0.693, 1.000] | 0.837 [0.667, 0.957] | 0.845 [0.639, 1.000] |
| sp-metal intermetallics | 24 | 0.860 [0.682, 1.000] | 0.867 [0.695, 0.960] | 0.876 [0.675, 0.984] |
| transition-metal-only intermetallics | 19 | 0.909 [0.744, 1.000] | 0.880 [0.769, 0.980] | 0.876 [0.730, 1.000] |

Coverage of experimental labels stratified (evaluation only) by the compound's own |DFT - experiment| gap:

| stratum (eV/atom) | n/split | in-silico calibrated | recalibrated (m) | expt-trained ref |
|---|---|---|---|---|
| |d|<0.05 | 44.9 | 0.934 [0.847, 1.000] | 0.921 [0.812, 0.990] | 0.921 [0.818, 0.989] |
| 0.05<=|d|<0.15 | 41.4 | 0.915 [0.811, 0.988] | 0.904 [0.758, 1.000] | 0.925 [0.834, 0.988] |
| |d|>=0.15 | 16.7 | 0.792 [0.606, 0.974] | 0.761 [0.485, 1.000] | 0.890 [0.717, 1.000] |

Abstention (share of target candidates whose recalibrated interval is wider than the in-silico interval): constant-width 0.550, sigma-normalised 0.363. Triage (RMSE reduction on the 50 % most confident by tree spread): vs DFT 27.1 [12.3, 44.9] %, vs experiment 26.5 [10.8, 42.8] %; relative AURC vs experiment 0.768 [0.660, 0.883].


k-sweep (coverage of experimental labels / normalised width; k = 0 is the in-silico calibration):

| k | plain recal cov | plain nw | +global bias cov | +global bias nw | d nw (global - plain) | +class bias cov | +class bias nw | DFT look-up cov | DFT look-up nw | Beta theory 5/50/95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.903 | 1.062 | - | - | - | - | - | - | - | - |
| 5 | 1.000 | inf | 1.000 | inf | - | 1.000 | inf | 1.000 | inf | - |
| 9 | 0.877 | 1.181 | 0.866 | 1.211 | 0.029 [-0.250, 0.364] | 0.871 | 1.205 | 0.888 | 1.053 | 0.72, 0.93, 0.99 |
| 10 | 0.909 | 1.272 | 0.896 | 1.298 | 0.026 [-0.233, 0.413] | 0.896 | 1.292 | 0.895 | 1.064 | 0.74, 0.93, 0.99 |
| 20 | 0.900 | 1.136 | 0.899 | 1.141 | 0.005 [-0.201, 0.192] | 0.891 | 1.108 | 0.920 | 0.971 | 0.78, 0.92, 0.98 |
| 30 | 0.889 | 1.068 | 0.897 | 1.103 | 0.035 [-0.135, 0.188] | 0.900 | 1.115 | 0.911 | 0.939 | 0.80, 0.91, 0.97 |
| 50 | 0.898 | 1.074 | 0.894 | 1.054 | -0.019 [-0.118, 0.108] | 0.898 | 1.054 | 0.904 | 0.842 | 0.83, 0.91, 0.96 |
| 100 | 0.898 | 1.054 | 0.901 | 1.057 | 0.003 [-0.064, 0.052] | 0.898 | 1.043 | 0.907 | 0.847 | 0.85, 0.90, 0.94 |
| all (=103) | 0.900 | 1.061 | 0.902 | 1.061 | -0.000 [-0.061, 0.049] | 0.897 | 1.042 | 0.912 | 0.861 | 0.85, 0.91, 0.95 |

k < 9: the finite-sample conformal quantile is infinite at alpha = 0.1 (coverage 1 trivially, infinite interval). 'n/a (pool < k)': the recalibration pool of every split is smaller than k.


Semi-synthetic surrogate-accuracy sweep (sensitivity analysis; real DFT-experiment discrepancy, surrogate error vs DFT rescaled by lambda; prediction = DFT + lambda x (fitted-model error); lambda = 1 is the fitted model, lambda = 0 a perfect DFT surrogate):

| lambda | surrogate RMSE vs DFT | in-silico cov vs DFT | in-silico cov vs expt | in-silico nw | recal (m) cov vs expt | recal (m) nw |
|---|---|---|---|---|---|---|
| 1.0 | 0.143 | 0.912 | 0.903 [0.849, 0.961] | 1.062 | 0.889 [0.761, 0.986] | 1.068 |
| 0.75 | 0.107 | 0.912 | 0.856 [0.796, 0.918] | 0.797 | 0.891 [0.781, 0.981] | 0.911 |
| 0.5 | 0.071 | 0.912 | 0.760 [0.679, 0.841] | 0.531 | 0.897 [0.762, 0.981] | 0.832 |
| 0.35 | 0.050 | 0.912 | 0.655 [0.538, 0.728] | 0.372 | 0.900 [0.752, 0.976] | 0.827 |
| 0.25 | 0.036 | 0.912 | 0.520 [0.397, 0.607] | 0.266 | 0.900 [0.732, 0.986] | 0.837 |
| 0.15 | 0.021 | 0.912 | 0.351 [0.271, 0.467] | 0.159 | 0.904 [0.742, 0.986] | 0.869 |
| 0.1 | 0.014 | 0.912 | 0.240 [0.189, 0.331] | 0.106 | 0.905 [0.748, 0.986] | 0.891 |
| 0.0 | 0.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000 | 0.911 [0.741, 0.990] | 0.939 |

(lambda = 0: zero-width in-silico interval, so DFT-label coverage is trivially 1 and experimental-label coverage is the share of exact DFT = experiment ties.)


### OQMD / HistGB

Sizes (seed 0): {'train': 342, 'cal': 137, 'rpool': 103, 'ttest': 103}; m = [30].

| setting (target-test, 20 splits) | coverage | norm. width (/IQR expt) | interval score (eV/atom) | WIS | cal. error (mean) | worst-cluster cov |
|---|---|---|---|---|---|---|
| (i) in-silico calibrated, vs **DFT** labels (sanity) | 0.910 [0.834, 0.966] | 0.941 [0.781, 1.118] | 0.587 [0.467, 0.785] | 0.057 [0.048, 0.071] | 0.054 [0.017, 0.107] | 0.775 [0.555, 0.916] |
| (i) in-silico calibrated, vs **experimental** labels | 0.862 [0.776, 0.932] | 0.877 [0.728, 1.042] | 0.640 [0.509, 0.714] | 0.067 [0.056, 0.076] | 0.066 [0.023, 0.150] | 0.727 [0.568, 0.884] |
| (ii) recalibrated with m = 30 measurements | 0.888 [0.755, 0.986] | 1.002 [0.693, 1.514] | 0.663 [0.515, 0.861] | 0.067 [0.057, 0.077] | 0.069 [0.032, 0.120] | 0.760 [0.403, 0.951] |
| (iv) + global signed-bias correction (m) | 0.889 [0.727, 0.995] | 1.030 [0.655, 1.678] | 0.676 [0.515, 0.892] | 0.068 [0.057, 0.077] | 0.064 [0.025, 0.112] | 0.760 [0.403, 0.975] |
| (iv) + class-wise bias correction (m) | 0.891 [0.745, 0.990] | 1.016 [0.685, 1.497] | 0.663 [0.516, 0.793] | 0.068 [0.057, 0.077] | 0.068 [0.028, 0.129] | 0.765 [0.451, 0.968] |
| (ii) sigma-normalised recalibration (m) | 0.889 [0.716, 0.976] | 1.018 [0.673, 1.500] | 0.655 [0.498, 0.823] | n/a | n/a | 0.778 [0.478, 0.976] |
| (iii-a) reference: trained + calibrated on experimental labels | 0.923 [0.833, 0.976] | 0.895 [0.748, 1.068] | 0.545 [0.441, 0.649] | 0.054 [0.048, 0.063] | 0.042 [0.020, 0.075] | 0.810 [0.605, 0.928] |
| (iii-b) reference: DFT value itself + m measurements | 0.911 [0.741, 0.990] | 0.939 [0.443, 1.615] | 0.641 [0.499, 0.887] | 0.057 [0.049, 0.066] | 0.082 [0.039, 0.167] | 0.822 [0.611, 0.974] |

(sanity row: width normalised by the IQR of DFT values.)

RMSE on target-test (eV/atom): DFT-trained model vs DFT 0.123 [0.097, 0.154]; vs experiment 0.139 [0.116, 0.156]; experiment-trained model vs experiment 0.114 [0.100, 0.132]; DFT look-up vs experiment 0.122 [0.097, 0.143]. Correlation(model error on DFT, DFT-expt discrepancy) -0.34 [-0.55, -0.10]. Mean signed (prediction - experiment) on test 0.011 [-0.013, 0.035]; bias estimated from m recal records (expt - prediction) -0.009 [-0.055, 0.026].

Paired coverage shortfall (DFT-label coverage - experimental-label coverage, same intervals, same compounds): 0.048 [0.005, 0.112].

| chemistry class (target-test) | n/split | in-silico cov vs DFT | in-silico cov vs expt | expt-trained ref cov |
|---|---|---|---|---|
| f-block (Ln/An) compounds | 43 | 0.967 [0.906, 1.000] | 0.896 [0.764, 0.988] | 0.984 [0.937, 1.000] |
| B/C/Si/Ge compounds | 17 | 0.892 [0.720, 1.000] | 0.846 [0.738, 1.000] | 0.866 [0.665, 1.000] |
| sp-metal intermetallics | 24 | 0.842 [0.648, 0.982] | 0.830 [0.697, 0.960] | 0.887 [0.722, 1.000] |
| transition-metal-only intermetallics | 19 | 0.887 [0.726, 1.000] | 0.828 [0.718, 0.960] | 0.891 [0.665, 1.000] |

Coverage of experimental labels stratified (evaluation only) by the compound's own |DFT - experiment| gap:

| stratum (eV/atom) | n/split | in-silico calibrated | recalibrated (m) | expt-trained ref |
|---|---|---|---|---|
| |d|<0.05 | 44.9 | 0.931 [0.834, 1.000] | 0.938 [0.817, 1.000] | 0.934 [0.813, 1.000] |
| 0.05<=|d|<0.15 | 41.4 | 0.879 [0.732, 0.974] | 0.910 [0.785, 0.989] | 0.930 [0.847, 0.988] |
| |d|>=0.15 | 16.7 | 0.632 [0.482, 0.839] | 0.712 [0.435, 1.000] | 0.876 [0.727, 1.000] |

Abstention (share of target candidates whose recalibrated interval is wider than the in-silico interval): constant-width 0.700, sigma-normalised 0.588. Triage (RMSE reduction on the 50 % most confident by kNN distance): vs DFT 15.6 [-9.9, 32.9] %, vs experiment 12.3 [-2.7, 26.2] %; relative AURC vs experiment 0.854 [0.731, 0.949].


k-sweep (coverage of experimental labels / normalised width; k = 0 is the in-silico calibration):

| k | plain recal cov | plain nw | +global bias cov | +global bias nw | d nw (global - plain) | +class bias cov | +class bias nw | DFT look-up cov | DFT look-up nw | Beta theory 5/50/95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.862 | 0.877 | - | - | - | - | - | - | - | - |
| 5 | 1.000 | inf | 1.000 | inf | - | 1.000 | inf | 1.000 | inf | - |
| 9 | 0.875 | 1.128 | 0.864 | 1.171 | 0.044 [-0.215, 0.462] | 0.864 | 1.168 | 0.888 | 1.053 | 0.72, 0.93, 0.99 |
| 10 | 0.898 | 1.166 | 0.898 | 1.233 | 0.066 [-0.197, 0.427] | 0.895 | 1.229 | 0.895 | 1.064 | 0.74, 0.93, 0.99 |
| 20 | 0.890 | 1.024 | 0.897 | 1.051 | 0.027 [-0.212, 0.160] | 0.893 | 1.020 | 0.920 | 0.971 | 0.78, 0.92, 0.98 |
| 30 | 0.888 | 1.002 | 0.889 | 1.030 | 0.029 [-0.179, 0.198] | 0.891 | 1.016 | 0.911 | 0.939 | 0.80, 0.91, 0.97 |
| 50 | 0.895 | 1.013 | 0.892 | 0.999 | -0.014 [-0.138, 0.100] | 0.895 | 0.989 | 0.904 | 0.842 | 0.83, 0.91, 0.96 |
| 100 | 0.894 | 0.981 | 0.891 | 0.971 | -0.010 [-0.117, 0.067] | 0.899 | 0.969 | 0.907 | 0.847 | 0.85, 0.90, 0.94 |
| all (=103) | 0.899 | 0.994 | 0.894 | 0.980 | -0.014 [-0.105, 0.073] | 0.902 | 0.978 | 0.912 | 0.861 | 0.85, 0.91, 0.95 |

k < 9: the finite-sample conformal quantile is infinite at alpha = 0.1 (coverage 1 trivially, infinite interval). 'n/a (pool < k)': the recalibration pool of every split is smaller than k.


Semi-synthetic surrogate-accuracy sweep (sensitivity analysis; real DFT-experiment discrepancy, surrogate error vs DFT rescaled by lambda; prediction = DFT + lambda x (fitted-model error); lambda = 1 is the fitted model, lambda = 0 a perfect DFT surrogate):

| lambda | surrogate RMSE vs DFT | in-silico cov vs DFT | in-silico cov vs expt | in-silico nw | recal (m) cov vs expt | recal (m) nw |
|---|---|---|---|---|---|---|
| 1.0 | 0.123 | 0.910 | 0.862 [0.776, 0.932] | 0.877 | 0.888 [0.755, 0.986] | 1.002 |
| 0.75 | 0.092 | 0.910 | 0.809 [0.699, 0.883] | 0.658 | 0.898 [0.799, 0.976] | 0.891 |
| 0.5 | 0.061 | 0.910 | 0.708 [0.572, 0.806] | 0.439 | 0.908 [0.805, 0.981] | 0.881 |
| 0.35 | 0.043 | 0.910 | 0.582 [0.437, 0.685] | 0.307 | 0.908 [0.780, 0.986] | 0.877 |
| 0.25 | 0.031 | 0.910 | 0.452 [0.344, 0.550] | 0.219 | 0.903 [0.762, 0.986] | 0.874 |
| 0.15 | 0.018 | 0.910 | 0.287 [0.224, 0.393] | 0.132 | 0.907 [0.765, 0.986] | 0.891 |
| 0.1 | 0.012 | 0.910 | 0.199 [0.141, 0.282] | 0.088 | 0.906 [0.760, 0.986] | 0.906 |
| 0.0 | 0.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000 | 0.911 [0.741, 0.990] | 0.939 |

(lambda = 0: zero-width in-silico interval, so DFT-label coverage is trivially 1 and experimental-label coverage is the share of exact DFT = experiment ties.)


## Reading

- **Headline (honest).** On this genuine computation-to-measurement shift, conformal intervals calibrated only on DFT labels still cover 0.862-0.903 of the *measured* enthalpies at a nominal 0.90 (paired shortfall vs DFT-label coverage MP/RF +0.010 [-0.047, +0.067], MP/HistGB +0.018 [-0.037, +0.082], OQMD/RF +0.008 [-0.034, +0.044], OQMD/HistGB +0.048 [+0.005, +0.112]). There is **no coverage collapse** comparable to the design-region extrapolations of R1 (source-calibrated coverage 0.15-0.76). This does NOT support a claim that in-silico calibration generally collapses between computation and experiment; it supports the narrower statement that the size of the coverage loss depends on how large the computation-measurement discrepancy is relative to the surrogate's own error.
- **Why.** The composition-only surrogates' own error on DFT labels is as large as or larger than the DFT-experiment discrepancy on the same test compounds (RMSE surrogate-vs-DFT / DFT-vs-experiment, ratio, coverage shortfall: MP/RF 0.136 / 0.107 eV/atom, 1.27, +0.010; MP/HistGB 0.115 / 0.107 eV/atom, 1.08, +0.018; OQMD/RF 0.143 / 0.122 eV/atom, 1.17, +0.008; OQMD/HistGB 0.123 / 0.122 eV/atom, 1.01, +0.048), and the surrogate error is negatively correlated with the discrepancy (r = -0.31, -0.31, -0.38, -0.34): the smooth composition model partly averages out compound-specific DFT errors. The in-silico interval is therefore already about as wide as the experiment-trained reference and absorbs most of the discrepancy; the configuration whose surrogate is closest to DFT accuracy shows the largest shortfall.
- **When it would collapse (MP, RF, semi-synthetic).** Shrinking the surrogate's error towards the DFT labels while keeping the real DFT-experiment discrepancy, in-silico-calibrated coverage of experimental values falls: lambda 1.0 (RMSE vs DFT 0.136): 0.90, lambda 0.5 (RMSE vs DFT 0.068): 0.77, lambda 0.25 (RMSE vs DFT 0.034): 0.53, lambda 0.1 (RMSE vs DFT 0.014): 0.25, lambda 0.0 (RMSE vs DFT 0.000): 0.00; recalibration with m measurements restores 0.90, 0.89, 0.89, 0.87, 0.85. A surrogate that reproduces DFT closely inherits DFT's systematic error while its in-silico interval shrinks, so the collapse appears exactly when the in-silico model is good. This is a sensitivity analysis, not an observation of a real, more accurate model.
- **When it would collapse (OQMD, RF, semi-synthetic).** Shrinking the surrogate's error towards the DFT labels while keeping the real DFT-experiment discrepancy, in-silico-calibrated coverage of experimental values falls: lambda 1.0 (RMSE vs DFT 0.143): 0.90, lambda 0.5 (RMSE vs DFT 0.071): 0.76, lambda 0.25 (RMSE vs DFT 0.036): 0.52, lambda 0.1 (RMSE vs DFT 0.014): 0.24, lambda 0.0 (RMSE vs DFT 0.000): 0.00; recalibration with m measurements restores 0.89, 0.90, 0.90, 0.90, 0.91. A surrogate that reproduces DFT closely inherits DFT's systematic error while its in-silico interval shrinks, so the collapse appears exactly when the in-silico model is good. This is a sensitivity analysis, not an observation of a real, more accurate model.
- **Signed-bias correction.** The global DFT-experiment offset is small (MP +0.006, OQMD +0.015 eV/atom; class-level offsets up to 0.049 eV/atom), so estimating it from k measurements adds estimation noise without removing much systematic error: at k = m the change in normalised width is MP/RF +0.040 [-0.095, +0.212]; MP/HistGB +0.025 [-0.130, +0.184]; OQMD/RF +0.035 [-0.135, +0.188]; OQMD/HistGB +0.029 [-0.179, +0.198] (positive = wider). The bias correction does not narrow the intervals here.
- **Conditional coverage.** Coverage is uneven across chemistry classes. In-silico-calibrated, vs experiment: MP/RF 0.84 (B/C/Si/Ge compounds) to 0.96 (f-block (Ln/An) compounds); MP/HistGB 0.83 (sp-metal intermetallics) to 0.95 (f-block (Ln/An) compounds); OQMD/RF 0.84 (B/C/Si/Ge compounds) to 0.96 (f-block (Ln/An) compounds); OQMD/HistGB 0.83 (transition-metal-only intermetallics) to 0.90 (f-block (Ln/An) compounds). The same intervals vs DFT labels: MP/RF 0.84 (sp-metal intermetallics) to 0.98 (f-block (Ln/An) compounds); MP/HistGB 0.82 (sp-metal intermetallics) to 0.97 (f-block (Ln/An) compounds); OQMD/RF 0.86 (sp-metal intermetallics) to 0.96 (f-block (Ln/An) compounds); OQMD/HistGB 0.84 (sp-metal intermetallics) to 0.97 (f-block (Ln/An) compounds); experiment-trained reference: MP/RF 0.86 (transition-metal-only intermetallics) to 0.98 (f-block (Ln/An) compounds); MP/HistGB 0.85 (sp-metal intermetallics) to 0.97 (f-block (Ln/An) compounds); OQMD/RF 0.85 (B/C/Si/Ge compounds) to 0.99 (f-block (Ln/An) compounds); OQMD/HistGB 0.87 (B/C/Si/Ge compounds) to 0.98 (f-block (Ln/An) compounds). The unevenness is present against DFT labels and for the experiment-trained reference too, so it is mostly heteroscedastic model error, not the computation-measurement shift. The shift does bite on the compounds whose own DFT-experiment gap is large (|d|>=0.15 eV/atom, ~12 test records/split): coverage MP/RF in-silico 0.74, recalibrated 0.77, expt-trained 0.85; MP/HistGB in-silico 0.69, recalibrated 0.71, expt-trained 0.82; OQMD/RF in-silico 0.79, recalibrated 0.76, expt-trained 0.89; OQMD/HistGB in-silico 0.63, recalibrated 0.71, expt-trained 0.88. Marginal (global) recalibration does not repair this sub-group; these compounds cannot be identified before measurement, which is the practical limit of any marginal guarantee.
- **Resampling check (MP).** With the 20 protocol splits the DFT look-up conformalised on m measurements averages 0.853; the same model-free procedure over 500 splits averages 0.896 (per-split SD 0.066; finite-sample expectation 0.903); the sub-nominal 20-split mean is resampling noise of the 20 splits, not a violated split-conformal guarantee.
- **Resampling check (OQMD).** With the 20 protocol splits the DFT look-up conformalised on m measurements averages 0.911; the same model-free procedure over 500 splits averages 0.902 (per-split SD 0.062; finite-sample expectation 0.903); the 20-split mean is consistent with the split-conformal guarantee.
- **Measurement budget.** The in-silico route uses 0 measured labels for training and calibration; the experiment-trained reference uses the measured labels of train + calibration compounds (MP ~450, OQMD ~479 records). Normalised width / coverage, in-silico vs experiment-trained: MP/RF 1.06 / 0.90 vs 1.05 / 0.91; MP/HistGB 0.89 / 0.89 vs 0.88 / 0.91; OQMD/RF 1.06 / 0.90 vs 1.03 / 0.92; OQMD/HistGB 0.88 / 0.86 vs 0.90 / 0.92. For this property and these surrogates, DFT labels are nearly as informative as measurements; recalibration with m = 30 measurements is a check that costs 30 experiments and here confirms rather than repairs the in-silico interval.
- **Scope.** Inorganic intermetallic/silicide/boride/carbide compounds, formation enthalpy, composition-only descriptors, and only ~320 DFT training labels (the matched subset); not biomedical materials. A genuine computation-to-measurement analogue, not the in-silico -> in-vitro -> in-vivo gap. Splits resample one dataset; percentile intervals describe split-to-split variability.
- **MP / RF.** In-silico-calibrated intervals cover 0.907 of held-out DFT labels but 0.897 [0.84, 0.95] of the measured values of the same compounds (shortfall +0.010). Recalibrating on m = 30 measurements gives 0.901 [0.82, 0.98] at normalised width 1.090 (in-silico 1.055; experiment-trained reference 0.910 at 1.050). A global signed-bias correction changes the width by +0.040 IQR [-0.095, +0.212] at k = m.
- **MP / HistGB.** In-silico-calibrated intervals cover 0.907 of held-out DFT labels but 0.890 [0.80, 0.94] of the measured values of the same compounds (shortfall +0.018). Recalibrating on m = 30 measurements gives 0.906 [0.79, 0.98] at normalised width 0.985 (in-silico 0.891; experiment-trained reference 0.909 at 0.880). A global signed-bias correction changes the width by +0.025 IQR [-0.130, +0.184] at k = m.
- **OQMD / RF.** In-silico-calibrated intervals cover 0.912 of held-out DFT labels but 0.903 [0.85, 0.96] of the measured values of the same compounds (shortfall +0.008). Recalibrating on m = 30 measurements gives 0.889 [0.76, 0.99] at normalised width 1.068 (in-silico 1.062; experiment-trained reference 0.917 at 1.033). A global signed-bias correction changes the width by +0.035 IQR [-0.135, +0.188] at k = m.
- **OQMD / HistGB.** In-silico-calibrated intervals cover 0.910 of held-out DFT labels but 0.862 [0.78, 0.93] of the measured values of the same compounds (shortfall +0.048). Recalibrating on m = 30 measurements gives 0.888 [0.75, 0.99] at normalised width 1.002 (in-silico 0.877; experiment-trained reference 0.923 at 0.895). A global signed-bias correction changes the width by +0.029 IQR [-0.179, +0.198] at k = m.

## Deviations from PROTOCOL.md and notes

- Split fractions 50/20/30 (train / calibration / target pool) over one pool of matched compounds, as specified for this analysis, instead of the 60/20/20 source split plus a separate design-region target pool: here source and target are the same compounds and the shift is in how the label is obtained (DFT vs calorimetry), so there is no design axis and no source-test half.
- 'In-distribution' coverage is the DFT-label coverage on the target-test compounds (sanity row); there is no separate source-test set.
- Normalised width uses the IQR of the experimental values over all matched records of that source (a fixed normalising constant, as R1 uses the IQR of the whole target pool); it is not used for any choice.
- Sigma for normalised scores / triage: RF tree spread (as in R1); HistGB has no ensemble spread, so a model-agnostic proxy (mean distance to the 10 nearest training compositions, standardised magpie space) is used.
- Width-stratified coverage equals marginal coverage for constant-width intervals; it is informative only for the sigma-normalised rows. Abstention for constant-width intervals is 0/1 per split (share of splits in which the recalibrated interval is wider than the in-silico one).
- Bias-corrected recalibration uses leave-one-out calibration residuals (bias estimated from the same k records); this is a jackknife-style construction whose finite-sample guarantee is not exactly the split-conformal one.
- k = 100 exceeds the recalibration pool for MP (~97 records) and is reported as n/a; 'all' uses the whole pool.
- Surrogate-accuracy sweep and the 500-split look-up check are additional diagnostics (semi-synthetic / model-free), labelled as such; the 500-split check reuses the same split function with seeds 0..499.
- RF uses common.make_model with n_jobs set to 1 inside 6 joblib workers (CPU sharing); results do not depend on n_jobs.
