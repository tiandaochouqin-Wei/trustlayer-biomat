# R8 budget: the cost of target recalibration and a retrospective replay

Script `revision/code/exp_R8_budget.py`; JSON `revision/results/R8_budget.json`. Reference model RF (common.make_model), unified protocol (grouped 60/20/20 source split; target pool halves = recalibration pool / target-test), alpha = 0.10 unless stated, finite-sample conformal quantile, 20 splits (seeds 0-19). Intervals across splits are 2.5-97.5 percentiles; the splits resample one finite dataset, so they describe split-to-split variability, not population sampling error. Runtime 1.2 min.

Scope statements that apply to everything below:

* Unmeasured candidates receive calibrated **intervals**, not measured values. Any 'saving' is a saving in coverage-certified-prediction terms only (fewer candidates need a measurement to obtain an interval with the stated marginal coverage); it is not a saving in experimental outcomes, and the interval is wide (normalised widths below).
* The replay assumes that outcomes do not depend on acquisition order, uniform cost per measurement, no batch or instrument effects, and that historical records are an unbiased stand-in for fresh measurements. It is a retrospective look-up-table replay of existing records, **not a prospective closed loop**.
* Quantities marked *oracle* read target-test / remainder labels and are descriptive; nothing is selected with them. Protocol-compliant counterparts (selection inside the recalibration pool only, or label-free pre-registered stopping rules) are reported alongside.

## (a) Realised target-test coverage vs recalibration size k

Recalibration on the first k records of the recalibration pool, evaluation on the same target-test half. `Beta` = exchangeable conditional-coverage law for an infinite test set (common.beta_coverage_quantiles); `BetaBin` = exact exchangeable law for the actual target-test size (mixed over splits). k = 0 is the source-calibrated interval; k < 9 gives an infinite interval at alpha = 0.10 (reported, not dropped). `nw` = interval width / IQR of the target property; `res p5` = 5th percentile over 20 x 200 random k-subsets of the pool (supplement).

### glass_Tg (|T| = 258, pool 129, target-test 129, headline m = 30)

| k | inf. | cov mean [2.5, 97.5] | p5 / p50 / p95 | Beta 5/50/95 | BetaBin 5/50/95 | frac >= 0.90 | res p5 | nw median |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.403 [0.33, 0.48] | 0.34 / 0.40 / 0.48 | n/a | n/a | 0.00 | n/a | 0.61 |
| 4 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 5 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 9 | 0 | 0.894 [0.71, 1.00] | 0.71 / 0.93 / 1.00 | 0.72/0.93/0.99 | 0.71/0.93/1.00 | 0.60 | 0.73 | 2.77 |
| 10 | 0 | 0.897 [0.71, 1.00] | 0.71 / 0.93 / 1.00 | 0.74/0.93/0.99 | 0.74/0.93/1.00 | 0.60 | 0.78 | 2.77 |
| 15 | 0 | 0.935 [0.76, 1.00] | 0.83 / 0.95 / 1.00 | 0.82/0.95/1.00 | 0.81/0.95/1.00 | 0.80 | 0.84 | 3.01 |
| 20 | 0 | 0.924 [0.84, 0.98] | 0.84 / 0.94 / 0.98 | 0.78/0.92/0.98 | 0.78/0.91/0.98 | 0.70 | 0.80 | 2.77 |
| 21 | 0 | 0.926 [0.84, 0.99] | 0.84 / 0.94 / 0.98 | 0.79/0.92/0.98 | 0.78/0.92/0.99 | 0.70 | 0.81 | 2.77 |
| 30 | 0 | 0.930 [0.84, 0.99] | 0.84 / 0.94 / 0.98 | 0.80/0.91/0.97 | 0.79/0.91/0.98 | 0.80 | 0.82 | 2.72 |
| 40 | 0 | 0.924 [0.84, 0.99] | 0.84 / 0.93 / 0.98 | 0.82/0.91/0.97 | 0.81/0.91/0.98 | 0.75 | 0.84 | 2.57 |
| 60 | 0 | 0.921 [0.83, 0.97] | 0.84 / 0.93 / 0.97 | 0.83/0.91/0.96 | 0.82/0.91/0.97 | 0.80 | 0.85 | 2.63 |
| 80 | 0 | 0.912 [0.87, 0.96] | 0.87 / 0.91 / 0.95 | 0.84/0.90/0.95 | 0.82/0.91/0.96 | 0.60 | 0.87 | 2.53 |
| 120 | 0 | 0.918 [0.87, 0.96] | 0.88 / 0.91 / 0.95 | 0.85/0.90/0.94 | 0.84/0.91/0.95 | 0.75 | 0.87 | 2.57 |
| 129 (pool) | 0 | 0.918 [0.88, 0.96] | 0.88 / 0.91 / 0.95 | 0.85/0.90/0.94 | 0.84/0.90/0.95 | 0.75 | 0.88 | 2.55 |

### glass_E (|T| = 67, pool 34, target-test 33, headline m = 22)

| k | inf. | cov mean [2.5, 97.5] | p5 / p50 / p95 | Beta 5/50/95 | BetaBin 5/50/95 | frac >= 0.90 | res p5 | nw median |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.230 [0.12, 0.33] | 0.12 / 0.24 / 0.33 | n/a | n/a | 0.00 | n/a | 0.77 |
| 4 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 5 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 9 | 0 | 0.920 [0.74, 1.00] | 0.78 / 0.94 / 1.00 | 0.72/0.93/0.99 | 0.70/0.94/1.00 | 0.65 | 0.67 | 3.68 |
| 10 | 0 | 0.930 [0.77, 1.00] | 0.84 / 0.94 / 1.00 | 0.74/0.93/0.99 | 0.73/0.94/1.00 | 0.75 | 0.70 | 3.74 |
| 15 | 0 | 0.944 [0.80, 1.00] | 0.84 / 0.97 / 1.00 | 0.82/0.95/1.00 | 0.79/0.97/1.00 | 0.80 | 0.82 | 3.89 |
| 20 | 0 | 0.905 [0.73, 1.00] | 0.75 / 0.91 / 1.00 | 0.78/0.92/0.98 | 0.76/0.91/1.00 | 0.60 | 0.76 | 3.52 |
| 21 | 0 | 0.911 [0.75, 1.00] | 0.81 / 0.92 / 1.00 | 0.79/0.92/0.98 | 0.76/0.94/1.00 | 0.65 | 0.79 | 3.52 |
| 22 | 0 | 0.921 [0.75, 1.00] | 0.81 / 0.94 / 1.00 | 0.80/0.92/0.98 | 0.76/0.94/1.00 | 0.75 | 0.82 | 3.53 |
| 30 | 0 | 0.911 [0.75, 1.00] | 0.81 / 0.92 / 1.00 | 0.80/0.91/0.97 | 0.76/0.91/1.00 | 0.60 | 0.82 | 3.60 |
| 34 (pool) | 0 | 0.926 [0.83, 1.00] | 0.85 / 0.94 / 1.00 | 0.83/0.92/0.98 | 0.79/0.94/1.00 | 0.70 | 0.85 | 3.65 |

### glass_HV (|T| = 71, pool 36, target-test 35, headline m = 23)

| k | inf. | cov mean [2.5, 97.5] | p5 / p50 / p95 | Beta 5/50/95 | BetaBin 5/50/95 | frac >= 0.90 | res p5 | nw median |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.210 [0.14, 0.29] | 0.14 / 0.20 / 0.29 | n/a | n/a | 0.00 | n/a | 1.42 |
| 4 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 5 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 9 | 0 | 0.921 [0.80, 1.00] | 0.80 / 0.93 / 1.00 | 0.72/0.93/0.99 | 0.69/0.94/1.00 | 0.65 | 0.74 | 4.29 |
| 10 | 0 | 0.926 [0.83, 1.00] | 0.85 / 0.93 / 1.00 | 0.74/0.93/0.99 | 0.71/0.94/1.00 | 0.70 | 0.74 | 4.29 |
| 15 | 0 | 0.957 [0.87, 1.00] | 0.88 / 0.97 / 1.00 | 0.82/0.95/1.00 | 0.80/0.97/1.00 | 0.85 | 0.80 | 6.73 |
| 20 | 0 | 0.919 [0.80, 1.00] | 0.83 / 0.93 / 1.00 | 0.78/0.92/0.98 | 0.74/0.91/1.00 | 0.60 | 0.77 | 4.86 |
| 21 | 0 | 0.927 [0.83, 1.00] | 0.83 / 0.93 / 1.00 | 0.79/0.92/0.98 | 0.77/0.91/1.00 | 0.60 | 0.77 | 4.90 |
| 23 | 0 | 0.933 [0.84, 1.00] | 0.86 / 0.93 / 1.00 | 0.81/0.93/0.98 | 0.77/0.94/1.00 | 0.70 | 0.77 | 5.04 |
| 30 | 0 | 0.907 [0.79, 0.99] | 0.80 / 0.91 / 0.97 | 0.80/0.91/0.97 | 0.77/0.91/1.00 | 0.65 | 0.77 | 4.34 |
| 36 (pool) | 0 | 0.913 [0.79, 0.99] | 0.80 / 0.91 / 0.97 | 0.84/0.93/0.98 | 0.80/0.94/1.00 | 0.75 | 0.80 | 4.76 |

### glass_Tliq (|T| = 131, pool 66, target-test 65, headline m = 30)

| k | inf. | cov mean [2.5, 97.5] | p5 / p50 / p95 | Beta 5/50/95 | BetaBin 5/50/95 | frac >= 0.90 | res p5 | nw median |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.406 [0.33, 0.49] | 0.34 / 0.41 / 0.46 | n/a | n/a | 0.00 | n/a | 0.71 |
| 4 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 5 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 9 | 0 | 0.888 [0.71, 1.00] | 0.71 / 0.92 / 1.00 | 0.72/0.93/0.99 | 0.71/0.92/1.00 | 0.60 | 0.69 | 2.78 |
| 10 | 0 | 0.914 [0.71, 1.00] | 0.71 / 0.95 / 1.00 | 0.74/0.93/0.99 | 0.72/0.94/1.00 | 0.70 | 0.71 | 2.89 |
| 15 | 0 | 0.931 [0.75, 1.00] | 0.80 / 0.96 / 1.00 | 0.82/0.95/1.00 | 0.80/0.95/1.00 | 0.75 | 0.80 | 3.93 |
| 20 | 0 | 0.892 [0.72, 1.00] | 0.74 / 0.91 / 1.00 | 0.78/0.92/0.98 | 0.77/0.92/1.00 | 0.55 | 0.75 | 2.79 |
| 21 | 0 | 0.900 [0.72, 1.00] | 0.74 / 0.92 / 1.00 | 0.79/0.92/0.98 | 0.77/0.92/1.00 | 0.60 | 0.77 | 2.80 |
| 30 | 0 | 0.898 [0.75, 0.99] | 0.80 / 0.92 / 0.97 | 0.80/0.91/0.97 | 0.78/0.91/0.98 | 0.60 | 0.78 | 2.81 |
| 40 | 0 | 0.902 [0.81, 0.99] | 0.81 / 0.91 / 0.97 | 0.82/0.91/0.97 | 0.80/0.91/0.98 | 0.55 | 0.80 | 2.72 |
| 60 | 0 | 0.905 [0.83, 0.99] | 0.86 / 0.91 / 0.97 | 0.83/0.91/0.96 | 0.80/0.91/0.97 | 0.50 | 0.80 | 2.81 |
| 66 (pool) | 0 | 0.912 [0.83, 0.99] | 0.86 / 0.92 / 0.97 | 0.85/0.91/0.96 | 0.82/0.92/0.98 | 0.55 | 0.86 | 2.81 |

### steel_yield (|T| = 41, pool 20, target-test 21, headline m = 13)

| k | inf. | cov mean [2.5, 97.5] | p5 / p50 / p95 | Beta 5/50/95 | BetaBin 5/50/95 | frac >= 0.90 | res p5 | nw median |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.155 [0.02, 0.34] | 0.05 / 0.14 / 0.29 | n/a | n/a | 0.00 | n/a | 0.56 |
| 4 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 5 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 9 | 0 | 0.926 [0.78, 1.00] | 0.81 / 0.95 / 1.00 | 0.72/0.93/0.99 | 0.67/0.95/1.00 | 0.70 | 0.71 | 2.97 |
| 10 | 0 | 0.929 [0.78, 1.00] | 0.81 / 0.95 / 1.00 | 0.74/0.93/0.99 | 0.71/0.95/1.00 | 0.75 | 0.76 | 2.97 |
| 13 | 0 | 0.952 [0.83, 1.00] | 0.85 / 0.95 / 1.00 | 0.79/0.95/1.00 | 0.76/0.95/1.00 | 0.90 | 0.81 | 3.10 |
| 15 | 0 | 0.955 [0.83, 1.00] | 0.85 / 0.98 / 1.00 | 0.82/0.95/1.00 | 0.76/0.95/1.00 | 0.90 | 0.81 | 3.11 |
| 20 | 0 | 0.919 [0.76, 1.00] | 0.76 / 0.95 / 1.00 | 0.78/0.92/0.98 | 0.71/0.90/1.00 | 0.70 | 0.76 | 3.02 |
| 20 (pool) | 0 | 0.919 [0.76, 1.00] | 0.76 / 0.95 / 1.00 | 0.78/0.92/0.98 | 0.71/0.90/1.00 | 0.70 | 0.76 | 3.02 |

### polymer_Tg (|T| = 963, pool 482, target-test 481, headline m = 30)

| k | inf. | cov mean [2.5, 97.5] | p5 / p50 / p95 | Beta 5/50/95 | BetaBin 5/50/95 | frac >= 0.90 | res p5 | nw median |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.763 [0.72, 0.82] | 0.73 / 0.76 / 0.82 | n/a | n/a | 0.00 | n/a | 1.34 |
| 4 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 5 | 20 | 1.000 [1.00, 1.00] | 1.00 / 1.00 / 1.00 | n/a | inf | 1.00 | 1.00 | n/a |
| 9 | 0 | 0.850 [0.65, 0.99] | 0.71 / 0.88 / 0.98 | 0.72/0.93/0.99 | 0.72/0.93/1.00 | 0.50 | 0.71 | 1.80 |
| 10 | 0 | 0.865 [0.72, 0.99] | 0.73 / 0.88 / 0.98 | 0.74/0.93/0.99 | 0.74/0.93/1.00 | 0.50 | 0.74 | 1.83 |
| 15 | 0 | 0.913 [0.73, 1.00] | 0.74 / 0.96 / 1.00 | 0.82/0.95/1.00 | 0.82/0.95/1.00 | 0.70 | 0.81 | 2.66 |
| 20 | 0 | 0.888 [0.77, 0.97] | 0.79 / 0.91 / 0.96 | 0.78/0.92/0.98 | 0.78/0.92/0.98 | 0.55 | 0.78 | 1.97 |
| 21 | 0 | 0.888 [0.77, 0.97] | 0.79 / 0.91 / 0.96 | 0.79/0.92/0.98 | 0.79/0.92/0.99 | 0.55 | 0.78 | 1.97 |
| 30 | 0 | 0.902 [0.82, 0.98] | 0.83 / 0.91 / 0.98 | 0.80/0.91/0.97 | 0.80/0.91/0.97 | 0.55 | 0.79 | 1.96 |
| 40 | 0 | 0.901 [0.83, 0.97] | 0.84 / 0.90 / 0.97 | 0.82/0.91/0.97 | 0.81/0.91/0.97 | 0.50 | 0.80 | 1.94 |
| 60 | 0 | 0.902 [0.84, 0.97] | 0.84 / 0.91 / 0.96 | 0.83/0.91/0.96 | 0.83/0.91/0.96 | 0.60 | 0.82 | 1.96 |
| 80 | 0 | 0.912 [0.85, 0.96] | 0.87 / 0.91 / 0.95 | 0.84/0.90/0.95 | 0.84/0.90/0.95 | 0.70 | 0.82 | 2.05 |
| 120 | 0 | 0.909 [0.87, 0.96] | 0.87 / 0.91 / 0.95 | 0.85/0.90/0.94 | 0.85/0.90/0.95 | 0.65 | 0.83 | 1.98 |
| 140 | 0 | 0.907 [0.87, 0.95] | 0.87 / 0.91 / 0.94 | 0.86/0.90/0.94 | 0.85/0.90/0.94 | 0.55 | 0.84 | 1.96 |
| 200 | 0 | 0.899 [0.84, 0.94] | 0.84 / 0.91 / 0.94 | 0.86/0.90/0.93 | 0.86/0.90/0.94 | 0.60 | 0.84 | 1.97 |
| 300 | 0 | 0.899 [0.84, 0.92] | 0.86 / 0.90 / 0.92 | 0.87/0.90/0.93 | 0.86/0.90/0.93 | 0.65 | 0.84 | 1.96 |
| 482 (pool) | 0 | 0.895 [0.85, 0.93] | 0.87 / 0.89 / 0.93 | 0.88/0.90/0.92 | 0.87/0.90/0.93 | 0.40 | 0.87 | 1.96 |

## (b) Smallest adequate k (5th-percentile coverage >= 0.80 and mean >= 0.88)

*Oracle* columns use target-test labels (descriptive). 'first' = smallest grid k meeting the criterion; 'stable' = smallest k from which every larger grid k also meets it. With 20 splits the 5th percentile is essentially the second-lowest split, so for the small target regions the oracle labels are noise-level; the reliable reference is the Beta-Binomial theory column (exact exchangeable law for the actual target-test size, evaluated at every integer k, not only the grid).

| dataset | |T| | pool | oracle first / stable | k/|T| (first) | resampled first | BetaBin theory, all integer k: first / stable (stable within pool) |
|---|---|---|---|---|---|---|
| glass_Tg | 258 | 129 | 15 / 15 | 0.06 | 15 | 15 / 40 (40, pool 129; n_test 129) |
| glass_E | 67 | 34 | 10 / 21 | 0.15 | 15 | 16 / 310 (never, pool 34; n_test 33) |
| glass_HV | 71 | 36 | 9 / never | 0.13 | 15 | 15 / 60 (34, pool 36; n_test 35) |
| glass_Tliq | 131 | 66 | 40 / 40 | 0.31 | 15 | 14 / 40 (40, pool 66; n_test 65) |
| steel_yield | 41 | 20 | 9 / never | 0.22 | 13 | 16 / never (never, pool 20; n_test 21) |
| polymer_Tg | 963 | 482 | 30 / 30 | 0.03 | 15 | 14 / 30 (30, pool 482; n_test 481) |

Exchangeable theory with an infinite test set (Beta): 5th percentile >= 0.80 first at n = 14, stably from n = 30.

Oracle near misses (5th percentile within 0.01 below 0.80): glass_HV: p5 = 0.7986 at k = 30; glass_HV: p5 = 0.7986 at k = 36 (pool); glass_Tliq: p5 = 0.7954 at k = 15; glass_Tliq: p5 = 0.7954 at k = 30; polymer_Tg: p5 = 0.7923 at k = 20; polymer_Tg: p5 = 0.7923 at k = 21.

**Protocol-compliant selection.** k is chosen from the recalibration pool alone and then evaluated once on target-test. Each of 2000 inner draws is a grouped permutation of the pool; the first k records calibrate (this is the deployed first-k rule) and the remaining pool records whose composition does not occur in the calibration set are held out (whole compositions only, >= 10 records). The first grid k whose inner-holdout coverage has 5th percentile >= 0.80 and mean >= 0.88 is chosen. Primary = inner-RNG replicate 0; 5 independent replicates give the Monte Carlo spread of the certification count. The inner holdout (pool - k records) is smaller than the target-test half, so the inner 5th percentile is more pessimistic than the target-test one. Record-level inner splits (the previous version, which put replicate records on both sides of the inner split and so violated the grouped protocol) are shown as a sensitivity only.

| dataset | pool compositions (mean) | certified splits (range over replicates) | splits whose certification / chosen k changes across replicates | chosen k (primary) | target-test cov at chosen k: mean, p5, min | record-level (sensitivity): certified, chosen k, target-test p5 |
|---|---|---|---|---|---|---|
| glass_Tg | 125 | 20/20 (17-20) | 4 / 9 | 15x18, 40x1, 60x1 | 0.937, 0.84, 0.70 | 20/20, 15x19, 60x1, 0.83 |
| glass_E | 34 | 0/20 (0-0) | 0 / 0 | none | n/a | 0/20, none, n/a |
| glass_HV | 36 | 6/20 (3-11) | 17 / 17 | 15x6 | 0.943, 0.89, 0.89 | 1/20, 15x1, 0.86 |
| glass_Tliq | 62 | 16/20 (14-16) | 13 / 13 | 15x16 | 0.918, 0.78, 0.71 | 20/20, 15x20, 0.80 |
| steel_yield | 20 | 0/20 (0-0) | 0 / 0 | none | n/a | 0/20, none, n/a |
| polymer_Tg | 362 | 20/20 (20-20) | 0 / 14 | 15x2, 40x3, 60x15 | 0.902, 0.84, 0.82 | 20/20, 15x20, 0.74 |

## (c) Chebyshev budget rule (submitted manuscript) vs exact Beta rule

Chebyshev: m = alpha(1-alpha)/(delta eps^2) - 1 (rounded up), as in the submitted manuscript and demo/exp6. Exact: smallest n with P_Beta(coverage < 1-alpha-eps) <= delta, where coverage ~ Beta(j, n+1-j), j = ceil((n+1)(1-alpha)) (conditional coverage over an infinite test population). Because j is discrete the failure probability is saw-toothed in n; 'first' is the first n meeting the target, 'stable' the n from which it is met for every larger n. Cantelli = one-sided Chebyshev with the same variance bound.

alpha = 0.10

| eps | delta | Chebyshev m | Cantelli m | exact Beta first / stable | Chebyshev / exact | P_Beta(fail) at Chebyshev m |
|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 719 | 683 | 68 / 110 | 6.54 | 3.3e-05 |
| 0.05 | 0.10 | 359 | 323 | 15 / 60 | 5.98 | 0.0022 |
| 0.08 | 0.05 | 281 | 267 | 16 / 41 | 6.85 | 6.2e-05 |
| 0.08 | 0.10 | 140 | 126 | 12 / 21 | 6.67 | 0.003 |
| 0.10 | 0.05 | 179 | 170 | 14 / 30 | 5.97 | 0.00011 |
| 0.10 | 0.10 | 89 | 80 | 11 / 11 | 8.09 | 0.004 |

alpha = 0.20

| eps | delta | Chebyshev m | Cantelli m | exact Beta first / stable | Chebyshev / exact | P_Beta(fail) at Chebyshev m |
|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 1279 | 1215 | 153 / 185 | 6.91 | 1.1e-05 |
| 0.05 | 0.10 | 639 | 575 | 73 / 105 | 6.09 | 0.0013 |
| 0.08 | 0.05 | 499 | 474 | 53 / 70 | 7.13 | 1.7e-05 |
| 0.08 | 0.10 | 249 | 224 | 8 / 40 | 6.22 | 0.0016 |
| 0.10 | 0.05 | 319 | 303 | 28 / 45 | 7.09 | 2.2e-05 |
| 0.10 | 0.10 | 159 | 143 | 7 / 25 | 6.36 | 0.0018 |

Variance bound used in the manuscript (Var <= alpha(1-alpha)/(m+1)) holds for n = 9..2000: True (max ratio 1.000).

Per dataset, the realised coverage is measured on a finite target-test half, so binomial test-set noise adds to the calibration-set noise. As k grows, the exact finite-test (Beta-Binomial) failure probability tends to the Binomial(n_test, 0.9) tail. This is a **limit, not a floor**: at conservative saw-tooth sizes (the conformal order statistic is the sample maximum or close to it) the failure probability dips below it. Where the limit exceeds delta, no k meets delta *stably* ('stable' = never), but delta can still be met at such dips; the table gives the minimum over k and the k values (up to 200, and within the recalibration pool) at which delta is met. Empirical = resampled k-subsets of the pool.

| dataset | n_test | eps | delta | empirical first / stable k (resampled, grid) | BetaBin first / stable k | BetaBin limit as k->inf | BetaBin min P(fail) over k <= 200 (at k) | k meeting delta (<= 200) | within pool | Chebyshev m | Chebyshev m feasible in pool / in replay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glass_Tg | 129 | 0.05 | 0.05 | 60 / 60 | 308 / 490 | 0.032 | 0.057 (198) | none | none | 719 | no / no |
| glass_Tg | 129 | 0.05 | 0.10 | 15 / 21 | 15 / 110 | 0.032 | 0.057 (198) | 15-18, 26-28, 36-38, 46-48, ... | 15-18, 26-28, 36-38, 46-48, ... | 359 | no / no |
| glass_Tg | 129 | 0.08 | 0.05 | 15 / 30 | 16 / 60 | 0.002 | 0.010 (198) | 16-18, 26-28, 35-38, 44-48, ... | 16-18, 26-28, 35-38, 44-48, ... | 281 | no / no |
| glass_Tg | 129 | 0.08 | 0.10 | 10 / 10 | 12 / 21 | 0.002 | 0.010 (198) | 12-18, 21-200 | 12-18, 21-129 | 140 | no / yes |
| glass_Tg | 129 | 0.10 | 0.05 | 15 / 21 | 15 / 40 | 0.000 | 0.004 (198) | 15-18, 24-28, 32-38, 40-200 | 15-18, 24-28, 32-38, 40-129 | 179 | no / yes |
| glass_Tg | 129 | 0.10 | 0.10 | 10 / 10 | 11 / 11 | 0.000 | 0.004 (198) | 11-200 | 11-129 | 89 | yes / yes |
| glass_E | 33 | 0.05 | 0.05 | never / never | never / never | 0.230 (> delta: not stably reachable) | 0.101 (18) | none | none | 719 | no / no |
| glass_E | 33 | 0.05 | 0.10 | never / never | never / never | 0.230 (> delta: not stably reachable) | 0.101 (18) | none | none | 359 | no / no |
| glass_E | 33 | 0.08 | 0.05 | 34 / 34 | never / never | 0.106 (> delta: not stably reachable) | 0.061 (18) | none | none | 281 | no / no |
| glass_E | 33 | 0.08 | 0.10 | 15 / 22 | 15 / never | 0.106 (> delta: not stably reachable) | 0.061 (18) | 15-18, 26-28, 37-38, 48, ... | 15-18, 26-28 | 140 | no / no |
| glass_E | 33 | 0.10 | 0.05 | 15 / 22 | 16 / 310 | 0.042 | 0.037 (18) | 16-18, 28, 38, 48, ... | 16-18, 28 | 179 | no / no |
| glass_E | 33 | 0.10 | 0.10 | 10 / 10 | 12 / 30 | 0.042 | 0.037 (18) | 12-18, 21-28, 30-200 | 12-18, 21-28, 30-34 | 89 | no / no |
| glass_HV | 35 | 0.05 | 0.05 | never / never | never / never | 0.132 (> delta: not stably reachable) | 0.071 (18) | none | none | 719 | no / no |
| glass_HV | 35 | 0.05 | 0.10 | 15 / never | 16 / never | 0.132 (> delta: not stably reachable) | 0.071 (18) | 16-18, 28 | 16-18, 28 | 359 | no / no |
| glass_HV | 35 | 0.08 | 0.05 | never / never | 18 / never | 0.055 (> delta: not stably reachable) | 0.044 (18) | 18 | 18 | 281 | no / no |
| glass_HV | 35 | 0.08 | 0.10 | 15 / 36 | 13 / 50 | 0.055 | 0.044 (18) | 13-18, 23-28, 32-38, 41-48, ... | 13-18, 23-28, 32-36 | 140 | no / no |
| glass_HV | 35 | 0.10 | 0.05 | 15 / 36 | 15 / 60 | 0.020 | 0.024 (198) | 15-18, 25-28, 34-38, 43-48, ... | 15-18, 25-28, 34-36 | 179 | no / no |
| glass_HV | 35 | 0.10 | 0.10 | 10 / 10 | 11 / 11 | 0.020 | 0.024 (198) | 11-200 | 11-36 | 89 | no / no |
| glass_Tliq | 65 | 0.05 | 0.05 | 66 / 66 | never / never | 0.111 (> delta: not stably reachable) | 0.074 (18) | none | none | 719 | no / no |
| glass_Tliq | 65 | 0.05 | 0.10 | 15 / 60 | 16 / never | 0.111 (> delta: not stably reachable) | 0.074 (18) | 16-18, 28 | 16-18, 28 | 359 | no / no |
| glass_Tliq | 65 | 0.08 | 0.05 | 66 / 66 | 17 / 170 | 0.026 | 0.037 (198) | 17-18, 28, 58, 68, ... | 17-18, 28, 58 | 281 | no / no |
| glass_Tliq | 65 | 0.08 | 0.10 | 15 / 40 | 13 / 40 | 0.026 | 0.037 (198) | 13-18, 23-28, 32-38, 40-200 | 13-18, 23-28, 32-38, 40-66 | 140 | no / no |
| glass_Tliq | 65 | 0.10 | 0.05 | 15 / 40 | 14 / 40 | 0.004 | 0.009 (198) | 14-18, 24-28, 32-38, 40-200 | 14-18, 24-28, 32-38, 40-66 | 179 | no / no |
| glass_Tliq | 65 | 0.10 | 0.10 | 15 / 15 | 11 / 11 | 0.004 | 0.009 (198) | 11-200 | 11-66 | 89 | no / yes |
| steel_yield | 21 | 0.05 | 0.05 | never / never | never / never | 0.152 (> delta: not stably reachable) | 0.073 (18) | none | none | 719 | no / no |
| steel_yield | 21 | 0.05 | 0.10 | 15 / never | 16 / never | 0.152 (> delta: not stably reachable) | 0.073 (18) | 16-18, 28 | 16-18 | 359 | no / no |
| steel_yield | 21 | 0.08 | 0.05 | never / never | never / never | 0.152 (> delta: not stably reachable) | 0.073 (18) | none | none | 281 | no / no |
| steel_yield | 21 | 0.08 | 0.10 | 15 / never | 16 / never | 0.152 (> delta: not stably reachable) | 0.073 (18) | 16-18, 28 | 16-18 | 140 | no / no |
| steel_yield | 21 | 0.10 | 0.05 | 13 / never | 16 / never | 0.052 (> delta: not stably reachable) | 0.035 (18) | 16-18, 27-28, 38 | 16-18 | 179 | no / no |
| steel_yield | 21 | 0.10 | 0.10 | 10 / 10 | 12 / 21 | 0.052 | 0.035 (18) | 12-18, 21-200 | 12-18 | 89 | no / no |
| polymer_Tg | 481 | 0.05 | 0.05 | 482 / 482 | 87 / 131 | 0.000 | 0.021 (198) | 87-88, 96-98, 105-108, 114-118, ... | 87-88, 96-98, 105-108, 114-118, ... | 719 | no / yes |
| polymer_Tg | 481 | 0.05 | 0.10 | 15 / 120 | 15 / 70 | 0.000 | 0.021 (198) | 15-18, 25-28, 35-38, 44-48, ... | 15-18, 25-28, 35-38, 44-48, ... | 359 | yes / yes |
| polymer_Tg | 481 | 0.08 | 0.05 | 60 / 60 | 16 / 50 | 0.000 | 0.002 (198) | 16-18, 25-28, 34-38, 42-48, ... | 16-18, 25-28, 34-38, 42-48, ... | 281 | yes / yes |
| polymer_Tg | 481 | 0.08 | 0.10 | 15 / 30 | 12 / 21 | 0.000 | 0.002 (198) | 12-18, 21-200 | 12-18, 21-482 | 140 | yes / yes |
| polymer_Tg | 481 | 0.10 | 0.05 | 15 / 40 | 14 / 30 | 0.000 | 0.000 (198) | 14-18, 23-28, 30-200 | 14-18, 23-28, 30-482 | 179 | yes / yes |
| polymer_Tg | 481 | 0.10 | 0.10 | 15 / 15 | 11 / 11 | 0.000 | 0.000 (198) | 11-200 | 11-482 | 89 | yes / yes |

'first' values below the 'stable' value come from the saw-tooth: at sizes where the conformal order statistic is the sample maximum (k = 9..18 at alpha = 0.10 gives j = k) or just below it, the interval is conservative (E[coverage] = j/(k+1) well above 0.90), so the failure probability dips; it rises again when j/(k+1) falls back towards 0.90. A budget read off such a dip is fragile.

## (d) alpha sensitivity (0.10 vs 0.20)

Minimum k for a finite interval: alpha 0.10 -> 9 (ceil(1/alpha)-1 = 9); alpha 0.20 -> 4 (formula 4).

| dataset | k | alpha 0.10: cov mean / p5 / nw median | alpha 0.20: cov mean / p5 / nw median | width ratio 0.20/0.10 (mean) |
|---|---|---|---|---|
| glass_Tg | 0 | 0.403 / 0.34 / 0.61 | 0.299 / 0.25 / 0.38 | 0.63 |
| glass_Tg | 4 | infinite | 0.827 / 0.58 / 2.24 | n/a |
| glass_Tg | 5 | infinite | 0.834 / 0.66 / 2.24 | n/a |
| glass_Tg | 9 | 0.894 / 0.71 / 2.77 | 0.807 / 0.57 / 2.19 | 0.79 |
| glass_Tg | 10 | 0.897 / 0.71 / 2.77 | 0.820 / 0.59 / 2.31 | 0.81 |
| glass_Tg | 15 | 0.935 / 0.83 / 3.01 | 0.817 / 0.60 / 2.29 | 0.75 |
| glass_Tg | 20 | 0.924 / 0.84 / 2.77 | 0.843 / 0.72 / 2.36 | 0.85 |
| glass_Tg | 30 | 0.930 / 0.84 / 2.72 | 0.837 / 0.75 / 2.28 | 0.84 |
| glass_Tg | 129 (pool) | 0.918 / 0.88 / 2.55 | 0.821 / 0.74 / 2.27 | 0.87 |
| glass_E | 0 | 0.230 / 0.12 / 0.77 | 0.102 / 0.06 / 0.42 | 0.56 |
| glass_E | 4 | infinite | 0.811 / 0.57 / 3.31 | n/a |
| glass_E | 5 | infinite | 0.856 / 0.58 / 3.44 | n/a |
| glass_E | 9 | 0.920 / 0.78 / 3.68 | 0.812 / 0.63 / 3.34 | 0.91 |
| glass_E | 10 | 0.930 / 0.84 / 3.74 | 0.817 / 0.63 / 3.36 | 0.91 |
| glass_E | 15 | 0.944 / 0.84 / 3.89 | 0.802 / 0.58 / 3.33 | 0.86 |
| glass_E | 20 | 0.905 / 0.75 / 3.52 | 0.797 / 0.63 / 3.32 | 0.92 |
| glass_E | 22 | 0.921 / 0.81 / 3.53 | 0.841 / 0.72 / 3.33 | 0.93 |
| glass_E | 30 | 0.911 / 0.81 / 3.60 | 0.798 / 0.63 / 3.27 | 0.92 |
| glass_E | 34 (pool) | 0.926 / 0.85 / 3.65 | 0.788 / 0.63 / 3.24 | 0.91 |
| glass_HV | 0 | 0.210 / 0.14 / 1.42 | 0.149 / 0.09 / 0.83 | 0.62 |
| glass_HV | 4 | infinite | 0.826 / 0.54 / 3.19 | n/a |
| glass_HV | 5 | infinite | 0.851 / 0.54 / 3.24 | n/a |
| glass_HV | 9 | 0.921 / 0.80 / 4.29 | 0.817 / 0.65 / 3.09 | 0.75 |
| glass_HV | 10 | 0.926 / 0.85 / 4.29 | 0.831 / 0.65 / 3.10 | 0.76 |
| glass_HV | 15 | 0.957 / 0.88 / 6.73 | 0.824 / 0.68 / 3.08 | 0.61 |
| glass_HV | 20 | 0.919 / 0.83 / 4.86 | 0.833 / 0.71 / 3.14 | 0.74 |
| glass_HV | 23 | 0.933 / 0.86 / 5.04 | 0.851 / 0.74 / 3.19 | 0.70 |
| glass_HV | 30 | 0.907 / 0.80 / 4.34 | 0.816 / 0.71 / 3.01 | 0.73 |
| glass_HV | 36 (pool) | 0.913 / 0.80 / 4.76 | 0.814 / 0.68 / 2.98 | 0.69 |
| glass_Tliq | 0 | 0.406 / 0.34 / 0.71 | 0.256 / 0.11 / 0.46 | 0.64 |
| glass_Tliq | 4 | infinite | 0.821 / 0.62 / 2.43 | n/a |
| glass_Tliq | 5 | infinite | 0.845 / 0.64 / 2.62 | n/a |
| glass_Tliq | 9 | 0.888 / 0.71 / 2.78 | 0.802 / 0.52 / 2.37 | 0.73 |
| glass_Tliq | 10 | 0.914 / 0.71 / 2.89 | 0.826 / 0.68 / 2.43 | 0.71 |
| glass_Tliq | 15 | 0.931 / 0.80 / 3.93 | 0.798 / 0.66 / 2.25 | 0.57 |
| glass_Tliq | 20 | 0.892 / 0.74 / 2.79 | 0.801 / 0.68 / 2.25 | 0.74 |
| glass_Tliq | 30 | 0.898 / 0.80 / 2.81 | 0.787 / 0.68 / 2.15 | 0.75 |
| glass_Tliq | 66 (pool) | 0.912 / 0.86 / 2.81 | 0.779 / 0.71 / 2.04 | 0.73 |
| steel_yield | 0 | 0.155 / 0.05 / 0.56 | 0.090 / 0.00 / 0.40 | 0.73 |
| steel_yield | 4 | infinite | 0.860 / 0.70 / 2.93 | n/a |
| steel_yield | 5 | infinite | 0.900 / 0.81 / 2.93 | n/a |
| steel_yield | 9 | 0.926 / 0.81 / 2.97 | 0.857 / 0.71 / 2.83 | 0.88 |
| steel_yield | 10 | 0.929 / 0.81 / 2.97 | 0.857 / 0.71 / 2.85 | 0.89 |
| steel_yield | 13 | 0.952 / 0.85 / 3.10 | 0.893 / 0.76 / 2.98 | 0.93 |
| steel_yield | 15 | 0.955 / 0.85 / 3.11 | 0.874 / 0.66 / 2.90 | 0.90 |
| steel_yield | 20 | 0.919 / 0.76 / 3.02 | 0.850 / 0.66 / 2.81 | 0.90 |
| steel_yield | 20 (pool) | 0.919 / 0.76 / 3.02 | 0.850 / 0.66 / 2.81 | 0.90 |
| polymer_Tg | 0 | 0.763 / 0.73 / 1.34 | 0.615 / 0.58 / 0.93 | 0.70 |
| polymer_Tg | 4 | infinite | 0.792 / 0.58 / 1.33 | n/a |
| polymer_Tg | 5 | infinite | 0.813 / 0.58 / 1.47 | n/a |
| polymer_Tg | 9 | 0.850 / 0.71 / 1.80 | 0.748 / 0.47 / 1.37 | 0.76 |
| polymer_Tg | 10 | 0.865 / 0.73 / 1.83 | 0.780 / 0.58 / 1.37 | 0.78 |
| polymer_Tg | 15 | 0.913 / 0.74 / 2.66 | 0.745 / 0.54 / 1.27 | 0.58 |
| polymer_Tg | 20 | 0.888 / 0.79 / 1.97 | 0.782 / 0.66 / 1.33 | 0.73 |
| polymer_Tg | 30 | 0.902 / 0.83 / 1.96 | 0.802 / 0.73 / 1.44 | 0.74 |
| polymer_Tg | 482 (pool) | 0.895 / 0.87 / 1.96 | 0.797 / 0.76 / 1.46 | 0.75 |

## (e) Retrospective replay over the whole target pool

Candidates = the whole target pool T. After n measurements (random = protocol order; novelty = decreasing mean distance to the 10 nearest standardised training inputs), recalibrate on the n measured records and evaluate on the |T| - n unmeasured ones (n <= |T| - 10; n < 9 gives an infinite interval and cannot qualify). C1 (task) = remainder coverage >= 0.90 in >= 80% of splits for every later n; C2 = the (b) criterion (5th percentile >= 0.80 and mean >= 0.88) for every later n. 'Every later n' runs up to a stopping horizon at which the remainder is still >= max(20, 20% of |T|) records (on smaller remainders one miss moves coverage by more than 0.05). **This horizon is an analyst convention and the stopping points depend on it strongly** (see the horizon-sensitivity table below; the strict variant runs to a remainder of 10). A stop must be verified over at least 10 further measurements before the horizon; otherwise a stop at the very end of the horizon would hold trivially over one or two points. Stopping points are *oracle* (they read remainder labels). 'unmeasured' = share of the target region that receives an interval instead of a measurement.

| dataset | |T| | order | C1 stop n (frac of T) | max share of splits >= 0.90 | C2 stop n (frac of T) | unmeasured at C2 | remainder cov at C2 (mean, p5) | nw median at C2 |
|---|---|---|---|---|---|---|---|---|
| glass_Tg | 258 | random | never | 0.90 | 17 (0.07) | 0.93 | 0.945, 0.81 | 3.01 |
| glass_Tg | 258 | novelty | 61 (0.24) | 1.00 | 9 (0.03) | 0.97 | 1.000, 1.00 | 5.08 |
| glass_E | 67 | random | never | 0.85 | never | n/a | n/a | n/a |
| glass_E | 67 | novelty | 32 (0.48) | 1.00 | 21 (0.31) | 0.69 | 0.883, 0.80 | 3.41 |
| glass_HV | 71 | random | never | 0.95 | never | n/a | n/a | n/a |
| glass_HV | 71 | novelty | 9 (0.13) | 1.00 | 9 (0.13) | 0.87 | 0.952, 0.92 | 5.16 |
| glass_Tliq | 131 | random | never | 0.90 | 35 (0.27) | 0.73 | 0.905, 0.82 | 2.81 |
| glass_Tliq | 131 | novelty | 21 (0.16) | 1.00 | 20 (0.15) | 0.85 | 0.883, 0.84 | 2.65 |
| steel_yield | 41 | random | never | 0.90 | never | n/a | n/a | n/a |
| steel_yield | 41 | novelty | 9 (0.22) | 1.00 | 9 (0.22) | 0.78 | 0.995, 0.97 | 3.23 |
| polymer_Tg | 963 | random | never | 0.80 | 22 (0.02) | 0.98 | 0.902, 0.82 | 2.00 |
| polymer_Tg | 963 | novelty | 9 (0.01) | 1.00 | 9 (0.01) | 0.99 | 0.986, 0.97 | 3.71 |

Horizon sensitivity of the oracle stopping points: 'stays from' n for C1 / C2 when 'every later n' runs up to a remainder of at least the given number of records (default = max(20, 20% of |T|), marked *; 10 = strict), each stop verified over >= 10 further measurements ('-' = horizon too short to verify any stop). A stopping point that moves or disappears as the horizon is extended rests on the convention.

| dataset | order | rem >= 10 | rem >= 20 | rem >= 27 | rem >= 30 | rem >= 50 | rem >= 52 | rem >= 193 | default (rem >= ...) |
|---|---|---|---|---|---|---|---|---|---|
| glass_Tg | random | never / never | never / never | - | never / 210 | never / 17 | never / 17* | - | 52 |
| glass_Tg | novelty | 61 / 9 | 61 / 9 | - | 61 / 9 | 61 / 9 | 61 / 9* | - | 52 |
| glass_E | random | never / never | never / never* | - | never / 21 | - | - | - | 20 |
| glass_E | novelty | 32 / 21 | 32 / 21* | - | never / 21 | - | - | - | 20 |
| glass_HV | random | never / never | never / never* | - | never / never | never / 10 | - | - | 20 |
| glass_HV | novelty | never / never | 9 / 9* | - | 9 / 9 | 9 / 9 | - | - | 20 |
| glass_Tliq | random | never / never | never / never | never / 35* | never / 35 | never / 35 | - | - | 27 |
| glass_Tliq | novelty | never / 20 | 21 / 20 | 21 / 20* | 21 / 20 | 21 / 20 | - | - | 27 |
| steel_yield | random | never / never | never / never* | - | - | - | - | - | 20 |
| steel_yield | novelty | 9 / 9 | 9 / 9* | - | - | - | - | - | 20 |
| polymer_Tg | random | never / never | never / never | - | never / never | never / 892 | - | never / 22* | 193 |
| polymer_Tg | novelty | 9 / 9 | 9 / 9 | - | 9 / 9 | 9 / 9 | - | 9 / 9* | 193 |

Composition straddle at the replay cut. The replay cuts between measured and unmeasured at record level; replicate records are contiguous in both orders, so at most one composition per cut can have records on both sides. Shown: remainder records sharing a composition with a measured record, mean over n >= 9 and splits (max), and the largest share of the remainder they ever make up.

| dataset | random: mean (max), max share | novelty: mean (max), max share |
|---|---|---|
| glass_Tg | 0.03 (2), 0.111 | 0.04 (2), 0.154 |
| glass_E | 0.00 (0), 0.000 | 0.00 (0), 0.000 |
| glass_HV | 0.00 (0), 0.000 | 0.00 (0), 0.000 |
| glass_Tliq | 0.05 (1), 0.091 | 0.04 (1), 0.100 |
| steel_yield | 0.00 (0), 0.000 | 0.00 (0), 0.000 |
| polymer_Tg | 0.39 (5), 0.417 | 0.40 (5), 0.111 |

C1 at the C1 stopping point (where reached): median normalised width the unmeasured candidates receive.

| dataset | order | C1 stop n | frac of T | remainder cov mean | share of splits >= 0.90 | nw median [2.5, 97.5] |
|---|---|---|---|---|---|---|
| glass_Tg | random | never | - | - | - | - |
| glass_Tg | novelty | 61 | 0.24 | 0.923 | 0.85 | 2.68 [2.29, 2.96] |
| glass_E | random | never | - | - | - | - |
| glass_E | novelty | 32 | 0.48 | 0.940 | 0.80 | 3.63 [3.18, 4.30] |
| glass_HV | random | never | - | - | - | - |
| glass_HV | novelty | 9 | 0.13 | 0.952 | 1.00 | 5.16 [4.06, 6.54] |
| glass_Tliq | random | never | - | - | - | - |
| glass_Tliq | novelty | 21 | 0.16 | 0.932 | 0.90 | 3.07 [2.80, 3.82] |
| steel_yield | random | never | - | - | - | - |
| steel_yield | novelty | 9 | 0.22 | 0.995 | 1.00 | 3.23 [2.90, 3.44] |
| polymer_Tg | random | never | - | - | - | - |
| polymer_Tg | novelty | 9 | 0.01 | 0.986 | 1.00 | 3.71 [2.73, 4.27] |

Why C1 is not reached under random order: split conformal guarantees the *mean* coverage, not coverage >= 0.90 in 80% of realisations. Under exchangeability P(remainder coverage >= 0.90) has a median over n of 0.55-0.72 and exceeds 0.8 only at a few conservative saw-tooth sizes where the conformal order statistic is the sample maximum (n <= 18), so 'for every later n' cannot hold:

| dataset | theory P(>= 0.90), random order: median / max over n (at n) | observed max share of splits (random) | observed max share (novelty) | Spearman(novelty, |residual|) |
|---|---|---|---|---|
| glass_Tg | 0.59 / 0.85 (18) | 0.90 | 1.00 | 0.57 [0.51, 0.65] |
| glass_E | 0.68 / 0.84 (17) | 0.85 | 1.00 | 0.46 [0.36, 0.55] |
| glass_HV | 0.67 / 0.84 (18) | 0.95 | 1.00 | 0.49 [0.34, 0.66] |
| glass_Tliq | 0.62 / 0.84 (18) | 0.90 | 1.00 | 0.11 [0.01, 0.19] |
| steel_yield | 0.72 / 0.83 (18) | 0.90 | 1.00 | 0.54 [0.37, 0.69] |
| polymer_Tg | 0.55 / 0.85 (18) | 0.80 | 1.00 | 0.26 [0.19, 0.33] |

Label-free, pre-registered stopping rules (protocol-compliant; for novelty order the exchangeability assumption behind the Beta rule does not hold):

| dataset | rule (n) | order | frac of T measured | remainder cov mean [2.5, 97.5] | p5 | share >= 0.90 | nw median |
|---|---|---|---|---|---|---|---|
| glass_Tg | headline_m (30) | random | 0.12 | 0.923 [0.80, 0.98] | 0.82 | 0.80 | 2.72 |
| glass_Tg | headline_m (30) | novelty | 0.12 | 0.922 [0.83, 0.97] | 0.84 | 0.80 | 2.65 |
| glass_Tg | exact_beta_eps0.08_delta0.10 (21) | random | 0.08 | 0.919 [0.80, 0.99] | 0.81 | 0.70 | 2.77 |
| glass_Tg | exact_beta_eps0.08_delta0.10 (21) | novelty | 0.08 | 0.952 [0.92, 0.99] | 0.92 | 1.00 | 2.88 |
| glass_Tg | exact_beta_eps0.05_delta0.10 (60) | random | 0.23 | 0.918 [0.84, 0.96] | 0.85 | 0.80 | 2.63 |
| glass_Tg | exact_beta_eps0.05_delta0.10 (60) | novelty | 0.23 | 0.920 [0.87, 0.96] | 0.88 | 0.70 | 2.68 |
| glass_Tg | chebyshev_eps0.08_delta0.10 (140) | random | 0.54 | 0.913 [0.86, 0.97] | 0.86 | 0.65 | 2.54 |
| glass_Tg | chebyshev_eps0.08_delta0.10 (140) | novelty | 0.54 | 0.992 [0.99, 0.99] | 0.99 | 1.00 | 2.83 |
| glass_E | headline_m (22) | random | 0.33 | 0.920 [0.78, 1.00] | 0.82 | 0.70 | 3.53 |
| glass_E | headline_m (22) | novelty | 0.33 | 0.896 [0.81, 1.00] | 0.82 | 0.40 | 3.45 |
| glass_E | exact_beta_eps0.08_delta0.10 (21) | random | 0.31 | 0.911 [0.77, 1.00] | 0.80 | 0.65 | 3.52 |
| glass_E | exact_beta_eps0.08_delta0.10 (21) | novelty | 0.31 | 0.883 [0.79, 1.00] | 0.80 | 0.35 | 3.41 |
| glass_E | exact_beta_eps0.05_delta0.10 (60) | random | infeasible (> |T| - 10) | - | - | - | - |
| glass_E | exact_beta_eps0.05_delta0.10 (60) | novelty | infeasible (> |T| - 10) | - | - | - | - |
| glass_E | chebyshev_eps0.08_delta0.10 (140) | random | infeasible (> |T| - 10) | - | - | - | - |
| glass_E | chebyshev_eps0.08_delta0.10 (140) | novelty | infeasible (> |T| - 10) | - | - | - | - |
| glass_HV | headline_m (23) | random | 0.32 | 0.938 [0.83, 1.00] | 0.83 | 0.75 | 5.04 |
| glass_HV | headline_m (23) | novelty | 0.32 | 0.958 [0.96, 0.96] | 0.96 | 1.00 | 5.01 |
| glass_HV | exact_beta_eps0.08_delta0.10 (21) | random | 0.30 | 0.933 [0.83, 1.00] | 0.84 | 0.70 | 4.90 |
| glass_HV | exact_beta_eps0.08_delta0.10 (21) | novelty | 0.30 | 0.960 [0.96, 0.96] | 0.96 | 1.00 | 5.01 |
| glass_HV | exact_beta_eps0.05_delta0.10 (60) | random | 0.85 | 0.914 [0.82, 1.00] | 0.82 | 0.75 | 3.62 |
| glass_HV | exact_beta_eps0.05_delta0.10 (60) | novelty | 0.85 | 0.791 [0.73, 0.82] | 0.73 | 0.00 | 3.18 |
| glass_HV | chebyshev_eps0.08_delta0.10 (140) | random | infeasible (> |T| - 10) | - | - | - | - |
| glass_HV | chebyshev_eps0.08_delta0.10 (140) | novelty | infeasible (> |T| - 10) | - | - | - | - |
| glass_Tliq | headline_m (30) | random | 0.23 | 0.897 [0.73, 0.96] | 0.77 | 0.70 | 2.81 |
| glass_Tliq | headline_m (30) | novelty | 0.23 | 1.000 [1.00, 1.00] | 1.00 | 1.00 | 6.79 |
| glass_Tliq | exact_beta_eps0.08_delta0.10 (21) | random | 0.16 | 0.899 [0.73, 0.98] | 0.74 | 0.70 | 2.80 |
| glass_Tliq | exact_beta_eps0.08_delta0.10 (21) | novelty | 0.16 | 0.932 [0.88, 0.96] | 0.89 | 0.90 | 3.07 |
| glass_Tliq | exact_beta_eps0.05_delta0.10 (60) | random | 0.46 | 0.906 [0.83, 0.98] | 0.84 | 0.65 | 2.81 |
| glass_Tliq | exact_beta_eps0.05_delta0.10 (60) | novelty | 0.46 | 1.000 [1.00, 1.00] | 1.00 | 1.00 | 4.22 |
| glass_Tliq | chebyshev_eps0.08_delta0.10 (140) | random | infeasible (> |T| - 10) | - | - | - | - |
| glass_Tliq | chebyshev_eps0.08_delta0.10 (140) | novelty | infeasible (> |T| - 10) | - | - | - | - |
| steel_yield | headline_m (13) | random | 0.32 | 0.955 [0.87, 1.00] | 0.89 | 0.85 | 3.10 |
| steel_yield | headline_m (13) | novelty | 0.32 | 0.995 [0.96, 1.00] | 0.96 | 1.00 | 3.23 |
| steel_yield | exact_beta_eps0.08_delta0.10 (21) | random | 0.51 | 0.922 [0.77, 1.00] | 0.80 | 0.70 | 3.02 |
| steel_yield | exact_beta_eps0.08_delta0.10 (21) | novelty | 0.51 | 1.000 [1.00, 1.00] | 1.00 | 1.00 | 3.16 |
| steel_yield | exact_beta_eps0.05_delta0.10 (60) | random | infeasible (> |T| - 10) | - | - | - | - |
| steel_yield | exact_beta_eps0.05_delta0.10 (60) | novelty | infeasible (> |T| - 10) | - | - | - | - |
| steel_yield | chebyshev_eps0.08_delta0.10 (140) | random | infeasible (> |T| - 10) | - | - | - | - |
| steel_yield | chebyshev_eps0.08_delta0.10 (140) | novelty | infeasible (> |T| - 10) | - | - | - | - |
| polymer_Tg | headline_m (30) | random | 0.03 | 0.903 [0.81, 0.98] | 0.84 | 0.60 | 1.96 |
| polymer_Tg | headline_m (30) | novelty | 0.03 | 0.982 [0.96, 1.00] | 0.97 | 1.00 | 3.28 |
| polymer_Tg | exact_beta_eps0.08_delta0.10 (21) | random | 0.02 | 0.890 [0.78, 0.97] | 0.79 | 0.55 | 1.97 |
| polymer_Tg | exact_beta_eps0.08_delta0.10 (21) | novelty | 0.02 | 0.985 [0.97, 0.99] | 0.97 | 1.00 | 3.49 |
| polymer_Tg | exact_beta_eps0.05_delta0.10 (60) | random | 0.06 | 0.905 [0.85, 0.97] | 0.85 | 0.50 | 1.96 |
| polymer_Tg | exact_beta_eps0.05_delta0.10 (60) | novelty | 0.06 | 0.988 [0.98, 1.00] | 0.98 | 1.00 | 3.34 |
| polymer_Tg | chebyshev_eps0.08_delta0.10 (140) | random | 0.15 | 0.911 [0.88, 0.96] | 0.88 | 0.65 | 1.96 |
| polymer_Tg | chebyshev_eps0.08_delta0.10 (140) | novelty | 0.15 | 0.980 [0.97, 0.99] | 0.97 | 1.00 | 2.81 |

Novelty vs random order at the same n (remainder coverage mean and median normalised width):

| dataset | n | random cov / nw | novelty cov / nw | novelty/random width (mean) |
|---|---|---|---|---|
| glass_Tg | 9 | 0.888 / 2.77 | 1.000 / 5.08 | 1.92 |
| glass_Tg | 15 | 0.928 / 3.01 | 1.000 / 5.08 | 1.74 |
| glass_Tg | 17 | 0.945 / 3.01 | 1.000 / 5.08 | 1.68 |
| glass_Tg | 30 | 0.923 / 2.72 | 0.922 / 2.65 | 1.00 |
| glass_Tg | 60 | 0.918 / 2.63 | 0.920 / 2.68 | 1.00 |
| glass_Tg | 61 | 0.921 / 2.64 | 0.923 / 2.68 | 1.00 |
| glass_Tg | 120 | 0.918 / 2.57 | 0.963 / 2.75 | 1.06 |
| glass_E | 9 | 0.919 / 3.68 | 0.771 / 3.28 | 0.90 |
| glass_E | 15 | 0.949 / 3.89 | 0.844 / 3.36 | 0.91 |
| glass_E | 21 | 0.911 / 3.52 | 0.883 / 3.41 | 0.96 |
| glass_E | 22 | 0.920 / 3.53 | 0.896 / 3.45 | 0.97 |
| glass_E | 30 | 0.914 / 3.60 | 0.909 / 3.48 | 0.98 |
| glass_E | 32 | 0.920 / 3.65 | 0.940 / 3.63 | 0.99 |
| glass_HV | 9 | 0.917 / 4.29 | 0.952 / 5.16 | 1.24 |
| glass_HV | 15 | 0.954 / 6.73 | 0.963 / 5.39 | 1.06 |
| glass_HV | 23 | 0.938 / 5.04 | 0.958 / 5.01 | 1.10 |
| glass_HV | 30 | 0.910 / 4.34 | 0.950 / 4.75 | 1.15 |
| glass_HV | 60 | 0.914 / 3.62 | 0.791 / 3.18 | 0.87 |
| glass_Tliq | 9 | 0.888 / 2.78 | 0.930 / 3.07 | 1.09 |
| glass_Tliq | 15 | 0.927 / 3.93 | 0.926 / 3.07 | 0.86 |
| glass_Tliq | 20 | 0.891 / 2.79 | 0.883 / 2.65 | 0.96 |
| glass_Tliq | 21 | 0.899 / 2.80 | 0.932 / 3.07 | 1.10 |
| glass_Tliq | 30 | 0.897 / 2.81 | 1.000 / 6.79 | 2.49 |
| glass_Tliq | 35 | 0.905 / 2.81 | 1.000 / 6.79 | 2.39 |
| glass_Tliq | 60 | 0.906 / 2.81 | 1.000 / 4.22 | 1.53 |
| glass_Tliq | 120 | 0.923 / 2.80 | 0.977 / 2.81 | 1.01 |
| steel_yield | 9 | 0.923 / 2.97 | 0.995 / 3.23 | 1.08 |
| steel_yield | 13 | 0.955 / 3.10 | 0.995 / 3.23 | 1.03 |
| steel_yield | 15 | 0.962 / 3.11 | 0.994 / 3.23 | 1.03 |
| steel_yield | 30 | 0.918 / 3.01 | 1.000 / 3.02 | 1.02 |
| polymer_Tg | 9 | 0.851 / 1.80 | 0.986 / 3.71 | 2.06 |
| polymer_Tg | 15 | 0.915 / 2.66 | 0.987 / 3.72 | 1.57 |
| polymer_Tg | 22 | 0.902 / 2.00 | 0.985 / 3.49 | 1.66 |
| polymer_Tg | 30 | 0.903 / 1.96 | 0.982 / 3.28 | 1.58 |
| polymer_Tg | 60 | 0.905 / 1.96 | 0.988 / 3.34 | 1.62 |
| polymer_Tg | 120 | 0.913 / 1.98 | 0.980 / 2.89 | 1.39 |

## Re-computing the submitted '65% and 90% summed across six cases'

* Total target-region size across the six cases: 1531 records; the polymer contributes 63% of it, so any summed saving is dominated by one dataset.
* As coded in demo/exp4 the 'rule' budget is min(140, max(3, |T|-15)), capped again at |T|-5: {'glass_Tg': 140, 'glass_E': 52, 'glass_HV': 56, 'glass_Tliq': 116, 'steel_yield': 26, 'polymer_Tg': 140}. It equals the Chebyshev m = 140 only where |T| >= 155; it was truncated in 4/6 cases (glass_E, glass_HV, glass_Tliq, steel_yield). Summed saving as coded: rule 65.4%, heuristic 90.3%.
* Under the protocol the recalibration draws must come from the recalibration pool (half of T). The Chebyshev m = 140 fits inside the pool only for ['polymer_Tg'] and inside the whole-T replay only for ['glass_Tg', 'polymer_Tg']; the '65%' therefore does not describe the rule. Capping the rule at the pool size gives 72.2% summed, but that is again not the rule.
* The heuristic's 'saving' is arithmetic in |T| (1 - sum(m)/sum|T|) = 90.3% summed, 77.6% as a per-dataset mean (glass_Tg 88%, glass_E 67%, glass_HV 68%, glass_Tliq 77%, steel_yield 68%, polymer_Tg 97%); 79.2% summed without the polymer. It is not an experimental result; what is empirical is whether coverage holds at that m:
  * glass_Tg: at m = 30 (12% of |T|). Target-test half: mean 0.930, 2.5% 0.84, min 0.84, share of splits >= 0.90 = 0.80. Replay remainder (random order, the 88% unmeasured): mean 0.923, min 0.79, share >= 0.90 = 0.80, median nw 2.72.
  * glass_E: at m = 22 (33% of |T|). Target-test half: mean 0.921, 2.5% 0.75, min 0.70, share of splits >= 0.90 = 0.75. Replay remainder (random order, the 67% unmeasured): mean 0.920, min 0.73, share >= 0.90 = 0.70, median nw 3.53.
  * glass_HV: at m = 23 (32% of |T|). Target-test half: mean 0.933, 2.5% 0.84, min 0.83, share of splits >= 0.90 = 0.70. Replay remainder (random order, the 68% unmeasured): mean 0.938, min 0.83, share >= 0.90 = 0.75, median nw 5.04.
  * glass_Tliq: at m = 30 (23% of |T|). Target-test half: mean 0.898, 2.5% 0.75, min 0.71, share of splits >= 0.90 = 0.60. Replay remainder (random order, the 77% unmeasured): mean 0.897, min 0.69, share >= 0.90 = 0.70, median nw 2.81.
  * steel_yield: at m = 13 (32% of |T|). Target-test half: mean 0.952, 2.5% 0.83, min 0.81, share of splits >= 0.90 = 0.90. Replay remainder (random order, the 68% unmeasured): mean 0.955, min 0.86, share >= 0.90 = 0.85, median nw 3.10.
  * polymer_Tg: at m = 30 (3% of |T|). Target-test half: mean 0.902, 2.5% 0.82, min 0.82, share of splits >= 0.90 = 0.55. Replay remainder (random order, the 97% unmeasured): mean 0.903, min 0.79, share >= 0.90 = 0.60, median nw 1.96.
* Two readings of 'stays at or above the nominal 0.90 throughout (0.91 to 0.97)': as a mean over re-partitions (how the original numbers were computed) it essentially holds under the protocol (see the means above; the original values were partly inflated by the one-rank-too-high quantile); per realisation it does not hold (see the minima and shares above). Split conformal promises only the former.
* The exact Beta rule for (eps 0.08, delta 0.10) is n = 21 (stable); it fits inside the recalibration pool for ['glass_Tg', 'glass_E', 'glass_HV', 'glass_Tliq', 'polymer_Tg'].
* Replay (oracle), random order, C1_task: stop n (default horizon) = {'glass_Tg': None, 'glass_E': None, 'glass_HV': None, 'glass_Tliq': None, 'steel_yield': None, 'polymer_Tg': None}; unmeasured share: reached in 0/6 datasets only. Strict horizon (remainder >= 10): {'glass_Tg': None, 'glass_E': None, 'glass_HV': None, 'glass_Tliq': None, 'steel_yield': None, 'polymer_Tg': None}.
* Replay (oracle), random order, C2_b_analogue: stop n (default horizon) = {'glass_Tg': 17, 'glass_E': None, 'glass_HV': None, 'glass_Tliq': 35, 'steel_yield': None, 'polymer_Tg': 22}; unmeasured share: reached in 3/6 datasets only. Strict horizon (remainder >= 10): {'glass_Tg': None, 'glass_E': None, 'glass_HV': None, 'glass_Tliq': None, 'steel_yield': None, 'polymer_Tg': None}.
* Replay (oracle), novelty order, C1_task: stop n (default horizon) = {'glass_Tg': 61, 'glass_E': 32, 'glass_HV': 9, 'glass_Tliq': 21, 'steel_yield': 9, 'polymer_Tg': 9}; unmeasured share: 90.8% summed, 79.5% per-dataset mean. Strict horizon (remainder >= 10): {'glass_Tg': 61, 'glass_E': 32, 'glass_HV': None, 'glass_Tliq': None, 'steel_yield': 9, 'polymer_Tg': 9}.
* Replay (oracle), novelty order, C2_b_analogue: stop n (default horizon) = {'glass_Tg': 9, 'glass_E': 21, 'glass_HV': 9, 'glass_Tliq': 20, 'steel_yield': 9, 'polymer_Tg': 9}; unmeasured share: 95.0% summed, 85.7% per-dataset mean. Strict horizon (remainder >= 10): {'glass_Tg': 9, 'glass_E': 21, 'glass_HV': None, 'glass_Tliq': 20, 'steel_yield': 9, 'polymer_Tg': 9}.

## Original-code quantile check (demo/exp4, demo/exp6)

The demo scripts compute the conformal quantile as np.quantile(r, min(ceil((n+1)(1-a))/n, 1), method='higher'). Order statistic returned vs the finite-sample conformal one (common.conformal_q):

| n | original rank | conformal rank | E[coverage] original | E[coverage] conformal |
|---|---|---|---|---|
| 3 | 3 | inf | 0.750 | 1.000 |
| 5 | 5 | inf | 0.833 | 1.000 |
| 8 | 8 | inf | 0.889 | 1.000 |
| 9 | 9 | 9 | 0.900 | 0.900 |
| 10 | 10 | 10 | 0.909 | 0.909 |
| 13 | 13 | 13 | 0.929 | 0.929 |
| 15 | 15 | 15 | 0.938 | 0.938 |
| 20 | 20 | 19 | 0.952 | 0.905 |
| 22 | 22 | 21 | 0.957 | 0.913 |
| 23 | 23 | 22 | 0.958 | 0.917 |
| 26 | 26 | 25 | 0.963 | 0.926 |
| 30 | 29 | 28 | 0.935 | 0.903 |
| 40 | 38 | 37 | 0.927 | 0.902 |
| 52 | 49 | 48 | 0.925 | 0.906 |
| 56 | 53 | 52 | 0.930 | 0.912 |
| 116 | 107 | 106 | 0.915 | 0.906 |
| 140 | 128 | 127 | 0.908 | 0.901 |

## Interpretation

* Source-calibrated intervals (k = 0) cover 0.15-0.76 of the target-test half (mean over splits); every finite k >= 9 gives split-mean coverage 0.85-0.96 (nominal 0.90), so the question is how variable the realised coverage is, and at what width.
* Because the recalibration pool and the target-test half are a random (grouped) partition of the same target pool, calibration and test residuals are exchangeable by construction, and realised coverage follows the exact Beta-Binomial law: across datasets and k >= 9 the 5th percentile of the 200-subset resampling differs from the theoretical one by a median +0.000 (range -0.03 to +0.06); the 20-split first-k percentiles scatter more (median +0.031, range -0.08 to +0.14) because 20 nested draws estimate a 5th percentile poorly. The agreement is guaranteed by the design, not an empirical property of the materials: under a random partition the coverage budget is a function of k and n_test only. It does not establish exchangeability for prospective, non-random acquisitions (the novelty-order replay in (e) breaks it), and the width cost, i.e. the usefulness of the interval, is material-dependent (median normalised width at the headline m 2.0-5.0 x the target-property IQR).
* At the headline m (13-30 target records) mean target-test coverage is 0.898-0.952, but single splits fall to 0.70 and only 0.55-0.90 of splits reach 0.90; the median normalised width is 2.0-5.0 x the target-property IQR.
* Protocol caveat (polymer_Tg): replicate records of one composition stay contiguous in the grouped permutation, so the first k = 9 pool records contain on average 6.7 distinct compositions; first-k mean coverage at k = 9/10 is 0.850/0.865 versus 0.895/0.904 for record-level random subsets of the same pool (theory 0.900/0.909). Small recalibration budgets should be counted in distinct compositions when replicates exist.
* (b, oracle) Smallest grid k with 5th-percentile coverage >= 0.8 and mean >= 0.88: glass_Tg 15 = 6% of |T|; glass_E 10 (stable 21) = 15% of |T|; glass_HV 9 (stable never) = 13% of |T|; glass_Tliq 40 = 31% of |T|; steel_yield 9 (stable never) = 22% of |T|; polymer_Tg 30 = 3% of |T|. For the small target regions these first/stable labels are noise-level: with 20 splits the 5th percentile is essentially the second-lowest split, and several verdicts rest on near misses (glass_HV p5 = 0.7986 at k = 30; glass_HV p5 = 0.7986 at k = 36 (pool); glass_Tliq p5 = 0.7954 at k = 15; glass_Tliq p5 = 0.7954 at k = 30; polymer_Tg p5 = 0.7923 at k = 20; polymer_Tg p5 = 0.7923 at k = 21). The reliable reference is exchangeable theory for the actual target-test size over all integer k (Beta-Binomial 5th percentile >= 0.80; first / stable over k <= 3000; stable within the pool): glass_Tg (n_test 129) 15 / 40; within pool (129) 40; glass_E (n_test 33) 16 / 310; within pool (34) never; glass_HV (n_test 35) 15 / 60; within pool (36) 34; glass_Tliq (n_test 65) 14 / 40; within pool (66) 40; steel_yield (n_test 21) 16 / never; within pool (20) never; polymer_Tg (n_test 481) 14 / 30; within pool (482) 30. With an infinite test set the Beta law needs n = 14 (stably 30). Never stable at any k: steel_yield (the Binomial(n_test, 0.9) 5th percentile, the limit as k grows, is below 0.80, so the target-test half is too small for the criterion). Stable only beyond the recalibration pool: glass_E (from k = 310, pool 34).
* (b, protocol-compliant) Choosing k from the recalibration pool alone (2000 grouped inner splits: the first k records of a grouped permutation calibrate, the remaining whole compositions are held out) certifies a budget in glass_Tg 20/20 (range 17-20 over 5 inner-RNG replicates), glass_E 0/20 (range 0-0 over 5 inner-RNG replicates), glass_HV 6/20 (range 3-11 over 5 inner-RNG replicates), glass_Tliq 16/20 (range 14-16 over 5 inner-RNG replicates), steel_yield 0/20 (range 0-0 over 5 inner-RNG replicates), polymer_Tg 20/20 (range 20-20 over 5 inner-RNG replicates) splits. The counts are approximate: the criterion is a knife edge on a coarse lattice (steps of 1/(pool - k) of the inner holdout), and 34 of the 120 split-level certify / not-certify decisions and 53 chosen k values change between replicates. The inner holdout (pool - k records) is smaller than the target-test half, so the inner 5th percentile is more pessimistic than the one it predicts. Chosen k (primary replicate, all certified splits): {15: 42, 40: 4, 60: 16}. Target-test coverage at the chosen k: glass_Tg mean 0.937 (p5 0.84, min 0.70), glass_HV mean 0.943 (p5 0.89, min 0.89), glass_Tliq mean 0.918 (p5 0.78, min 0.71), polymer_Tg mean 0.902 (p5 0.84, min 0.82). glass_E, steel_yield cannot certify any budget from their own pool in any replicate. Certification inside the pool does not transfer to a single target-test realisation: at the chosen k the target-test 5th percentile is glass_Tg 0.84, glass_HV 0.89, glass_Tliq 0.78, polymer_Tg 0.84 and the worst split glass_Tg 0.70, glass_HV 0.89, glass_Tliq 0.71, polymer_Tg 0.82, i.e. below the 0.80 that was certified for glass_Tliq. The inner criterion describes the distribution over inner holdouts; it is not a guarantee for one evaluation half, which is itself a finite sample.
* Record-level vs grouped inner splits. Where the pool contains replicate records, glass_Tliq (pool 66 records, 62 compositions): record-level chose k = 15x20 with target-test p5 0.80; grouped chooses k = 15x16 with target-test p5 0.78; polymer_Tg (pool 482 records, 362 compositions): record-level chose k = 15x20 with target-test p5 0.74; grouped chooses k = 15x2, 40x3, 60x15 with target-test p5 0.84. With record-level inner splits, replicates of one composition sit on both sides of the inner split, so the pool looks easier than the deployed first-k rule, which calibrates on fewer distinct compositions. For polymer_Tg this mismatch explains why the previous version certified k = 15 in every split and then saw a target-test 5th percentile below the 0.80 it had certified. glass_Tg, glass_E, glass_HV, steel_yield have (almost) no replicates, so the two versions are the same procedure and differ only by Monte Carlo noise (glass_Tg record-level 20/20 vs grouped 20/20 (replicates 17-20); glass_E record-level 0/20 vs grouped 0/20 (replicates 0-0); glass_HV record-level 1/20 vs grouped 6/20 (replicates 3-11); steel_yield record-level 0/20 vs grouped 0/20 (replicates 0-0)), another sign that the certification counts are knife-edge.
* (c) The submitted Chebyshev rule is valid but 6.0-8.1x more demanding than the exact Beta rule (stable n) at alpha = 0.10; for the manuscript's eps = 0.08, delta = 0.10 it asks for 140 records where 21 suffice (exact failure probability at 140: 0.003). The manuscript's 'knee at about 13' matches the first exact n (12), which lies on a conservative saw-tooth dip; the stable exact value is 21. 140 fits inside the recalibration pool only for polymer_Tg.
* With a finite target-test half, the exact finite-test (Beta-Binomial) failure probability tends, as k grows, to the Binomial(n_test, 0.9) tail; at eps = 0.08 this limit is glass_Tg 0.002, glass_E 0.106, glass_HV 0.055, glass_Tliq 0.026, steel_yield 0.152, polymer_Tg 0.000. It is a limit, not a floor. Where it exceeds delta = 0.10 (glass_E, steel_yield), no budget meets delta stably, but delta is still met at conservative saw-tooth sizes where the interval is the sample maximum or close to it (glass_E (n_test 33): limit 0.106, delta met at k = 15-18, 26-28, 37-38, 48, ..., minimum 0.061 at k = 18; within the 34-record pool: 15-18, 26-28; steel_yield (n_test 21): limit 0.152, delta met at k = 16-18, 28, minimum 0.073 at k = 18; within the 20-record pool: 16-18). A budget read off such a dip is fragile: adding one record can break it. Small design regions therefore cannot support a stable (eps 0.08, delta 0.10) guarantee on their own evaluation set.
* (d) alpha = 0.20 gives a finite interval from k = 4 (vs 9 at alpha = 0.10). At the headline m the mean over splits of the per-split width ratio (alpha 0.20 / alpha 0.10) is glass_Tg 0.84, glass_E 0.93, glass_HV 0.70, glass_Tliq 0.75, steel_yield 0.93, polymer_Tg 0.74, with mean coverage 0.79-0.89 (5th percentile down to 0.68); at k = 4 the 5th percentile is 0.54-0.70. A looser alpha buys fewer required records and narrower intervals at the price of a weaker, noisier guarantee.
* (e) Random-order replay: the task criterion C1 (remainder coverage >= 0.90 in >= 80% of splits for every later n, verified over >= 10 further measurements) is reached in 0/6 datasets under the default horizon and under every other horizon tested. Exchangeable theory predicts this. The weaker C2 criterion is reached, under the default horizon convention (remainder >= max(20, 20% of |T|)), at glass_Tg 17, glass_E never, glass_HV never, glass_Tliq 35, steel_yield never, polymer_Tg 22; with the strict convention (remainder down to 10) these become glass_Tg never, glass_E never, glass_HV never, glass_Tliq never, steel_yield never, polymer_Tg never. The random-order C2 stopping points therefore depend on an analyst-chosen horizon and must not be quoted as a share of |T| 'saved' without that caveat (default-horizon values: glass_Tg 7% of |T|, glass_Tliq 27% of |T|, polymer_Tg 2% of |T|).
* Novelty-order replay reaches C1 (default horizon) at glass_Tg 61 (24%), glass_E 32 (48%), glass_HV 9 (13%), glass_Tliq 21 (16%), steel_yield 9 (22%), polymer_Tg 9 (1%) measurements. The mechanism is non-exchangeability in the favourable direction: the novel candidates measured first tend to have larger errors than the familiar ones left unmeasured (mean |residual| of the remainder falls as measurement proceeds), so a quantile calibrated on the former over-covers the latter. Sometimes this costs width: novelty/random width ratio at the C1 stopping n is glass_Tg 1.00, glass_E 0.99, glass_HV 1.24, glass_Tliq 1.10, steel_yield 1.08, polymer_Tg 2.06; the largest ratio among the compared n is glass_Tg 1.92 (n = 9), glass_E 0.99 (n = 32), glass_HV 1.24 (n = 9), glass_Tliq 2.49 (n = 30), steel_yield 1.08 (n = 9), polymer_Tg 2.06 (n = 9). It is not a guarantee: whether it over- or under-covers depends on how novelty relates to error in each dataset.
* Horizon dependence of the oracle stopping points (default remainder >= max(20, 20% of |T|) -> strict remainder >= 10): glass_Tg random C2 17 -> never; glass_HV novelty C1 9 -> never; glass_HV novelty C2 9 -> never; glass_Tliq random C2 35 -> never; glass_Tliq novelty C1 21 -> never; polymer_Tg random C2 22 -> never. Full grid in the horizon-sensitivity table. For novelty order a change means that coverage on the last, most familiar candidates breaks the criterion after the default horizon.
* Novelty-order under-coverage observed: glass_E mean remainder coverage 0.77 at n = 9; glass_HV mean remainder coverage 0.77 at n = 61; glass_Tliq mean remainder coverage 0.88 at n = 19.
* Spearman correlation of the novelty score with |residual| on the target pool: glass_Tg 0.57, glass_E 0.46, glass_HV 0.49, glass_Tliq 0.11, steel_yield 0.54, polymer_Tg 0.26.
* Grouping in the replay: the cut between measured and unmeasured records is at record level (groups are contiguous in both orders), so a remainder record can share a composition with a measured one. Mean over n and splits (max): glass_Tg 0.04 (2), glass_E 0.00 (0), glass_HV 0.00 (0), glass_Tliq 0.05 (1), steel_yield 0.00 (0), polymer_Tg 0.40 (5); at the oracle stopping points the mean is at most 0.40 records. The effect on remainder coverage is negligible, but strictly the replay treats replicate records as separate candidates.
* '65% and 90% summed across six cases': as coded in demo/exp4 these are 65.4% (rule) and 90.3% (heuristic). Both are arithmetic in |T| rather than experimental outcomes; the rule was truncated to |T|-15 in 4/6 cases (so the 65% is not the rule), and the summed figure is dominated by polymer_Tg (63% of all target records). Under the protocol the Chebyshev rule is executable in 1/6 datasets; the heuristic's 90.3% summed (77.6% per-dataset mean, 67-97% per dataset) is unchanged arithmetically.
* The accompanying statement that coverage on the remaining measurements 'stays at or above the nominal 0.90 throughout (0.91 to 0.97)' has two readings. (i) As a mean over re-partitions, which is how the original figures were computed: under the protocol, mean remainder coverage in the random-order replay at the heuristic m is 0.897-0.955 and mean target-test coverage at m is 0.898-0.952, so the mean reading essentially holds (glass_Tliq marginally below 0.90: replay glass_Tliq 0.897, target-test glass_Tliq 0.898); the original 0.91-0.97 were partly inflated by the one-rank-too-high quantile. (ii) Per realisation it does not hold: 10-45% of splits fall below 0.90 on the target-test half (worst split 0.70) and 15-40% on the replay remainder (worst split 0.69). Split conformal never promises per-realisation coverage, so 'throughout' should be replaced by the mean plus the worst-split range.
* A defensible replacement sentence (all numbers from the random-order replay at the pre-registered headline m): 'In a retrospective replay in which 13-30 target records (3-33% of each design region) were measured and the rest received calibrated intervals, mean coverage on the unmeasured 67-97% was 0.90-0.96 across 20 random splits (worst split 0.69-0.86) at median widths of 2.0-5.0 x the target-property IQR. These are coverage-certified intervals, not measurements, and the replay is not a prospective campaign.'
* The original demo quantile np.quantile(r, min(ceil((n+1)(1-a))/n, 1), method='higher') returns one order statistic too high for n in [20, 22, 23, 26, 30, 40, 52, 56, 116, 140] (checked sizes), e.g. rank 22 of 22 instead of 21 (E[coverage] 0.957 vs 0.913), and a finite sample maximum instead of an infinite interval for n < 9. This biases the submitted replay coverages (0.91-0.97) upwards at the budgets used there that are affected: [22, 23, 26, 30, 52, 56, 116, 140] (unaffected: [13]).

## Fix round: independent-verifier issues and how they were addressed

1. *'Floor' of the finite-test failure rate* (adopted). The Binomial(n_test, 0.9) tail is the limit as k -> infinity, not a floor; at conservative saw-tooth sizes the exact Beta-Binomial failure probability falls below it. The field `betabinom_reachable` was renamed `betabinom_stably_reachable` (= a stable k exists), and the JSON and the (c) table now give the minimum failure probability over k, its argmin, and the k values (up to 200 and within the recalibration pool) at which delta is met. Wording in (c) and in the interpretation was corrected.
2. *Record-level inner splits in the protocol-compliant (b) selection* (adopted). Inner splits are now grouped (cal = first k records of a grouped permutation of the pool, which is the deployed first-k rule; holdout = remaining whole compositions). B raised from 200 to 2000, and 5 independent inner-RNG replicates report the Monte Carlo spread of the certification counts, which are stated as approximate. The record-level version is kept as a labelled sensitivity.
3. *Replay horizon as an undisclosed analyst degree of freedom* (adopted). A horizon-sensitivity table covers every dataset, order and criterion. The interpretation lists every stopping point that changes between the default and strict horizons, and states that random-order C2 is reached only under the default convention. Building that table exposed a second artefact: without a minimum verification window, a 'stays from' stop at the very end of a horizon holds trivially (e.g. polymer_Tg random order 'reached' at n = 953 of 953). Every stop must now hold over >= 10 further measurements. The default-horizon stops were far from their horizon and did not change.
4. *Exchangeability overreach* (adopted). The agreement with Beta-Binomial theory is now described as guaranteed by the random pool/test partition, not as a property of the materials; the width cost is material-dependent, and prospective non-random acquisition is not covered.
5. *'Throughout' wording* (adopted). Both readings are stated. The mean reading essentially holds under the protocol (on the replay remainder and on target-test). The per-realisation reading does not, and split conformal never promised it.
6. *Float ceil in the Chebyshev / Cantelli budgets* (adopted). Exact rational arithmetic (fractions.Fraction). At alpha = 0.20, eps = 0.08 the Chebyshev budgets are now 499 (delta 0.05) and 249 (delta 0.10); Cantelli 474 and 224. The alpha = 0.10 values were unaffected (140 etc.).
7. *Unsupported causal explanation for 'no stable k'* (adopted). Replaced by: the oracle labels are noise-level for small regions (the 20-split p5 is essentially the second-lowest split; near misses listed), and the reliable reference is the Beta-Binomial theory at every integer k for the actual test size (new column in (b)).
8. *'Median width' label in (d)* (adopted). The text now says 'mean over splits of the per-split width ratio'.
9. *Proposed manuscript sentence mixing populations* (adopted). The replacement sentence now uses only the random-order replay at the pre-registered headline m. Coverage, worst split and width all refer to the unmeasured remainder, which is the population the '67-97% unmeasured' statement is about.
10. *Composition straddle at the replay cut* (adopted as disclosure). Straddle counts are computed in the script and tabulated in (e); they are negligible.

## Verifier issues not adopted

None. All ten issues were judged valid and addressed as described above.
