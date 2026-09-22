# R9 legacy analyses re-run under the unified protocol

This report covers the four remaining analyses of the submitted manuscript: Figure 11a (weighted conformal), Figure 12 (local coverage), Figure 10 (conformal risk control for glass-forming ability) and Figure 8a,b (ESOL, also Table 3/4 rows). Each one was re-run under `PROTOCOL.md` and `common.py`, checked for leakage, and its submitted numbers were reproduced from the original code path. That reproduction lets us attribute every change to a specific fix.

- Scripts: `revision/code/exp_R9_wcp.py`, `exp_R9_local.py`, `exp_R9_classif.py`, `exp_R9_esol.py`
- Output: `revision/results/R9_legacy.json`, which merges `R9_legacy_{wcp,local,classif,esol}.json`. Run logs are in `R9_legacy_*.log`.
- New split families were persisted with `save_splits` in `results/splits/`: `*_tilt{3,6,10}`, `gfa_composition`, `gfa_chemsys`, `esol_identity`, `esol_smiles`, `esol_none` and `esol_scaffold`. The design splits are those already saved by R1.
- Conventions: alpha = 0.10, finite-sample conformal quantile, and 20 splits (30 for the classification task, as specified). Intervals are given as mean [2.5, 97.5 percentile] across splits. All splits resample one dataset, so the intervals describe split-to-split variability, not population sampling error. Infinite conformal intervals are counted and reported, never dropped.
- Runtimes of this (second, corrected) run: WCP 22.6 min, local 3.3 min, classification 23.9 min, ESOL 1.8 min. The CPU was shared with other jobs and `n_jobs` was 4.
- This version incorporates every issue raised by the independent verifier. Section 5 lists each one, what was changed, and the two places where the verifier's framing was adopted but its own numbers differ slightly from ours (different RNG keys for a newly added arm).

## Verdicts at a glance

| Analysis | Submitted claim | Re-run under protocol | Verdict |
|---|---|---|---|
| Fig. 11a, weighted conformal | Two failure regimes. Glass and polymer: ESS/n 0.01–0.15, coverage stuck at 0.41–0.79. Steel: diffuse weights (ESS/n > 0.78), coverage = the 0.12 baseline. Recalibration gives 0.92–0.94. | The submitted numbers reproduce exactly from the original code. That code has an RFF bug and does **not** implement Tibshirani et al.'s weighted conformal (it omits the test-point weight). Under the correct procedure, every weight model lands in one of three regimes, none of which is a usable interval: (i) finite intervals that badly under-cover (glass Tg/E/Tliq at every D, glass HV at D = 5, coverage 0.30–0.58); (ii) an intermediate regime with partial vacuity (glass HV at D = 20: coverage 0.643 with 33 % infinite and 0.498 among the finite; polymer at D = 5: 0.879 with 28 % infinite); (iii) near-nominal coverage bought almost entirely with infinite intervals (steel, polymer at D >= 20, glass HV at D >= 50, and all raw-feature weights: 0.80–1.00 with 57–100 % infinite). Clipping the weights, the standard practical fix, restores finite intervals but collapses coverage back to 0.21–0.77, i.e. to roughly the unweighted level. Recalibration gives 0.90–0.95. Source and target supports are disjoint by construction in all six datasets. | Conclusion survives ("reweighting does not give informative valid intervals; target recalibration does"). The mechanism and the numbers in the text and the "two regimes" figure must be rewritten. |
| Fig. 12, local coverage | Marginal 0.81; about 0.64 in the target region | Marginal 0.797 [0.782, 0.813] over 2,349 held-out records. Direct target coverage is 0.401 [0.362, 0.432]. Kernel-smoothed local coverage at the target is 0.60–0.75 (leave-one-out), 0.663 at the submitted h = 0.55. | Qualitative claim holds and is stronger than stated. The submitted marginal is not wrong so much as uninformative: it is a mixture number that hides target coverage of 0.40. The two real defects are the inconsistent q0.80 target threshold and the single split with one unjustified bandwidth. |
| Fig. 10, CRC (FNR) | Naive 0.5 threshold FNR 6.0 %. CRC 10.2 % ± 1.3 % at a 10 % target. 82 % → 59 % flagged. | Naive 0.061. CRC 0.102 [0.080, 0.129] at 0.10. Flagged 0.82 → 0.59. Exact reproduction of the submitted split: 0.1021 ± 0.0137. | No leakage; numbers confirmed. The guarantee is in expectation, and 40–70 % of individual splits exceed alpha. Under a chemical-system split the split-to-split SD grows 4–5x (0.007–0.019 → 0.027–0.089) and per-split FNR at alpha = 0.10 spans 0.026–0.333, so the procedure is no longer usable for unseen chemistries. |
| Fig. 8a,b / Table 3, ESOL | Coverage at 90 %: conformal 0.91, RF heuristic 0.82. Triage cuts error by 28 % (RMSE 0.63 vs 0.88). 15 splits. | Identity-grouped 60/20/20, 20 splits: conformal 0.911 [0.867, 0.949], heuristic 0.803 [0.749, 0.865]. Triage 22.2 % [10.4, 33.5] (RMSE 0.735 vs 0.946). Scaffold split: 8.4 % [−5.0, 17.1]. | Calibration claim holds. The triage number falls because the six descriptors do not identify a molecule: 112 descriptor cells hold several **different** molecules, and leaving them on both sides of the split measures memorisation of a cell. Exact duplicates (11 molecules) contribute nothing. The triage panel also used a different, uncalibrated split. |

---

## 1. Figure 11a: weighted (likelihood-ratio) conformal (`exp_R9_wcp.py`)

### What changed compared with the original (`demo/exp5_singular_shift_diagnostic.py`, `demo/exp_rev_phase.py`)

1. **Splits.** The original used one 70/30 source split with seed 0. The re-run uses the protocol design split: 20 seeds, grouped, source-train 60 / source-cal 20 / source-test 20, with the target pool split into a recalibration pool and a target-test half. All six design-shift datasets are included (glass Tg, E, HV, Tliq; steel; polymer).
2. **Weight model uses no evaluation data.** The domain classifier is now trained only on source-train inputs versus target recalibration-pool inputs. The original trained it on source-cal inputs versus *all* target inputs, including the points on which coverage was then measured, and evaluated the weights in-sample on the same calibration points.
3. **Correct weighted conformal.** The re-run uses Tibshirani et al. (2019) via `common.weighted_conformal_q`: one quantile per target-test point, including the test point's own weight as a mass at +inf. The original used one global plug-in weighted quantile over the calibration weights, without the test weight. That estimator can never return an infinite interval and is anti-conservative exactly when test weights are large.
4. **RFF bug fixed.** The original `rff_embed` drew a new random projection on every call. The calibration inputs and the target inputs were therefore embedded with *two different random feature maps*, so the domain classifier compared two unrelated embeddings. The re-run uses one map (W, b) per (seed, D) for every domain.
5. **Weights from log-odds, no clipping in the main arms.** Weights are w = exp(logit e(x)), with a per-test-point common rescaling (the weighted quantile is scale-invariant). The original clipped probabilities to [1e-4, 1 − 1e-4].
6. **New arms.**
   - `raw_LR`: the standard low-dimensional set-up, logistic regression on standardised raw features.
   - `raw_HGB`: a flexible gradient-boosting domain classifier on raw features.
   - `*_clip0.99` (new in this revision): the practitioner's usual fix. Every log-weight, calibration and test, is capped at the 99th percentile of the **calibration** log-weights. Added for `rff_50`, `rff_500`, `raw_LR` and `raw_HGB`. This uses no target-test labels.
   - **Positive control:** an overlapping covariate shift with a known likelihood ratio. The source and target halves are thinned by sigma(∓lambda(s − 0.5)) on the rank-percentile of the shift feature, lambda ∈ {3, 6, 10}. The true ratio is then exp(lambda·s).
   - **Support check.**
   - **Diagnostics:** share of infinite intervals, **number of finite intervals** (new), coverage among finite intervals, IQR-normalised width, top-10 weight share, and domain AUC (an evaluation-only diagnostic on source-cal vs target-test inputs).

### Legacy reproduction (original code path, one split, 8 RFF draws)

| Case | code path | ESS/n over D = 5..500 | WCP coverage over D | vanilla | few-shot |
|---|---|---|---|---|---|
| glass_Tg | original code | 0.090, 0.009, 0.027, 0.041, 0.039, 0.045 | 0.46, 0.45, 0.55, 0.45, 0.42, 0.41 | 0.34 | 0.94 |
| glass_Tg | original, RFF bug fixed only | 0.105, 0.049, 0.079, 0.099, 0.111, 0.106 | 0.38, 0.56, 0.49, 0.48, 0.48, 0.49 | 0.34 | 0.94 |
| steel_yield | original code | 0.784, 0.883, 0.970, 0.985, 0.985, 0.988 | 0.12 at every D | 0.12 | 0.94 |
| steel_yield | original, RFF bug fixed only | 0.680, 0.751, 0.865, 0.845, 0.840, 0.841 | 0.15, 0.16, 0.14, 0.16, 0.17, 0.17 | 0.12 | 0.94 |
| polymer_Tg | original code | 0.037, 0.145, 0.098, 0.085, 0.102, 0.120 | 0.76, 0.77, 0.78, 0.76, 0.79, 0.79 | 0.72 | 0.92 |
| polymer_Tg | original, RFF bug fixed only | 0.107, 0.081, 0.153, 0.183, 0.205, 0.204 | 0.71, 0.68, 0.70, 0.70, 0.70, 0.70 | 0.72 | 0.92 |

- The submitted numbers are reproduced exactly: "ESS/n 0.01–0.15", "0.41–0.79", steel "ESS/n > 0.78 … 0.12" and recalibration "0.92–0.94".
- Fixing only the RFF bug changes the ESS values (for example, steel at D = 5 falls to 0.68, below the stated "> 0.78"). The qualitative pattern survives within the legacy procedure.
- The large change comes from using the correct weighted-conformal procedure (next table).

### Results under the protocol (20 splits)

Each weighted-conformal (WCP) cell shows coverage / share of target-test points with an infinite interval / ESS/n.

| Dataset (m) | Source-calibrated | Few-shot recal. (m) | WCP rff_5 | WCP rff_20 | WCP rff_50 | WCP rff_500 | WCP raw_LR | WCP raw_HGB |
|---|---|---|---|---|---|---|---|---|
| glass_Tg (30) | 0.40 [0.33, 0.48] | 0.93 [0.84, 0.99] | 0.46 / 0.00 / 0.232 | 0.52 / 0.00 / 0.106 | 0.53 / 0.02 / 0.099 | 0.58 / 0.05 / 0.117 | 0.89 / 0.57 / 0.008 | 1.00 / 1.00 / 0.917 |
| glass_E (22) | 0.23 [0.12, 0.33] | 0.92 [0.75, 1.00] | 0.33 / 0.00 / 0.215 | 0.39 / 0.03 / 0.266 | 0.38 / 0.01 / 0.300 | 0.47 / 0.06 / 0.347 | 0.87 / 0.67 / 0.015 | 1.00 / 1.00 / 0.301 |
| glass_HV (23) | 0.21 [0.14, 0.29] | 0.93 [0.84, 1.00] | 0.30 / 0.06 / 0.290 | 0.64 / 0.33 / 0.338 | 0.84 / 0.58 / 0.413 | 0.85 / 0.64 / 0.594 | 0.98 / 0.97 / 0.064 | 1.00 / 1.00 / 0.988 |
| glass_Tliq (30) | 0.41 [0.33, 0.49] | 0.90 [0.75, 0.99] | 0.48 / 0.00 / 0.445 | 0.49 / 0.00 / 0.287 | 0.53 / 0.00 / 0.290 | 0.54 / 0.00 / 0.297 | 0.87 / 0.70 / 0.058 | 1.00 / 1.00 / 0.454 |
| steel_yield (13) | 0.15 [0.02, 0.34] | 0.95 [0.83, 1.00] | 0.80 / 0.62 / 0.772 | 0.87 / 0.69 / 0.823 | 0.91 / 0.77 / 0.845 | 0.91 / 0.75 / 0.862 | 1.00 / 1.00 / 0.190 | 1.00 / 1.00 / 0.805 |
| polymer_Tg (30) | 0.76 [0.72, 0.82] | 0.90 [0.82, 0.98] | 0.88 / 0.28 / 0.124 | 0.98 / 0.81 / 0.161 | 0.99 / 0.90 / 0.175 | 1.00 / 0.95 / 0.211 | 1.00 / 1.00 / 0.081 | 1.00 / 1.00 / 1.000 |

- The source-calibrated and recalibrated columns match R1_core exactly (same splits, same RF).
- **Three regimes, not two.** Reading coverage together with the share of infinite intervals:
  - *finite but invalid*: glass Tg, E and Tliq at every D, and glass HV at D = 5. Coverage 0.30–0.58, essentially no infinite intervals, median finite width 0.77–1.48 × target IQR. These are informative intervals that simply do not cover.
  - *intermediate*: glass HV at D = 20 (coverage 0.643, 33.1 % infinite, 0.498 among the finite) and polymer at D = 5 (0.879, 28.3 % infinite, 0.847 among the finite). Partly vacuous, and the finite part still under-covers.
  - *vacuous*: steel at every D (62–77 % infinite), polymer at D ≥ 20 (81–96 %), glass HV at D ≥ 50 (58–64 %), and every raw-feature weight model (57–100 %). Coverage 0.80–1.00 is bought by refusing to predict.
- **Coverage of the finite intervals only**, where any exist: glass Tg RFF 0.46–0.57; glass E RFF 0.32–0.48; glass HV RFF 0.27–0.61; glass Tliq 0.48–0.54; steel RFF 0.52–0.70; polymer RFF 0.85–0.94. `raw_LR` gives 0.74 (glass Tg, 55 of 129 test points finite per split), 0.63 (Tliq, 20 of 65), 0.57 (E, 11 of 33) and, for glass HV, 0.49 — but that last figure rests on about **one finite interval per split in only 13 of the 20 splits** (mean 1 of 35 test points), so it should not be quoted as an endpoint of a range. `raw_LR` and `raw_HGB` produce no finite intervals at all for steel or polymer.
- Median width of the finite RFF intervals: 0.77 (glass Tg, D = 5) to 2.19 (polymer, D = 250) × target IQR. The recalibrated width is 2.08 (polymer) to 5.14 (glass HV) × IQR, which is the price of validity here.
- Domain AUC: 0.77–0.96 for RFF with D = 5; 0.93–1.00 for D ≥ 50 and for the raw-feature classifiers.

**Why this happens (support check).** In all six datasets every target record lies beyond the source maximum of the shift axis (`frac_target_beyond_source_max = 1.0`):
- glass source: P = 0 exactly; target: P ≥ 0.012–0.031;
- steel: source Ni ≤ 10.3 wt %, target Ni ≥ 17.5 wt %;
- polymer: source aromatic fraction ≤ 0.545, target ≥ 0.762.

The shift axis is itself an input feature, so the target distribution puts zero mass on the source support. The likelihood ratio that weighted conformal needs does not exist, whatever the embedding dimension. Estimated weights are extrapolations of the classifier. Where the classifier is smooth (low-D RFF, glass), the weights stay finite and coverage stays low. Where it separates the domains, the correct procedure returns infinite intervals.

**ESS/n is not a reliable diagnostic here.** Steel has the highest ESS/n (0.77–0.86) and yet 62–77 % infinite intervals, because failure is governed by the *test* point's weight relative to the calibration mass, which ESS/n of the calibration weights does not see. The share of infinite intervals, or the test-weight share, is the informative diagnostic.

**Does weight clipping rescue it? No.** Capping the log-weights at the 99th percentile of the calibration log-weights removes the infinite intervals almost everywhere: 0 % infinite for 21 of the 24 clipped cells, the exceptions being steel `raw_LR` (still 100 %, because essentially every weight is already at the cap), steel `raw_HGB` (20 %) and glass HV `raw_LR` (10 %). It also removes the coverage:

| Dataset | source-calibrated | rff_500 (inf) | rff_500_clip0.99 (inf) | raw_LR (inf) | raw_LR_clip0.99 (inf) | raw_HGB_clip0.99 (inf) |
|---|---|---|---|---|---|---|
| glass_Tg | 0.403 | 0.584 (0.05) | 0.497 (0.00) | 0.890 (0.57) | 0.548 (0.00) | 0.416 (0.00) |
| glass_E | 0.230 | 0.473 (0.06) | 0.345 (0.00) | 0.865 (0.67) | 0.412 (0.00) | 0.333 (0.00) |
| glass_HV | 0.210 | 0.847 (0.64) | 0.223 (0.00) | 0.984 (0.97) | 0.429 (0.10) | 0.213 (0.00) |
| glass_Tliq | 0.406 | 0.540 (0.00) | 0.473 (0.00) | 0.871 (0.70) | 0.503 (0.00) | 0.512 (0.00) |
| steel_yield | 0.155 | 0.912 (0.75) | 0.274 (0.00) | 1.000 (1.00) | 1.000 (1.00) | 0.321 (0.20) |
| polymer_Tg | 0.763 | 0.997 (0.95) | 0.766 (0.00) | 1.000 (1.00) | 0.758 (0.00) | 0.763 (0.00) |

Across all 24 clipped cells (excluding steel `raw_LR`, which stays vacuous) the coverage runs 0.213–0.766, i.e. it never comes closer than 0.13 to nominal, and it exceeds the unweighted source-calibrated coverage by only 0.00–0.22 — on polymer by −0.005 to +0.003, which is indistinguishable from not reweighting at all. This closes the obvious reviewer objection: after clipping, the choice is still between intervals that are finite and invalid and intervals that are valid and vacuous.

**It does work when its assumption holds.** In the positive control with overlapping supports and a known ratio:

| Dataset | lambda | n cal / n target-test (seed 0) | unweighted | oracle-weight WCP (share inf) | raw_LR WCP (share inf) | rff_500 WCP (share inf) |
|---|---|---|---|---|---|---|
| glass_Tg | 6 | 408 / 902 | 0.86 [0.83, 0.89] | 0.91 [0.85, 0.98] (0.00) | 0.93 [0.79, 0.98] (0.03) | 0.89 [0.85, 0.94] (0.00) |
| glass_Tg | 10 | 435 / 825 | 0.84 [0.80, 0.87] | 0.94 [0.89, 1.00] (0.31) | 0.85 [0.47, 1.00] (0.17) | 0.87 [0.84, 0.90] (0.02) |
| glass_HV | 10 | 100 / 180 | 0.79 [0.68, 0.85] | 0.91 [0.81, 0.98] (0.39) | 0.87 [0.60, 0.97] (0.20) | 0.88 [0.78, 0.96] (0.08) |
| glass_Tliq | 10 | 450 / 925 | 0.87 [0.84, 0.89] | 0.92 [0.85, 1.00] (0.14) | 0.89 [0.74, 0.98] (0.06) | 0.89 [0.85, 0.93] (0.00) |
| polymer_Tg | 10 | 388 / 1002 | 0.88 [0.82, 0.92] | 0.92 [0.84, 0.98] (0.26) | 0.90 [0.72, 0.98] (0.19) | 0.91 [0.83, 0.97] (0.12) |
| steel_yield | 6 | 13 / 34 | 0.73 [0.45, 0.99] | 0.97 [0.89, 1.00] (0.73) | 0.93 [0.60, 1.00] (0.64) | 0.97 [0.87, 1.00] (0.62) |

(The full table for lambda ∈ {3, 6, 10} and all six datasets is in the JSON under `wcp.positive_control`.)

- With oracle weights, weighted conformal is at or near nominal in every case (0.90–0.94 for glass and polymer; 0.896 for polymer at lambda = 3, where there is no under-coverage to correct). This validates the implementation.
- **The oracle arm is not free of vacuity either.** At lambda = 3 and 6 it returns almost no infinite intervals for glass and polymer (0–1 %, except glass HV at lambda = 6: 27 %). At lambda = 10, where the true ratio spans e^10, it is infinite for 31 % (glass Tg), 39 % (glass HV), 15 % (glass E), 14 % (glass Tliq) and 26 % (polymer) of target points, and for steel it is 47–81 % at every lambda. Since the whole argument of this section is that coverage bought with vacuous intervals does not count, the honest statement is: with overlapping support and a correct ratio, weighted conformal restores validity *and* keeps most intervals finite when the shift is mild-to-moderate; under a strong shift or a small calibration set it degrades towards vacuity even with perfect weights.
- Estimated weights help but are unstable across splits. For example, raw_LR on glass Tg at lambda = 10 gives 0.47–1.00, and in some cases does worse than no weighting (glass E lambda = 10: 0.81 vs 0.87; glass HV lambda = 6: 0.80 vs 0.83).

### Issues found in the original

1. The RFF embedding used different random maps for the calibration and target inputs (implementation bug).
2. The procedure was not the cited weighted conformal: no test-point weight, a single global quantile, so it could not produce vacuous intervals.
3. The weight classifier used all target inputs, including the evaluated points, and the weights were evaluated in-sample.
4. One source split, with 8 RFF draws presented as variability. Few-shot recalibration was evaluated on random complements of the whole target rather than on a fixed target-test half.
5. The failure was attributed to "high-capacity embeddings", but source and target are disjoint by construction, so the likelihood ratio is undefined at every D. This is a design issue rather than a leakage issue.

No label leakage affected the numbers. The "two failure regimes, coverage stuck at 0.41–0.79 / 0.12" description is an artefact of issues 1 and 2. Note that the Section 4.2 prose ("reweighting collapses to vacuous, uninformative prediction sets") is what the correct procedure shows. It is the Figure 11a caption and the Section 5 text that need rewriting. Suggested panel: coverage together with the share of infinite intervals versus D for each dataset, plus the recalibration reference, the clipped arm, and the positive-control inset.

---

## 2. Figure 12: local coverage in glass design space (`exp_R9_local.py`)

### What changed compared with the original (`demo/exp_rev_local_coverage.py`)

1. **Data and target definition.** The re-run uses `common.glass_dataset("Tg")` (25 elements; 7,623 records, 7,025 unique compositions) with the protocol target P ≥ q0.75(P | P > 0), which gives 258 glasses. The original used `bioglass_tg.npz` (24 elements) and a *different* target threshold, q0.80 (207 glasses), inconsistent with every other analysis.
2. **Splits.** 20 protocol design splits replace one 70/30 source split.
3. **Evaluation set, stated explicitly.** Every record not used for training or calibration is evaluated: source-test (1,318), all target records (258; both halves, since nothing is recalibrated here) and intermediate glasses with 0 < P < the target threshold (773). That is 2,349 per split. The original evaluated every non-training row, which for a 70/30 source split means its source rows were exactly its 1,978 calibration rows.
4. **Embedding.**
   - CLR: zeros replaced by a pseudo-count of 1e-4 (atomic fraction), rows re-closed to sum 1. Sensitivity checked at 1e-5 and 1e-3.
   - PCA: fitted once on all 7,623 **records**, using inputs only (no labels). PC1 + PC2 explain only 20.0 % + 16.5 % = 36.5 % of CLR variance, which the caption should state.
5. **Local-coverage estimators.**
   - Nadaraya–Watson with bandwidth h = {0.25, 0.5, 1, 2} × Scott (h_Scott ≈ 0.95), plus the submitted h = 0.55, reported **both in-sample and leave-one-out**. In the in-sample form a record's own coverage indicator enters with the Gaussian self-weight 1, which biases the estimate towards the raw local coverage; at the smallest bandwidth the difference is 0.035 (0.561 in-sample vs 0.596 LOO). The kNN estimator always excluded the point itself, so the two were defined inconsistently; the LOO column is now the comparable one and is used below.
   - A projection-free kNN local coverage in the full CLR space (k = 25, 50, 100).
   - Worst k-means-cluster coverage (k = 8, CLR).
   - The seed-averaged 90 × 90 field for h_Scott and h = 0.55, plus point coordinates and roles, are stored in the JSON (`local.field`, `local.points`) for the figure.
6. **Grid statistics are over supported cells.** `grid_frac_supported_below_0.9` now averages only over cells with kernel mass ≥ 3 (their share is reported as `grid_frac_supported`); the previous version compared a NaN-padded array, which silently counted every unsupported cell as "not below nominal" and understated the result. The old quantity is kept as `grid_frac_all_cells_below_0.9`.

### Numbers

| Quantity | Submitted | Legacy reproduction | Protocol (20 splits) |
|---|---|---|---|
| Marginal coverage on the evaluated set | 0.81 | 0.811 | 0.797 [0.782, 0.813] |
| Same, restricted to target + intermediate records (no source rows) | n/a | 0.638 | 0.664 |
| Source-test / intermediate / target coverage | n/a | source-cal rows 0.901; target (q0.80) 0.285; intermediate 0.727 | 0.900 [0.884, 0.918] / 0.752 [0.717, 0.774] / **0.401 [0.362, 0.432]** |
| Kernel-smoothed local coverage at target records, h = 0.55 (LOO) | about 0.64 | 0.636 (in-sample) | 0.663 [in-sample 0.652] |
| Same, h = 0.24 / 0.48 / 0.95 (Scott) / 1.91, LOO | n/a | n/a | 0.596 / 0.651 / 0.712 / 0.751 |
| Local coverage at source-test records, LOO (same bandwidths) | "above 0.90" | n/a | 0.865 / 0.853 / 0.836 / 0.820 (0.850 at h = 0.55) |
| kNN (CLR space) local coverage at target, k = 25 / 50 / 100 | n/a | n/a | 0.483 / 0.533 / 0.595 |
| Worst k-means cluster coverage (CLR, k = 8) | n/a | n/a | 0.43 [0.01, 0.59] |
| Share of **supported** grid area below 0.90 (h = 0.24 / 0.55 / 0.95 / 1.91) | n/a | n/a | 0.643 / 0.785 / 0.852 / 0.962 |
| Share of supported grid cells at those bandwidths | n/a | n/a | 0.43 / 0.81 / 0.93 / 1.00 |
| Pseudo-count 1e-5 / 1e-4 / 1e-3: Scott NW at target; kNN k = 25 | n/a | n/a | 0.701 / 0.707 / 0.730; 0.510 / 0.483 / 0.452 |

### Verdict and issues

- The qualitative message holds and is **stronger** than submitted. A marginal coverage of 0.80 coexists with a target region whose direct coverage is 0.40, and 64–96 % of the supported design-space grid (depending on bandwidth) sits below the nominal 0.90.
- The submitted "about 0.64" is a kernel-smoothed average that mixes target glasses with better-covered neighbours. It moves from 0.60 to 0.75 with bandwidth. Estimators with less smoothing (kNN, small h) go down towards the direct 0.40.
- The claim that source glasses are covered "above the nominal 0.90" is true for the direct source-test coverage (0.900). The smoothed field at source-test locations is 0.82–0.87, because smoothing also pulls those values towards the mixture.
- Issues found:
  1. **The evaluation set included the 1,978 source-calibration rows.** This is methodologically improper — those rows define the quantile, so their coverage is 0.90 by construction — but here it is numerically immaterial: they are covered at 0.9009 and the protocol's properly held-out source-test rows are covered at 0.9003, so swapping one for the other moves the marginal by about 0.001. The submitted 0.811 and the protocol 0.797 agree. The drop to 0.638 that appears when the calibration rows are simply deleted is *not* the removal of an optimistic bias: because the original used a 70/30 source split, its calibration rows were its only source rows, so deleting them deletes the entire source component of the mixture. The protocol's number over the same target + intermediate mixture is 0.664. **The real criticism is that a marginal number over an arbitrary mixture is not evidence about the design region**: it moves with the mixture and hides target coverage of 0.401.
  2. The target threshold (q0.80, 207 glasses) was inconsistent with all other figures (q0.75, 258 glasses). This is the one change that alters the target population itself.
  3. One split, and one bandwidth (0.55) fixed without justification or sensitivity analysis. The conclusion depends on the bandwidth by 0.15 in coverage.
  4. The at-point smoother was in-sample.
  5. The 2-D PCA captures only 36 % of the CLR variance, so the field is a projection, not the design space.
- No label leakage into the model: the RF used source-train only.

---

## 3. Figure 10: conformal risk control of the false-negative rate (`exp_R9_classif.py`)

### What changed compared with the original (`demo/exp1_classification_riskcontrol.py`)

- **Splits.** Grouped 60/20/20 (by identical Magpie vector) with 30 seeds. All 5,680 Magpie vectors and all 5,680 composition strings are unique, so this is equivalent to a random split. A different random stream is used than in the original.
- **Model.** Same fixed HistGradientBoosting hyperparameters as the original.
- **CRC threshold.** Vectorised, and asserted equal to the submitted loop implementation. The rule: the largest t with (n·R̂(t) + 1)/(n + 1) ≤ alpha on the calibration positives (n ≈ 805).
- **Row alignment verified.** The rows match matminer `matbench_glass`: labels are identical, and 25 randomly chosen re-featurised rows match. This gives the chemical system of each row.
- **Added:** share of splits whose realised FNR exceeds alpha; the full per-split FNR table at every alpha; min–max across splits; the excess of the mean FNR over alpha in units of the split-to-split standard error (one yardstick for all arms); FPR and precision; the uncorrected plug-in threshold as a comparator; a **chemical-system-grouped** sensitivity split (398 systems); and an exact reproduction of the submitted split.

### Numbers (30 splits)

| target alpha | CRC test FNR, grouped by composition (primary) | SD | excess (SE) | FNR range | share of splits FNR > alpha | flagged | FPR | CRC FNR, chemical-system split | SD | excess (SE) | FNR range | CRC FNR, submitted split (exact) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.022 [0.010, 0.037] | 0.007 | +0.0018 (1.35) | 0.009–0.038 | 0.53 | 0.820 | 0.430 | 0.028 [0.001, 0.080] | 0.027 | +0.0082 (1.69) | 0.000–0.100 | 0.020 |
| 0.04 | 0.041 [0.028, 0.058] | 0.010 | +0.0010 (0.55) | 0.026–0.059 | 0.47 | 0.774 | 0.316 | 0.052 [0.003, 0.126] | 0.041 | +0.0121 (1.64) | 0.000–0.127 | 0.039 |
| 0.06 | 0.063 [0.040, 0.078] | 0.011 | +0.0028 (1.41) | 0.036–0.079 | 0.70 | 0.739 | 0.249 | 0.080 [0.013, 0.189] | 0.054 | +0.0196 (1.97) | 0.012–0.246 | 0.061 |
| 0.08 | 0.082 [0.061, 0.108] | 0.014 | +0.0015 (0.62) | 0.057–0.108 | 0.40 | 0.713 | 0.207 | 0.103 [0.020, 0.226] | 0.065 | +0.0233 (1.97) | 0.013–0.320 | 0.081 |
| 0.10 | 0.102 [0.080, 0.129] | 0.014 | +0.0019 (0.76) | 0.080–0.132 | 0.57 | 0.690 | 0.177 | 0.122 [0.027, 0.247] | 0.069 | +0.0219 (1.75) | 0.026–0.333 | 0.102 |
| 0.12 | 0.124 [0.096, 0.155] | 0.017 | +0.0041 (1.33) | 0.090–0.155 | 0.60 | 0.666 | 0.149 | 0.142 [0.038, 0.263] | 0.071 | +0.0224 (1.74) | 0.035–0.356 | 0.122 |
| 0.15 | 0.156 [0.129, 0.187] | 0.018 | +0.0057 (1.73) | 0.128–0.189 | 0.57 | 0.635 | 0.120 | 0.172 [0.051, 0.303] | 0.078 | +0.0217 (1.52) | 0.045–0.401 | 0.151 |
| 0.20 | 0.206 [0.173, 0.237] | 0.019 | +0.0055 (1.56) | 0.172–0.237 | 0.60 | 0.591 | 0.088 | 0.220 [0.077, 0.363] | 0.089 | +0.0205 (1.26) | 0.075–0.465 | 0.203 |

"excess (SE)" is (mean realised FNR − alpha), and in brackets that excess divided by SD/sqrt(30). Because the 30 splits resample one finite dataset, this ratio is descriptive, not a valid hypothesis test; it is reported so that both arms are judged on the same scale.

**Naive 0.5 threshold:** FNR 0.061 [0.045, 0.084] with 0.741 flagged in the primary split; 0.060 on the submitted split; 0.122 on the chemical-system split. Test accuracy at 0.5 is 0.883 in the primary split and 0.782 on the chemical-system split. The uncorrected plug-in threshold gives FNR 0.001–0.002 higher than CRC at every alpha.

### Verdict and issues

- **No leakage found.** There are no duplicate compositions, and the threshold uses calibration data only. The submitted numbers reproduce exactly:
  - naive FNR 0.060 (submitted 6.0 %);
  - CRC 0.1021 ± 0.0137 at 0.10 (submitted 10.2 % ± 1.3 %; the submission used SD with ddof = 0);
  - flagged 0.82 → 0.59 (submitted 82 % → 59 %).
- **Both split families sit slightly above alpha on average, by a similar amount of evidence.** The excess is 0.6–1.7 split SE in the primary arm and 1.3–2.0 SE in the chemical-system arm. Neither is a demonstrated violation of E[FNR] ≤ alpha, and we do not claim one. The honest description of the primary arm is "tracks the target almost exactly", not "stays below the target".
- **What the chemical-system split actually shows** is not a larger mean but a different kind of failure, and this is unambiguous in the numbers:
  - the split-to-split SD rises from 0.007–0.019 to 0.027–0.089, a factor of 4–5;
  - the per-split FNR at alpha = 0.10 spans 0.026–0.333, against 0.080–0.132 in the primary arm;
  - 50–63 % of splits exceed alpha (against 40–70 %, i.e. the same);
  - flagging becomes far less selective (FPR 0.32–0.73 against 0.09–0.43), and accuracy falls from 0.883 to 0.782.
  - The reason is structural, not statistical: holding out whole chemical systems breaks point-level exchangeability between calibration and test, so the (n + 1) finite-sample correction is no longer the right one and the procedure has no guarantee to appeal to. The observed mean excess is consistent with that, but the usable statement for the manuscript is the variance and the loss of selectivity.
- **Wording.** The guarantee is on the *expected* FNR over calibration draws. In 40–70 % of individual splits the realised FNR exceeds alpha, which the text should say (Reviewer 2 comment 2). The FNR guarantee should be presented as valid for candidates exchangeable with the calibration set, not for new chemical systems.

---

## 4. ESOL/Delaney calibration and triage (`exp_R9_esol.py`; Fig. 8a,b and Table 3/4)

### What changed compared with the original (`demo/fig4_trust_layer_demo.py`)

- **One split family for both panels.** 60/20/20, 20 seeds, grouped by "identity" (primary). The original used two different families:
  - calibration panel: 55/20/25 with 15 seeds (Table 4 says 15);
  - triage panel: a separate 70/30 split (seeds 2000–2019) with *no calibration set*.
- **Identity groups** are connected components of records sharing a canonical RDKit SMILES or an identical six-descriptor vector. The file contains 11 duplicate molecules and 112 descriptor cells (314 records) that hold **more than one distinct molecule**. Grouping reduces 1,128 records to 919 groups. Sensitivity arms, new in this revision, separate the two effects:
  - `smiles`: group only the 11 true duplicate molecules;
  - `none`: plain random 60/20/20 at the same sizes;
  - `scaffold`: Bemis–Murcko scaffold groups. Two scaffolds hold 28 % and 23 % of the molecules, so groups are assigned by a randomised size-balancing rule rather than `common._cut`, which would have produced an empty calibration set.
- **Triage** is ranked by the normalised split-conformal half-width, with the scale calibrated on the calibration split. This gives the same ranking as the RF tree SD, because the calibrated scale is a common factor. The original ranked by raw tree SD without any calibration but labelled the panel "split conformal triage / conformal nonconformity ranking". The ranking effect belongs to the RF spread; conformal calibration adds the interval scale, not the ranking.
- **Full metric set.** Coverage curve, calibration error, width, IQR-normalised width, interval score, WIS, width-stratified and worst-cluster coverage (all for conformal, heuristic and normalised conformal); risk–coverage against random and oracle rankings; abstention at acceptable widths of 1.5 / 2 / 3 logS.
- The column "ESOL predicted log solubility" is excluded, as it was in the original. It would leak the label.

### Numbers (20 splits)

| Split | n train / test | cov90 conformal | cov90 RF heuristic | width conf / heur (logS) | calib. error conf / heur | interval score conf / heur | WIS conf / heur | worst-cluster cov conf / heur | triage reduction at 50 % | RMSE top-50 % / all |
|---|---|---|---|---|---|---|---|---|---|---|
| identity (primary) | 677 / 226 | 0.911 [0.867, 0.949] | 0.803 [0.749, 0.865] | 3.14 / 2.34 | 0.033 / 0.096 | 4.28 / 4.32 | 0.445 / 0.441 | 0.81 / 0.64 | 22.2 [10.4, 33.5] % | 0.735 / 0.946 |
| smiles (duplicates only) | 677 / 226 | 0.910 [0.865, 0.940] | 0.805 [0.754, 0.861] | 3.01 / 2.24 | 0.030 / 0.095 | 4.16 / 3.87 | 0.426 / 0.411 | 0.78 / 0.67 | 26.5 [16.5, 35.3] % | 0.663 / 0.904 |
| none (random) | 677 / 226 | 0.904 [0.865, 0.927] | 0.807 [0.750, 0.857] | 2.93 / 2.22 | 0.026 / 0.090 | 4.19 / 3.98 | 0.425 / 0.411 | 0.77 / 0.66 | 28.1 [16.7, 41.0] % | 0.650 / 0.906 |
| scaffold | 680 / 229 | 0.881 [0.771, 0.963] | 0.746 [0.650, 0.858] | 3.96 / 2.77 | 0.067 / 0.143 | 5.57 / 6.64 | 0.597 / 0.618 | 0.69 / 0.50 | 8.4 [−5.0, 17.1] % | 1.136 / 1.236 |

**Additional results, identity split:**
- Normalised conformal: coverage 0.916, width-stratified coverage 0.85 (heuristic 0.70).
- Triage baselines: a random ranking gives −0.3 % reduction; an oracle ranking gives 68.9 %. Relative AURC 0.79.
- Abstention at an acceptable full width of 3 logS: 55 % abstain; the accepted molecules have coverage 0.88 and RMSE 0.72.

**Legacy reproduction:**
- Calibration panel (55/20/25, 15 seeds): 0.9086 / 0.8160, widths 3.10 / 2.27, identical to the submission.
- Triage panel (70/30, seeds 2000–2019, 789 training molecules): 27.7 % (RMSE 0.637 vs 0.881); the submission reported 28.2 % (0.633 vs 0.881). The small gap comes from `int(c·n)` in the original versus `round` in `common.risk_coverage`.
- Overlap between test and training molecules in the submitted splits: an average of 62 of 283 test molecules (22 %) in the calibration panel and 84 of 339 (25 %) in the triage panel shared an identity group with a training molecule — but only 3.2 and 4.75 of them (about 1 %) were true duplicate molecules.

### Where the triage number actually goes (`esol.attribution_of_the_triage_change` in the JSON)

| Step | n train | triage reduction at 50 % | RMSE (all test) |
|---|---|---|---|
| Submitted 70/30, ungrouped | 789 | 27.7 % | 0.881 |
| Protocol 60/20/20, ungrouped | 677 | 28.1 % | 0.906 |
| + group the 11 duplicate molecules only | 677 | 26.5 % | 0.904 |
| + group descriptor-degenerate cells (= identity) | 677 | 22.2 % | 0.946 |
| Scaffold split | 680 | 8.4 % | 1.236 |

- The RMSE rise from 0.881 to 0.946 has **two** causes and the manuscript must not attribute it to one: 0.881 → 0.906 is the smaller training split (789 → 677 molecules), and only 0.906 → 0.946 is grouping.
- The triage drop from 28 % to 22 % is **not** duplicate leakage. Grouping the 11 exact duplicates changes it by 1.6 points, well inside the [16.5, 35.3] split-to-split interval. The whole effect comes from the 112 cells in which several *different* molecules share all six descriptors (314 records; within-cell SD of the measured logS 0.25 on average, up to 0.96 — these are genuinely different measurements). The six descriptors do not identify a molecule, so an ungrouped split lets the model retrieve a training label from a descriptor cell rather than generalise. Describing this as "leakage" would overstate it: a reviewer who opens `delaney.csv` finds only 11 duplicates. The correct description is **descriptor degeneracy**, and 22 % is the stricter, deployment-relevant number, not a correction of a data error.

### Verdict and issues

- **Calibration claim holds** under grouping: conformal 0.91 vs heuristic 0.80 at 90 %, with calibration error 0.033 vs 0.096. Under a harder scaffold split, conformal slips to 0.88 [0.77, 0.96], because exchangeability is only at the scaffold level, while the heuristic drops to 0.75.
- The two methods have similar interval score and WIS. Conformal's advantage is calibration and worst-cluster coverage, not a better proper score. The text should not claim more.
- Table 3 "28 % error reduction" and Fig. 8b "RMSE 0.63 vs 0.88" should become 22 % (0.74 vs 0.95), with the caveat above about what drives each part of the change. Table 4 should list 20 grouped 60/20/20 splits instead of 15.
- Issues found:
  1. Inconsistent split families between the two panels, and no calibration set in the triage panel.
  2. The "conformal" label on a raw-SD ranking.
  3. Ungrouped splits over a degenerate six-descriptor representation, which inflated the triage effect (28 % → 22 %) and understated RMSE (0.906 → 0.946 at matched training size).

---

## 5. Verifier issues and how each was handled

All ten issues raised by the independent verifier were accepted. None was rejected.

| # | Issue | Action |
|---|---|---|
| 1 | `grid_frac_supported_below_0.9` averaged over all 8,100 grid cells because `NaN < 0.9` is False, so unsupported cells were counted as "not below nominal". | Fixed in `exp_R9_local.py`: the statistic is now over `F[np.isfinite(F)]`, `grid_frac_supported` is stored alongside, and the old quantity is kept as `grid_frac_all_cells_below_0.9`. Re-run: 0.643 / 0.785 / 0.852 / 0.962 (was 0.278 / 0.634 / 0.791 / 0.961). The error was conservative; the corrected numbers strengthen the finding. |
| 2 | The ESOL 28 % → 22 % drop was attributed to duplicate/near-duplicate leakage; exact duplicates in fact contribute nothing. | A `smiles` split arm (duplicates only) was added and run: 26.5 % [16.5, 35.3], RMSE 0.904, against 28.1 % / 0.906 ungrouped and 22.2 % / 0.946 identity-grouped. The docstring and Section 4 now describe descriptor degeneracy, state that grouping exact duplicates alone changes nothing, and present 22 % as the stricter deployment-relevant number. (Our `smiles` figure is 26.5 %, the verifier's independent check gave 28.1 %; the arm uses its own RNG key, and the 1.6-point difference is well inside the split-to-split interval. The conclusion is the same.) |
| 3 | "RMSE rises from 0.88 to 0.95 (grouped)" confounded grouping with a smaller training split. | The JSON now carries `esol.attribution_of_the_triage_change`, and Section 4 splits it: 0.881 → 0.906 is training-set size (789 → 677), 0.906 → 0.946 is grouping. |
| 4 | "The evaluation set included the 1,978 calibration rows … without them the marginal is 0.638", framed as inflation, is not supported: calibration rows are covered at 0.9009 and held-out source-test rows at 0.9003. | Section 2 now says that evaluating on calibration rows is improper but numerically immaterial (≈0.001), that the 0.638 figure is a change of mixture (the 70/30 split left no other source rows — now recorded in `local.legacy.eval_mixture`), and puts the criticism where the evidence is: a mixture-dependent marginal that hides target coverage of 0.401, and the q0.80 threshold. The mixture-matched protocol number (0.664) is reported. |
| 5 | Two different statistical standards inside the classification section (primary arm "compatible with the guarantee", chemical-system arm "fails on average") for excesses of the same order. | `exp_R9_classif.py` now stores `fnr_excess_over_alpha`, `fnr_excess_in_split_se`, `fnr_min`, `fnr_max` and the full per-split FNR table for every arm and alpha. Section 3 uses one yardstick (0.6–1.7 SE vs 1.3–2.0 SE, neither a demonstrated violation) and leads with the unambiguous facts: SD up 4–5x, per-split FNR 0.026–0.333, FPR 0.32–0.73, and broken point-level exchangeability. |
| 6 | The verdict-table sentence named only one exception to the "finite-and-invalid or vacuous" dichotomy; `glass_HV` rff_20 (0.643, 33 % infinite) is a second. | The verdict table and Section 1 now name three regimes explicitly and list both intermediate cases. |
| 7 | The "0.49–0.74 (glass)" finite-interval range quoted an endpoint resting on ~1 finite interval per split in 13 of 20 splits. | `wcp_eval` now reports `n_finite` and `n_test`. Section 1 gives each `raw_LR` value with its support and states that the glass-HV 0.49 should not be quoted as a range endpoint. |
| 8 | The oracle positive control was described as near-nominal without noting its own infinite-interval shares at lambda = 10. | Section 1 now lists them (14–39 % for glass and polymer at lambda = 10; 47–81 % for steel at every lambda) and states the qualified conclusion. |
| 9 | The Nadaraya–Watson smoother was evaluated in-sample while the kNN estimator excluded the point itself. | `exp_R9_local.py` now computes both forms (`nw_self`) and reports `local_at_*_loo`; the md quotes the LOO values (target 0.596 / 0.651 / 0.712 / 0.751; 0.663 at h = 0.55) and the docstring states the difference. |
| 10 | "PCA fitted once on all 7,623 compositions" — there are 7,623 records but 7,025 unique compositions. | Corrected to "records" in the docstring, the JSON `protocol` block and the md. |

Not a defect, but adopted: the verifier's optional suggestion of a clipped-weight arm. `*_clip0.99` was added to `exp_R9_wcp.py` for `rff_50`, `rff_500`, `raw_LR` and `raw_HGB` (Section 1). It closes the obvious objection by showing that the practitioner's fix buys finite intervals at coverage 0.21–0.77.

**Unchanged numbers.** Every quantity that existed before this fix round and was not affected by a listed issue reproduces to the last digit: the whole WCP protocol table, the legacy WCP reproductions, the positive control, all three classification arms, the ESOL identity / none / scaffold arms and both ESOL legacy panels, and the local-coverage marginal and by-role coverages. The only changed values are the corrected grid statistics, the newly added LOO / `n_finite` / clipped / `smiles` / excess-SE quantities, and the runtimes.

---

## Deviations from protocol (all deliberate and disclosed)

1. **Classification (Fig. 10):** 30 seeds (per the task) instead of 20. It is an in-distribution task with no source/target pools; groups are identical Magpie vectors, which are all unique. It adds a chemical-system-grouped split family.
2. **ESOL:** in-distribution, with no target pool. The width normaliser is the IQR of the training-split logS. The scaffold arm uses a size-balancing grouped assignment instead of `common._cut` (`_cut` produced an empty calibration set with the 317- and 254-molecule scaffolds). The `smiles` arm is a grouping sensitivity arm, not a protocol result.
3. **Local coverage:** evaluates both target halves plus intermediate records, because nothing is recalibrated. PCA is fitted on the inputs of all records, target-test inputs included; no labels are used and the fit is purely descriptive.
4. **WCP:**
   - Domain AUC is computed on target-test *inputs* as an evaluation-only diagnostic; it is never used for a choice.
   - Target-pool IQR is used only to normalise reported widths, as in R1.
   - The positive control uses its own "tilt" split family (saved).
   - Classifier hyperparameters (LR C = 1; HGB 200 iterations, learning rate 0.1), the RFF bandwidth rule (median source-train–recal-pool distance) and the clipping quantile (0.99 of the calibration log-weights) were fixed a priori.
5. **All RF fits** use `make_model("RF")` with `n_jobs` overridden to 1 for CPU sharing. The fitted forests are identical to those in R1: the source-calibrated and recalibrated coverages in `wcp` match R1_core exactly.
6. **Legacy reproductions** intentionally use the submitted, non-protocol splits and code paths. They are labelled `legacy*` in the JSON and are not protocol results.
