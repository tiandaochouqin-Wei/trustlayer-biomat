# R4_realshift — genuine (non-constructed) shifts in SciGlass

Script `revision/code/exp_R4_realshift.py`; numbers from `revision/results/R4_realshift.json`. Reviewer 1 comment 2; Reviewer 2 comments 3 and 8. Protocol: `revision/code/PROTOCOL.md` (α = 0.1, finite-sample split-conformal quantile, grouped splits, 60/20/20 source, target halves = recalibration pool / target-test). Model: RF (`common.make_model('RF')`, 300 trees, min_samples_leaf = 2). Values are mean [2.5–97.5 percentile] over 20 evaluations (leave-lab-out: 4 repeats × 5 folds; other families: 20 seeds). The evaluations resample one finite dataset; the percentile range describes split-to-split variability, not population sampling error. 'Laboratory' = normalised first-author key (A2 §2.1), a proxy.

## 0. Bottom line

* **Held-out laboratories** (repeated grouped 5-fold, 20 evaluations). In-distribution row-cal coverage 0.90–0.91; on the target-test half row-cal (i.i.d.) coverage falls to 0.77–0.82 across the four properties (RMSE source-test→target-test: Tg 31.5→44.6; YoungModulus 5.81→8.89; Microhardness 0.834→1.14; Tliquidus 53→78.8). Laboratory-grouped calibration gives 0.85–0.90 at 1.3–1.9× the row-cal width. Weighted (covariate-shift) conformal gives 0.83–0.89 (infinite weighted quantiles are counted as covered; their share reaches 0.17, so the finite-interval coverage is lower than these numbers). Target recalibration with the headline m = 30 target records restores 0.90–0.91 (k = 10: 0.91–0.94) at 1.6–1.9× the row-cal width when recalibration and test records come from the same target pool (a); when they come from different laboratories (b) coverage is 0.87–0.90 (records spread over the pool's labs) and 0.82–0.87 (records taken in laboratory order, i.e. from the pool's first few laboratories).
  Per laboratory (≥ 20 records in the whole target fold, 3.8–20.2 laboratories per evaluation, evaluated on the records not used for that laboratory's own recalibration): row-cal median coverage 0.80–0.91 with 0.29–0.49 of labs < 0.80; lab-specific recalibration on 10 of the lab's own glasses gives median 0.89–0.91 with 0.00–0.01 of labs < 0.80. Decomposition: domain-classifier AUC 0.90–0.96 (genuine covariate shift), ESS/n 0.24–0.46, but density-ratio reweighting of the source residuals explains a covariate share of only 0.12–0.38 of the excess mean squared error under the disclosed exclusion rule (ESS/n ≥ 0.05 and a positive excess error; 0.12–0.19 if splits whose reweighted residuals exceed the target residuals are additionally discarded — that is a selection on the outcome, so the first range is the one to quote). The remainder is a laboratory/measurement (label) shift.
* **Future publications (temporal)** (20 seeds). In-distribution row-cal coverage 0.90–0.90; on the target-test half row-cal (i.i.d.) coverage falls to 0.68–0.79 across the four properties (RMSE source-test→target-test: Tg 31.2→55.5; YoungModulus 5.21→17.1; Microhardness 0.798→1.3; Tliquidus 55.2→76.9). Laboratory-grouped calibration gives 0.82–0.89 at 1.5–1.9× the row-cal width; recent-period calibration gives 0.59–0.93. Weighted (covariate-shift) conformal gives 0.69–0.90 (infinite weighted quantiles are counted as covered; their share reaches 0.28, so the finite-interval coverage is lower than these numbers). Target recalibration with the headline m = 30 target records restores 0.88–0.92 (k = 10: 0.89–0.92) at 1.7–5.0× the row-cal width when recalibration and test records come from the same target pool (a); when they come from different laboratories (b) coverage is 0.71–0.92 (records spread over the pool's labs) and 0.73–0.88 (records taken in laboratory order, i.e. from the pool's first few laboratories). Recalibration does not transfer across laboratories for Tliquidus (b, spread: 0.71; clustered: 0.73), whose target contains few laboratories.
  Per laboratory (≥ 20 records in the whole target fold, 2.0–5.0 laboratories per evaluation, evaluated on the records not used for that laboratory's own recalibration): row-cal median coverage 0.72–1.00 with 0.31–0.50 of labs < 0.80; lab-specific recalibration on 10 of the lab's own glasses gives median 0.90–0.91 with 0.00–0.02 of labs < 0.80. Decomposition: domain-classifier AUC 0.93–0.99 (genuine covariate shift), ESS/n 0.01–0.30, but density-ratio reweighting of the source residuals explains a covariate share of only 0.09–0.57 of the excess mean squared error under the disclosed exclusion rule (ESS/n ≥ 0.05 and a positive excess error; 0.09–0.43 if splits whose reweighted residuals exceed the target residuals are additionally discarded — that is a selection on the outcome, so the first range is the one to quote); not quoted because fewer than 10 of 20 splits are usable: YoungModulus (only 0 of 20 splits usable: 20 for ESS/n < 0.05); Tliquidus (only 0 of 20 splits usable: 20 for ESS/n < 0.05). The remainder is a laboratory/measurement (label) shift.
* **Chemically analysed compositions** (20 seeds). In-distribution row-cal coverage 0.90–0.90; on the target-test half row-cal (i.i.d.) coverage falls to 0.68–0.80 across the four properties (RMSE source-test→target-test: Tg 29.2→60.2; YoungModulus 5.99→7.45; Microhardness 0.765→1.49; Tliquidus 54.4→70.7). Laboratory-grouped calibration gives 0.81–0.93 at 1.6–2.0× the row-cal width. Weighted (covariate-shift) conformal gives 0.76–0.92 (infinite weighted quantiles are counted as covered; their share reaches 0.08, so the finite-interval coverage is lower than these numbers). Target recalibration with the headline m = 30 target records restores 0.89–0.91 (k = 10: 0.90–0.92) at 1.7–2.7× the row-cal width when recalibration and test records come from the same target pool (a); when they come from different laboratories (b) coverage is 0.86–0.91 (records spread over the pool's labs) and 0.78–0.88 (records taken in laboratory order, i.e. from the pool's first few laboratories).
  Per laboratory (≥ 20 records in the whole target fold, 2.0–15.0 laboratories per evaluation, evaluated on the records not used for that laboratory's own recalibration): row-cal median coverage 0.79–0.99 with 0.00–0.50 of labs < 0.80; lab-specific recalibration on 10 of the lab's own glasses gives median 0.91–0.92 with 0.00–0.01 of labs < 0.80. Decomposition: domain-classifier AUC 0.93–0.98 (genuine covariate shift), ESS/n 0.10–0.35, but density-ratio reweighting of the source residuals explains a covariate share of only 0.08–0.15 of the excess mean squared error under the disclosed exclusion rule (ESS/n ≥ 0.05 and a positive excess error; 0.08–0.15 if splits whose reweighted residuals exceed the target residuals are additionally discarded — that is a selection on the outcome, so the first range is the one to quote); not quoted because fewer than 10 of 20 splits are usable: YoungModulus (only 2 of 20 splits usable: 1 for no excess error; 18 because the reweighted residuals exceed the target residuals)  — where the reason is that the reweighted residuals exceed the target residuals, reweighting *over*-explains the excess error, which is not evidence of a small covariate share. The remainder is a laboratory/measurement (label) shift.
* **Identical compositions measured elsewhere** (model error on records whose exact composition, to 1e-3 atomic fraction, is in the training set; ungrouped random-row reference for the same-laboratory case): Tg RMSE 12.5 (same lab) vs 24.1 (other lab) vs 24.0 (held-out lab) K, implied laboratory component 20.4 K; YoungModulus RMSE 1.21 (same lab) vs 2.81 (other lab) vs 4.13 (held-out lab) GPa, implied laboratory component 3.95 GPa; Microhardness RMSE 0.48 (same lab) vs 0.79 (other lab) vs 0.75 (held-out lab) GPa, implied laboratory component 0.58 GPa; Tliquidus RMSE 20.9 (same lab) vs 26.1 (other lab) vs 28.5 (held-out lab) K, implied laboratory component 19.3 K. Repeating a composition in another laboratory therefore costs about as much as the average prediction error itself. It is **not**, however, the whole story: relative to the ungrouped random-row reference, holding out whole laboratories raises the RMSE by a similar factor on records whose composition the model has seen and on records whose composition it has not (lab-out ÷ random-row ratio, matched vs unmatched: Tg 1.30 vs 1.37; YoungModulus 2.22 vs 1.45; Microhardness 1.10 vs 1.41; Tliquidus 1.27 vs 1.44), so inter-laboratory disagreement and genuine novelty contribute comparably to the shift penalty (§ 5).
* **Phosphorus design shift, publication-grouped recalibration** (20 seeds): source-calibrated coverage on R1's own composition-grouped target-test (the uncorrected constructed shift, identical to `R1_core`) Tg 0.403, YoungModulus 0.230, Microhardness 0.210, Tliquidus 0.406; composition-grouped recalibration Tg 0.930, YoungModulus 0.921, Microhardness 0.933, Tliquidus 0.898; publication (Kod)-grouped, spread Tg 0.894, YoungModulus 0.878, Microhardness 0.891, Tliquidus 0.828; clustered Tg 0.825, YoungModulus 0.847, Microhardness 0.830, Tliquidus 0.792; Author+Year-grouped (A1 proxy), clustered Tg 0.813, YoungModulus 0.820, Microhardness 0.830, Tliquidus 0.792. The audit A1 reported microhardness 0.94 → 0.80 under publication grouping (10 seeds, submitted protocol); here the corresponding (clustered) arm gives 0.933 → 0.830: direction confirmed, magnitude milder (§ 6).
* **45S5 Bioglass**: 43 Tg records from 33 laboratories (1985–2016); Tg 728–887 K (median 805 K), SD 28.4 K, robust SD 17.8 K, SD of laboratory means 22.9 K. A random forest trained without any window record (and without any record sharing their composition group, but with chemically similar glasses outside the ±1 mol% window still in the training set — 10 further Tg records lie within ±2 mol% per oxide, so this is not an extrapolation test) predicts 797 K at the nominal composition. The row-calibrated 90 % interval is ±46 K and covers 0.89 of the records when each interval is centred on the model's prediction for that record's own composition (0.87 when every interval is centred on the nominal composition); the laboratory-calibrated interval is ±71 K and covers 0.95 (0.94 nominal-centred). For comparison 1.645 × SD = 47 K and 1.645 × robust SD = 29 K: the i.i.d. interval is barely wider than the inter-laboratory scatter of this one glass.

## 1. Data and shift definitions

| Property | n | labs | publications (Kod) | labs ≥ 20 rec. | largest lab | temporal cutoff: n source / target (target labs; share from labs unseen before) | analysed: n (labs; share from labs with no nominal record) |
|---|---|---|---|---|---|---|---|
| Tg | 7623 | 756 | 1271 | 90 | fujiwara_k (276) | ≥ 2016: 7216 / 407 (66; 0.69) | 946 (163; 0.74) |
| YoungModulus | 3812 | 259 | 431 | 44 | dejneka_m (499) | ≥ 2016: 3395 / 417 (22; 0.39) | 348 (36; 0.67) |
| Microhardness | 1756 | 215 | 292 | 19 | kawai_h (184) | ≥ 2012: 1367 / 389 (43; 0.76) | 209 (38; 0.69) |
| Tliquidus | 8282 | 351 | 678 | 101 | li_h (414) | ≥ 2017: 7937 / 345 (11; 0.52) | 1155 (52; 0.61) |

Leave-lab-out fold sizes (records per fold, min–max over folds and repeats): Tg 1524–1525; YoungModulus 762–763; Microhardness 351–352; Tliquidus 1656–1657.

**Calibration schemes compared in §§ 2–4.** All of them use α = 0.1 and the finite-sample split-conformal quantile; they differ in which records calibrate it and, for (ii) and (ii-t), also in which records train the model.

* **(i) row-cal** — the usual i.i.d. practice. One RF is trained on the composition-grouped 60 % of the source pool; the conformal quantile is taken from the composition-grouped 20 % source-calibration split. Every number labelled 'row-cal', every k-sweep and every target recalibration in (iii) and (iv) uses **this** model; only the calibration residuals change.
* **(ii) lab-cal** — the source pool is re-split 60/20/20 by *whole laboratories*. This arm therefore **refits the model** (a second RF on the laboratory-grouped 60 %) as well as changing the calibration set, so the lab-cal coverages and the width ratio (ii)/(i) mix a calibration effect with a model effect and are not a pure calibration comparison. Because laboratories are indivisible the realised fractions depart from 60/20/20: train 0.52–0.66, calibration 0.13–0.27, source-test 0.14–0.26 of the source pool (min–max over all evaluations and properties; the row-cal split is exact by construction, and the most unbalanced property is YoungModulus, whose largest laboratory dejneka_m alone holds 499 of 3812 records). The laboratory-grouped calibration set contains 8–176 laboratories.
* **(ii-t) recent-cal** (temporal family only) — trains on the earliest 60 % of the source by year and calibrates on the latest 20 %; the middle 20 % is its source-test. It also refits the model. Realised fractions: train 0.60–0.60, calibration 0.19–0.20.
* **(iii) target recalibration (recal k / recal m)** — the row-cal model with the conformal quantile recomputed from k (or the headline m) labelled records of the target recalibration pool.
* **(iv) lab-specific recalibration** — the row-cal model with the quantile recomputed from k = 10 records of the *same* held-out laboratory as the evaluated records.


## 2. PRIMARY — leave-laboratory-out (repeated grouped 5-fold, 4 repeats)

**Accuracy and marginal coverage on the target-test half (a) (same records for every method).** Coverage mean [2.5–97.5 pct]; normalised width = mean width / IQR of the target property values.

| Property | RMSE source-test → target-test | ID coverage row-cal (source-test) | (i) row-cal cov. | (ii) lab-cal cov. | weighted conformal cov. (inf. share) | (iii-a) recal m cov. | norm. width (i) / (ii) / (iii-a) | width ratio (ii)/(i), (iii-a)/(i) |
|---|---|---|---|---|---|---|---|---|
| Tg (m=30) | 31.5 → 44.6 K | 0.902 [0.889, 0.916] | 0.815 [0.754, 0.893] | 0.896 [0.819, 0.960] | 0.842 [0.786, 0.902] (0.01) | 0.910 [0.692, 0.992] | 0.84 / 1.27 / 1.53 | 1.52, 1.86 |
| YoungModulus (m=30) | 5.81 → 8.89 GPa | 0.908 [0.873, 0.933] | 0.788 [0.644, 0.922] | 0.899 [0.732, 0.968] | 0.889 [0.829, 0.949] (0.17) | 0.896 [0.791, 0.987] | 1.08 / 2.06 / 1.90 | 1.92, 1.82 |
| Microhardness (m=30) | 0.834 → 1.140 GPa | 0.898 [0.854, 0.936] | 0.783 [0.648, 0.931] | 0.848 [0.612, 0.969] | 0.834 [0.690, 0.945] (0.10) | 0.908 [0.828, 0.972] | 2.02 / 2.69 / 3.25 | 1.34, 1.63 |
| Tliquidus (m=30) | 53.0 → 78.8 K | 0.897 [0.880, 0.914] | 0.770 [0.642, 0.873] | 0.895 [0.799, 0.971] | 0.827 [0.772, 0.890] (0.06) | 0.906 [0.768, 0.972] | 0.86 / 1.49 / 1.60 | 1.73, 1.87 |

Pooled over all records of a repeat (each record held out exactly once per repeat; mean [min–max-like 2.5–97.5 pct] over 4 repeats): Tg row-cal 0.816 [0.811, 0.823], lab-cal 0.898 [0.893, 0.905]; YoungModulus row-cal 0.785 [0.778, 0.791], lab-cal 0.897 [0.883, 0.911]; Microhardness row-cal 0.777 [0.768, 0.792], lab-cal 0.845 [0.802, 0.897]; Tliquidus row-cal 0.770 [0.758, 0.780], lab-cal 0.894 [0.877, 0.920].

**Further metrics on target-test (a)** (interval score and WIS in property units; CE = mean |empirical − nominal| over levels 0.50–0.95; worst-cluster = lowest coverage among k-means clusters of the evaluated compositions).

| Property | method | interval score | WIS | CE mean | worst-cluster cov. |
|---|---|---|---|---|---|
| Tg | row | 234.0 | 20.8 | 0.130 | 0.507 |
| Tg | lab | 231.6 | 21.2 | 0.042 | 0.672 |
| Tg | recal m (a) | 245.0 | 20.6 | 0.067 | 0.715 |
| YoungModulus | row | 47.53 | 3.90 | 0.184 | 0.405 |
| YoungModulus | lab | 48.30 | 4.03 | 0.104 | 0.617 |
| YoungModulus | recal m (a) | 46.73 | 3.71 | 0.061 | 0.619 |
| Microhardness | row | 6.097 | 0.563 | 0.185 | 0.513 |
| Microhardness | lab | 5.905 | 0.546 | 0.099 | 0.568 |
| Microhardness | recal m (a) | 5.521 | 0.526 | 0.061 | 0.736 |
| Tliquidus | row | 431.1 | 38.1 | 0.161 | 0.427 |
| Tliquidus | lab | 408.5 | 38.7 | 0.052 | 0.632 |
| Tliquidus | recal m (a) | 403.0 | 36.8 | 0.071 | 0.712 |

**Target recalibration k-sweep on the target-test half.** (a) record-level halves (composition-grouped); (b) laboratory-grouped halves (recalibration and test records from different laboratories), recal records spread over the pool's labs or clustered (laboratory order). Coverage mean [2.5–97.5 pct]; normalised width in parentheses. k = 0 is the row-calibrated source quantile.

| Property | variant | k=0 | k=9 | k=10 | k=20 | k=30 | k=40 | headline m |
|---|---|---|---|---|---|---|---|---|
| Tg | (a) records | 0.815 [0.754, 0.893] (0.84) | 0.907 [0.693, 0.993] (1.74) | 0.914 [0.701, 0.993] (1.80) | 0.926 [0.712, 0.993] (1.82) | 0.910 [0.692, 0.992] (1.53) | 0.909 [0.845, 0.979] (1.34) | 0.910 [0.692, 0.992] (1.53; <0.90 in 0.25) |
| Tg | (b) labs, spread | 0.821 [0.757, 0.898] (0.84) | 0.904 [0.727, 0.992] (1.56) | 0.911 [0.727, 0.992] (1.61) | 0.908 [0.725, 0.989] (1.49) | 0.903 [0.737, 0.981] (1.37) | 0.906 [0.737, 0.991] (1.44) | 0.903 [0.737, 0.981] (1.37; <0.90 in 0.45) |
| Tg | (b) labs, clustered | 0.821 [0.757, 0.898] (0.84) | 0.794 [0.461, 0.968] (1.05) | 0.811 [0.502, 0.968] (1.08) | 0.851 [0.582, 0.993] (1.29) | 0.874 [0.672, 0.981] (1.22) | 0.887 [0.672, 0.982] (1.29) | 0.874 [0.672, 0.981] (1.22; <0.90 in 0.65) |
| YoungModulus | (a) records | 0.788 [0.644, 0.922] (1.08) | 0.936 [0.819, 0.995] (3.06) | 0.936 [0.819, 0.995] (3.06) | 0.908 [0.836, 0.987] (2.04) | 0.896 [0.791, 0.987] (1.90) | 0.907 [0.825, 0.969] (1.86) | 0.896 [0.791, 0.987] (1.90; <0.90 in 0.45) |
| YoungModulus | (b) labs, spread | 0.777 [0.615, 0.914] (1.08) | 0.899 [0.657, 1.000] (2.71) | 0.908 [0.657, 1.000] (2.94) | 0.876 [0.661, 0.992] (2.15) | 0.872 [0.671, 0.988] (1.86) | 0.876 [0.671, 0.994] (1.89) | 0.872 [0.671, 0.988] (1.86; <0.90 in 0.45) |
| YoungModulus | (b) labs, clustered | 0.777 [0.615, 0.914] (1.08) | 0.853 [0.459, 0.996] (2.26) | 0.862 [0.552, 0.996] (2.28) | 0.871 [0.644, 0.995] (2.30) | 0.872 [0.667, 0.992] (2.19) | 0.879 [0.644, 1.000] (2.33) | 0.872 [0.667, 0.992] (2.19; <0.90 in 0.50) |
| Microhardness | (a) records | 0.783 [0.648, 0.931] (2.02) | 0.930 [0.826, 0.997] (4.74) | 0.933 [0.826, 0.997] (4.77) | 0.917 [0.834, 0.994] (3.77) | 0.908 [0.828, 0.972] (3.25) | 0.893 [0.796, 0.969] (2.99) | 0.908 [0.828, 0.972] (3.25; <0.90 in 0.40) |
| Microhardness | (b) labs, spread | 0.759 [0.558, 0.961] (2.02) | 0.845 [0.448, 1.000] (3.29) | 0.857 [0.525, 1.000] (3.43) | 0.880 [0.631, 1.000] (3.60) | 0.877 [0.592, 0.996] (3.35) | 0.841 [0.627, 0.985] (2.83) | 0.877 [0.592, 0.996] (3.35; <0.90 in 0.35) |
| Microhardness | (b) labs, clustered | 0.759 [0.558, 0.961] (2.02) | 0.775 [0.392, 1.000] (3.27) | 0.783 [0.395, 1.000] (3.35) | 0.789 [0.374, 0.997] (3.29) | 0.857 [0.545, 1.000] (3.79) | 0.884 [0.468, 1.000] (4.11) | 0.857 [0.545, 1.000] (3.79; <0.90 in 0.45) |
| Tliquidus | (a) records | 0.770 [0.642, 0.873] (0.86) | 0.912 [0.688, 0.998] (2.33) | 0.914 [0.688, 0.998] (2.38) | 0.903 [0.688, 0.982] (1.63) | 0.906 [0.768, 0.972] (1.60) | 0.901 [0.764, 0.963] (1.50) | 0.906 [0.768, 0.972] (1.60; <0.90 in 0.35) |
| Tliquidus | (b) labs, spread | 0.760 [0.612, 0.886] (0.86) | 0.908 [0.649, 0.999] (2.05) | 0.908 [0.649, 0.999] (2.05) | 0.891 [0.680, 0.997] (1.62) | 0.885 [0.706, 0.993] (1.49) | 0.901 [0.807, 0.989] (1.54) | 0.885 [0.706, 0.993] (1.49; <0.90 in 0.60) |
| Tliquidus | (b) labs, clustered | 0.760 [0.612, 0.886] (0.86) | 0.783 [0.453, 1.000] (1.53) | 0.805 [0.503, 1.000] (1.68) | 0.837 [0.490, 0.999] (1.78) | 0.824 [0.501, 0.999] (1.72) | 0.813 [0.487, 0.999] (1.56) | 0.824 [0.501, 0.999] (1.72; <0.90 in 0.55) |

Pool sizes (records; labs in pool / test, mean): Tg (a) 762, (b) 770 (75 / 74 labs), labs in the m clustered records 5.0; YoungModulus (a) 381, (b) 389 (26 / 26 labs), labs in the m clustered records 4.2; Microhardness (a) 176, (b) 173 (20 / 23 labs), labs in the m clustered records 5.3; Tliquidus (a) 828, (b) 842 (34 / 36 labs), labs in the m clustered records 3.2.

**Per-laboratory conditional coverage** (laboratories with ≥ 20 records in the evaluated set; median over labs / share of labs with coverage < 0.80; mean over evaluations). 'exact' = share expected < 0.80 if every lab were covered at exactly 0.90 (binomial, given its n).

| Property | set | labs | row median / <0.8 | lab median / <0.8 | recal m (a) median / <0.8 | exact <0.8 |
|---|---|---|---|---|---|---|
| Tg | whole target, ≥ 20 | 18.0 | 0.91 / 0.32 | 0.98 / 0.15 | – | 0.03 |
| Tg | target-test (a), ≥ 20 | 8.2 | 0.89 / 0.31 | 0.95 / 0.17 | 0.96 / 0.15 | 0.04 |
| Tg | whole target, ≥ 10 | 37.4 | 0.90 / 0.35 | 0.98 / 0.15 | – | 0.07 |
| Tg | target-test (a), ≥ 10 | 18.9 | 0.91 / 0.32 | 0.98 / 0.17 | 0.98 / 0.14 | 0.07 |
| YoungModulus | whole target, ≥ 20 | 8.8 | 0.87 / 0.38 | 0.95 / 0.17 | – | 0.03 |
| YoungModulus | target-test (a), ≥ 20 | 3.9 | 0.87 / 0.37 | 0.94 / 0.11 | 0.96 / 0.13 | 0.03 |
| YoungModulus | whole target, ≥ 10 | 16.0 | 0.84 / 0.45 | 0.94 / 0.22 | – | 0.06 |
| YoungModulus | target-test (a), ≥ 10 | 9.3 | 0.87 / 0.34 | 0.96 / 0.13 | 0.97 / 0.12 | 0.07 |
| Microhardness | whole target, ≥ 20 | 3.8 | 0.91 / 0.29 | 0.93 / 0.21 | – | 0.04 |
| Microhardness | target-test (a), ≥ 20 | 0.9 | 0.82 / 0.33 | 0.90 / 0.17 | 0.96 / 0.06 | 0.04 |
| Microhardness | whole target, ≥ 10 | 10.2 | 0.93 / 0.32 | 0.98 / 0.23 | – | 0.07 |
| Microhardness | target-test (a), ≥ 10 | 3.8 | 0.86 / 0.32 | 0.93 / 0.22 | 0.99 / 0.08 | 0.07 |
| Tliquidus | whole target, ≥ 20 | 20.2 | 0.80 / 0.47 | 0.93 / 0.17 | – | 0.02 |
| Tliquidus | target-test (a), ≥ 20 | 11.6 | 0.78 / 0.52 | 0.93 / 0.16 | 0.94 / 0.17 | 0.03 |
| Tliquidus | whole target, ≥ 10 | 34.8 | 0.81 / 0.45 | 0.95 / 0.16 | – | 0.05 |
| Tliquidus | target-test (a), ≥ 10 | 21.2 | 0.80 / 0.50 | 0.94 / 0.18 | 0.94 / 0.20 | 0.06 |

Pooled per repeat (every held-out lab once per repeat; mean over 4 repeats), whole target: Tg (90 labs) row 0.93 / 0.32, lab-cal 0.98 / 0.15 (exact 0.03); YoungModulus (44 labs) row 0.89 / 0.41, lab-cal 0.99 / 0.18 (exact 0.03); Microhardness (19 labs) row 0.97 / 0.32, lab-cal 1.00 / 0.22 (exact 0.04); Tliquidus (101 labs) row 0.81 / 0.47, lab-cal 0.95 / 0.17 (exact 0.02).

**(iv) Lab-specific recalibration** ('a new laboratory measures 10 glasses'): per target laboratory with ≥ 20 records, k = 10 of its own records (composition-grouped; 10 draws per evaluation) recalibrate the row-cal model; coverage on the lab's remaining records (excluding the fold-level recalibration records). All methods on the same records. Median over labs / share of labs < 0.80.

| Property | labs | lab-specific k=10 | row | lab | recal m (a) | width lab-specific / row-cal (median) | lab-specific p10 over draws (median lab) |
|---|---|---|---|---|---|---|---|
| Tg | 18.0 | 0.91 / 0.00 | 0.91 / 0.33 | 0.97 / 0.15 | 0.97 / 0.15 | 1.09 | 0.80 |
| YoungModulus | 8.8 | 0.90 / 0.01 | 0.87 / 0.40 | 0.95 / 0.16 | 0.96 / 0.16 | 1.22 | 0.79 |
| Microhardness | 3.8 | 0.89 / 0.01 | 0.91 / 0.29 | 0.94 / 0.21 | 0.99 / 0.13 | 0.93 | 0.77 |
| Tliquidus | 20.2 | 0.91 / 0.01 | 0.80 / 0.49 | 0.93 / 0.17 | 0.94 / 0.17 | 1.58 | 0.81 |

**Covariate versus label/measurement shift** (A2 §2.5). Source-cal residual RMSE → the same residuals density-ratio reweighted to the target compositions (cross-fitted RF domain classifier) → actual target RMSE. Covariate share = (rw² − cal²)/(target² − cal²), a rough indicator. **Full exclusion rule** (the trimmed column): a split is dropped if (1) ESS/n < 0.05, (2) the target RMSE does not exceed the source-cal RMSE (no excess error to decompose), or (3) the reweighted RMSE exceeds the target RMSE (A2's caveat that the indicator is then meaningless). Rules (2) and (3) condition on the outcome, so the 'ESS-filter only' column (rules 1 and 2, i.e. no selection on the size of the estimate) and the per-reason exclusion counts are given alongside; the trimmed column can only be the lower of the two. A split may fail more than one condition, so the three counts do not add up to the number of splits removed (compare the n of the two share columns). 'Reweighted cov.' = coverage the row-cal quantile would have if only covariates shifted; 'actual' = row-cal coverage on the whole target.

| Property | domain AUC | ESS/n | RMSE cal → reweighted → target | covariate share, trimmed (n) | covariate share, ESS filter only: mean / median / max (n) | splits excluded: ESS / no excess / rw > target | reweighted cov. → actual cov. | novel-composition subset of target-test (a): share / row-cal / lab-cal / recal m |
|---|---|---|---|---|---|---|---|---|
| Tg | 0.896 | 0.43 | 31.6 → 33.7 → 43.8 | 0.15 (n=19 of 20) | 0.15 / 0.15 / 0.72 (n=19 of 20) | 0 / 1 / 1 | 0.885 → 0.816 | 0.93 / 0.807 / 0.891 / 0.906 |
| YoungModulus | 0.952 | 0.28 | 6.14 → 6.61 → 9.21 | 0.19 (n=14 of 20) | 0.38 / 0.23 / 2.42 (n=16 of 20) | 4 / 3 / 2 | 0.864 → 0.785 | 0.96 / 0.780 / 0.897 / 0.893 |
| Microhardness | 0.907 | 0.46 | 0.802 → 0.847 → 1.144 | 0.12 (n=19 of 20) | 0.12 / 0.08 / 0.60 (n=19 of 20) | 0 / 1 / 1 | 0.890 → 0.777 | 0.93 / 0.772 / 0.840 / 0.903 |
| Tliquidus | 0.956 | 0.24 | 52.8 → 58.6 → 78.8 | 0.16 (n=18 of 20) | 0.28 / 0.14 / 1.45 (n=20 of 20) | 0 / 0 / 2 | 0.881 → 0.770 | 0.97 / 0.763 / 0.892 / 0.903 |

Identical compositions (1e-3) in the target whose composition occurs in the source training set: Tg share 0.056 (cross-lab-only 0.056), RMSE matched 24.0 vs unmatched 44.7 K, row-cal cov. matched 0.937; YoungModulus share 0.031 (cross-lab-only 0.031), RMSE matched 4.13 vs unmatched 9.32 GPa, row-cal cov. matched 0.955; Microhardness share 0.039 (cross-lab-only 0.039), RMSE matched 0.752 vs unmatched 1.154 GPa, row-cal cov. matched 0.884; Tliquidus share 0.032 (cross-lab-only 0.032), RMSE matched 28.5 vs unmatched 79.9 K, row-cal cov. matched 0.971.

**Selective prediction and locally adaptive intervals** (normalised score |r|/RF tree SD; target-test (a)): RMSE reduction on the 50 % most confident (ID → target), width-stratified coverage (worst width quartile) and abstention rate (share of target candidates whose recalibrated adaptive interval is wider than the in-distribution conformal width).

| Property | RMSE reduction @50 % ID → target | adaptive source-cal cov. | adaptive recal m cov. | worst width-quartile cov. (recal) | abstention |
|---|---|---|---|---|---|
| Tg | 49.2 % → 30.4 % | 0.828 [0.759, 0.888] | 0.914 [0.756, 0.990] | 0.866 | 0.75 |
| YoungModulus | 71.1 % → 30.9 % | 0.866 [0.720, 0.939] | 0.906 [0.780, 0.987] | 0.850 | 0.69 |
| Microhardness | 57.8 % → 23.9 % | 0.788 [0.640, 0.937] | 0.906 [0.801, 0.983] | 0.809 | 0.79 |
| Tliquidus | 52.5 % → 37.1 % | 0.834 [0.765, 0.879] | 0.898 [0.727, 0.977] | 0.859 | 0.71 |


## 3. SECONDARY — temporal (future publications)

**Accuracy and marginal coverage on the target-test half (a) (same records for every method).** Coverage mean [2.5–97.5 pct]; normalised width = mean width / IQR of the target property values.

| Property | RMSE source-test → target-test | ID coverage row-cal (source-test) | (i) row-cal cov. | (ii) lab-cal cov. | (ii-t) recent-cal cov. | weighted conformal cov. (inf. share) | (iii-a) recal m cov. | norm. width (i) / (ii) / (iii-a) | width ratio (ii)/(i), (iii-a)/(i) |
|---|---|---|---|---|---|---|---|---|---|
| Tg (m=30) | 31.2 → 55.5 K | 0.895 [0.875, 0.912] | 0.683 [0.620, 0.747] | 0.820 [0.751, 0.922] | 0.872 [0.845, 0.901] | 0.780 [0.722, 0.850] (0.05) | 0.921 [0.839, 0.985] | 0.97 / 1.47 / 2.45 | 1.51, 2.53 |
| YoungModulus (m=30) | 5.21 → 17.13 GPa | 0.902 [0.868, 0.926] | 0.757 [0.683, 0.818] | 0.845 [0.741, 0.903] | 0.595 [0.540, 0.671] | 0.694 [0.606, 0.785] (0.16) | 0.908 [0.758, 0.995] | 1.08 / 2.07 / 5.47 | 1.93, 5.04 |
| Microhardness (m=30) | 0.798 → 1.304 GPa | 0.898 [0.836, 0.938] | 0.789 [0.697, 0.839] | 0.889 [0.818, 0.959] | 0.932 [0.910, 0.962] | 0.901 [0.770, 0.964] (0.07) | 0.906 [0.797, 0.970] | 2.28 / 3.66 / 4.14 | 1.62, 1.81 |
| Tliquidus (m=30) | 55.2 → 76.9 K | 0.901 [0.883, 0.916] | 0.751 [0.690, 0.780] | 0.853 [0.727, 0.963] | 0.877 [0.849, 0.905] | 0.823 [0.725, 0.936] (0.28) | 0.880 [0.751, 0.960] | 0.60 / 0.98 / 1.04 | 1.64, 1.74 |

**Further metrics on target-test (a)** (interval score and WIS in property units; CE = mean |empirical − nominal| over levels 0.50–0.95; worst-cluster = lowest coverage among k-means clusters of the evaluated compositions).

| Property | method | interval score | WIS | CE mean | worst-cluster cov. |
|---|---|---|---|---|---|
| Tg | row | 350.6 | 29.0 | 0.233 | 0.396 |
| Tg | lab | 319.9 | 28.0 | 0.094 | 0.461 |
| Tg | recent | 239.9 | 22.9 | 0.031 | 0.489 |
| Tg | recal m (a) | 286.0 | 26.8 | 0.060 | 0.774 |
| YoungModulus | row | 103.90 | 6.58 | 0.169 | 0.217 |
| YoungModulus | lab | 91.89 | 6.40 | 0.064 | 0.406 |
| YoungModulus | recent | 97.17 | 6.69 | 0.348 | 0.213 |
| YoungModulus | recal m (a) | 110.77 | 6.58 | 0.065 | 0.587 |
| Microhardness | row | 7.221 | 0.609 | 0.168 | 0.442 |
| Microhardness | lab | 6.813 | 0.599 | 0.057 | 0.573 |
| Microhardness | recent | 7.360 | 0.625 | 0.048 | 0.813 |
| Microhardness | recal m (a) | 6.903 | 0.586 | 0.059 | 0.652 |
| Tliquidus | row | 449.6 | 39.2 | 0.152 | 0.378 |
| Tliquidus | lab | 366.3 | 40.0 | 0.070 | 0.629 |
| Tliquidus | recent | 418.7 | 52.8 | 0.080 | 0.594 |
| Tliquidus | recal m (a) | 372.0 | 37.5 | 0.056 | 0.645 |

**Target recalibration k-sweep on the target-test half.** (a) record-level halves (composition-grouped); (b) laboratory-grouped halves (recalibration and test records from different laboratories), recal records spread over the pool's labs or clustered (laboratory order). Coverage mean [2.5–97.5 pct]; normalised width in parentheses. k = 0 is the row-calibrated source quantile.

| Property | variant | k=0 | k=9 | k=10 | k=20 | k=30 | k=40 | headline m |
|---|---|---|---|---|---|---|---|---|
| Tg | (a) records | 0.683 [0.620, 0.747] (0.97) | 0.885 [0.659, 0.993] (2.32) | 0.904 [0.675, 0.993] (2.49) | 0.916 [0.781, 0.988] (2.47) | 0.921 [0.839, 0.985] (2.45) | 0.918 [0.839, 0.985] (2.35) | 0.921 [0.839, 0.985] (2.45; <0.90 in 0.30) |
| Tg | (b) labs, spread | 0.674 [0.517, 0.824] (0.97) | 0.884 [0.670, 0.993] (2.20) | 0.887 [0.670, 0.993] (2.22) | 0.885 [0.599, 0.993] (2.40) | 0.880 [0.733, 0.993] (2.18) | 0.877 [0.676, 0.990] (2.06) | 0.880 [0.733, 0.993] (2.18; <0.90 in 0.50) |
| Tg | (b) labs, clustered | 0.674 [0.517, 0.824] (0.97) | 0.730 [0.206, 0.988] (1.61) | 0.737 [0.206, 0.988] (1.65) | 0.770 [0.238, 0.990] (1.93) | 0.833 [0.440, 0.990] (2.30) | 0.875 [0.535, 0.993] (2.53) | 0.833 [0.440, 0.990] (2.30; <0.90 in 0.40) |
| YoungModulus | (a) records | 0.757 [0.683, 0.818] (1.08) | 0.920 [0.789, 1.000] (7.25) | 0.920 [0.789, 1.000] (7.25) | 0.892 [0.730, 0.986] (5.50) | 0.908 [0.758, 0.995] (5.47) | 0.911 [0.780, 0.988] (5.11) | 0.908 [0.758, 0.995] (5.47; <0.90 in 0.40) |
| YoungModulus | (b) labs, spread | 0.777 [0.598, 0.892] (1.08) | 0.910 [0.658, 1.000] (6.40) | 0.930 [0.827, 1.000] (6.61) | 0.890 [0.522, 1.000] (5.69) | 0.917 [0.793, 1.000] (5.34) | 0.909 [0.797, 1.000] (4.73) | 0.917 [0.793, 1.000] (5.34; <0.90 in 0.40) |
| YoungModulus | (b) labs, clustered | 0.777 [0.598, 0.892] (1.08) | 0.850 [0.609, 1.000] (3.14) | 0.850 [0.609, 1.000] (3.17) | 0.886 [0.723, 1.000] (5.15) | 0.881 [0.685, 1.000] (5.59) | 0.891 [0.628, 1.000] (6.49) | 0.881 [0.685, 1.000] (5.59; <0.90 in 0.55) |
| Microhardness | (a) records | 0.789 [0.697, 0.839] (2.28) | 0.889 [0.694, 0.990] (4.66) | 0.910 [0.734, 1.000] (5.30) | 0.910 [0.743, 0.998] (4.82) | 0.906 [0.797, 0.970] (4.14) | 0.905 [0.807, 0.970] (4.12) | 0.906 [0.797, 0.970] (4.14; <0.90 in 0.40) |
| Microhardness | (b) labs, spread | 0.804 [0.715, 0.877] (2.28) | 0.901 [0.767, 1.000] (5.11) | 0.901 [0.767, 1.000] (5.11) | 0.896 [0.791, 0.995] (4.57) | 0.879 [0.765, 0.997] (3.78) | 0.897 [0.791, 0.997] (4.14) | 0.879 [0.765, 0.997] (3.78; <0.90 in 0.60) |
| Microhardness | (b) labs, clustered | 0.804 [0.715, 0.877] (2.28) | 0.793 [0.370, 0.980] (3.87) | 0.801 [0.370, 0.980] (4.01) | 0.796 [0.443, 1.000] (4.00) | 0.811 [0.488, 1.000] (3.90) | 0.832 [0.510, 1.000] (4.37) | 0.811 [0.488, 1.000] (3.90; <0.90 in 0.60) |
| Tliquidus | (a) records | 0.751 [0.690, 0.780] (0.60) | 0.887 [0.568, 0.989] (1.10) | 0.890 [0.568, 0.992] (1.13) | 0.888 [0.710, 0.989] (1.08) | 0.880 [0.751, 0.960] (1.04) | 0.884 [0.791, 0.949] (1.05) | 0.880 [0.751, 0.960] (1.04; <0.90 in 0.60) |
| Tliquidus | (b) labs, spread | 0.675 [0.541, 0.924] (0.60) | 0.792 [0.377, 0.997] (1.05) | 0.794 [0.397, 0.997] (1.05) | 0.766 [0.507, 0.997] (0.96) | 0.706 [0.506, 0.997] (0.83) | 0.713 [0.481, 0.997] (0.84) | 0.706 [0.506, 0.997] (0.83; <0.90 in 0.70) |
| Tliquidus | (b) labs, clustered | 0.675 [0.541, 0.924] (0.60) | 0.689 [0.256, 0.994] (0.82) | 0.695 [0.284, 0.994] (0.83) | 0.715 [0.397, 0.997] (0.82) | 0.727 [0.414, 0.986] (0.81) | 0.725 [0.441, 0.988] (0.84) | 0.727 [0.414, 0.986] (0.81; <0.90 in 0.70) |

Pool sizes (records; labs in pool / test, mean): Tg (a) 204, (b) 203 (32 / 34 labs), labs in the m clustered records 5.5; YoungModulus (a) 208, (b) 192 (13 / 9 labs), labs in the m clustered records 3.6; Microhardness (a) 194, (b) 194 (21 / 22 labs), labs in the m clustered records 3.8; Tliquidus (a) 172, (b) 171 (5 / 6 labs), labs in the m clustered records 3.0.

**Per-laboratory conditional coverage** (laboratories with ≥ 20 records in the evaluated set; median over labs / share of labs with coverage < 0.80; mean over evaluations). 'exact' = share expected < 0.80 if every lab were covered at exactly 0.90 (binomial, given its n).

| Property | set | labs | row median / <0.8 | lab median / <0.8 | time median / <0.8 | recal m (a) median / <0.8 | exact <0.8 |
|---|---|---|---|---|---|---|---|
| Tg | whole target, ≥ 20 | 3.0 | 1.00 / 0.33 | 1.00 / 0.20 | 1.00 / 0.00 | – | 0.06 |
| Tg | target-test (a), ≥ 20 | 0.0 | – / – | – / – | – / – | – / – | – |
| Tg | whole target, ≥ 10 | 13.0 | 0.74 / 0.53 | 0.97 / 0.30 | 1.00 / 0.16 | – | 0.09 |
| Tg | target-test (a), ≥ 10 | 4.6 | 0.89 / 0.37 | 0.96 / 0.25 | 1.00 / 0.07 | 0.96 / 0.16 | 0.10 |
| YoungModulus | whole target, ≥ 20 | 2.0 | 0.75 / 0.50 | 0.85 / 0.10 | 0.47 / 0.97 | – | 0.00 |
| YoungModulus | target-test (a), ≥ 20 | 2.0 | 0.74 / 0.50 | 0.85 / 0.28 | 0.47 / 0.95 | 0.87 / 0.23 | 0.02 |
| YoungModulus | whole target, ≥ 10 | 9.0 | 0.84 / 0.39 | 0.94 / 0.21 | 0.78 / 0.49 | – | 0.06 |
| YoungModulus | target-test (a), ≥ 10 | 2.8 | 0.83 / 0.38 | 0.89 / 0.24 | 0.63 / 0.78 | 0.92 / 0.17 | 0.03 |
| Microhardness | whole target, ≥ 20 | 5.0 | 0.99 / 0.31 | 1.00 / 0.20 | 1.00 / 0.20 | – | 0.03 |
| Microhardness | target-test (a), ≥ 20 | 1.4 | 0.88 / 0.25 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.06 |
| Microhardness | whole target, ≥ 10 | 15.0 | 0.99 / 0.20 | 1.00 / 0.14 | 1.00 / 0.13 | – | 0.07 |
| Microhardness | target-test (a), ≥ 10 | 4.9 | 0.95 / 0.26 | 1.00 / 0.13 | 1.00 / 0.11 | 0.99 / 0.08 | 0.08 |
| Tliquidus | whole target, ≥ 20 | 2.0 | 0.73 / 0.50 | 0.84 / 0.28 | 0.85 / 0.50 | – | 0.00 |
| Tliquidus | target-test (a), ≥ 20 | 2.0 | 0.73 / 0.53 | 0.84 / 0.33 | 0.85 / 0.50 | 0.87 / 0.30 | 0.01 |
| Tliquidus | whole target, ≥ 10 | 6.0 | 0.93 / 0.33 | 0.95 / 0.23 | 0.99 / 0.17 | – | 0.08 |
| Tliquidus | target-test (a), ≥ 10 | 2.6 | 0.84 / 0.41 | 0.91 / 0.26 | 0.93 / 0.40 | 0.91 / 0.25 | 0.02 |

**(iv) Lab-specific recalibration** ('a new laboratory measures 10 glasses'): per target laboratory with ≥ 20 records, k = 10 of its own records (composition-grouped; 10 draws per evaluation) recalibrate the row-cal model; coverage on the lab's remaining records (excluding the fold-level recalibration records). All methods on the same records. Median over labs / share of labs < 0.80.

| Property | labs | lab-specific k=10 | row | lab | time | recal m (a) | width lab-specific / row-cal (median) | lab-specific p10 over draws (median lab) |
|---|---|---|---|---|---|---|---|---|
| Tg | 3.0 | 0.91 / 0.00 | 1.00 / 0.33 | 1.00 / 0.20 | 1.00 / 0.00 | 1.00 / 0.13 | 0.51 | 0.79 |
| YoungModulus | 2.0 | 0.91 / 0.00 | 0.75 / 0.50 | 0.85 / 0.15 | 0.47 / 0.97 | 0.88 / 0.10 | 7.44 | 0.81 |
| Microhardness | 5.0 | 0.91 / 0.02 | 0.99 / 0.31 | 1.00 / 0.20 | 1.00 / 0.18 | 1.00 / 0.21 | 0.75 | 0.81 |
| Tliquidus | 2.0 | 0.90 / 0.00 | 0.72 / 0.50 | 0.84 / 0.30 | 0.85 / 0.50 | 0.87 / 0.25 | 1.71 | 0.81 |

**Covariate versus label/measurement shift** (A2 §2.5). Source-cal residual RMSE → the same residuals density-ratio reweighted to the target compositions (cross-fitted RF domain classifier) → actual target RMSE. Covariate share = (rw² − cal²)/(target² − cal²), a rough indicator. **Full exclusion rule** (the trimmed column): a split is dropped if (1) ESS/n < 0.05, (2) the target RMSE does not exceed the source-cal RMSE (no excess error to decompose), or (3) the reweighted RMSE exceeds the target RMSE (A2's caveat that the indicator is then meaningless). Rules (2) and (3) condition on the outcome, so the 'ESS-filter only' column (rules 1 and 2, i.e. no selection on the size of the estimate) and the per-reason exclusion counts are given alongside; the trimmed column can only be the lower of the two. A split may fail more than one condition, so the three counts do not add up to the number of splits removed (compare the n of the two share columns). 'Reweighted cov.' = coverage the row-cal quantile would have if only covariates shifted; 'actual' = row-cal coverage on the whole target.

| Property | domain AUC | ESS/n | RMSE cal → reweighted → target | covariate share, trimmed (n) | covariate share, ESS filter only: mean / median / max (n) | splits excluded: ESS / no excess / rw > target | reweighted cov. → actual cov. | novel-composition subset of target-test (a): share / row-cal / lab-cal / recal m |
|---|---|---|---|---|---|---|---|---|
| Tg | 0.931 | 0.25 | 30.5 → 33.2 → 55.4 | 0.09 (n=20 of 20) | 0.09 / 0.07 / 0.31 (n=20 of 20) | 0 / 0 / 0 | 0.880 → 0.682 | 0.95 / 0.676 / 0.810 / 0.917 |
| YoungModulus | 0.955 | 0.02 | 5.54 → 3.78 → 16.70 | – (n=0 of 20) | – / – / – (n=0 of 20) | 20 / 0 / 0 | 0.948 → 0.765 | 0.70 / 0.658 / 0.780 / 0.871 |
| Microhardness | 0.937 | 0.30 | 0.783 → 1.075 → 1.316 | 0.43 (n=17 of 20) | 0.57 / 0.55 / 1.87 (n=20 of 20) | 0 / 0 / 3 | 0.862 → 0.792 | 0.93 / 0.788 / 0.893 / 0.909 |
| Tliquidus | 0.986 | 0.01 | 54.2 → 37.7 → 76.3 | – (n=0 of 20) | – / – / – (n=0 of 20) | 20 / 0 / 0 | 0.946 → 0.751 | 0.78 / 0.692 / 0.815 / 0.850 |

Identical compositions (1e-3) in the target whose composition occurs in the source training set: Tg share 0.052 (cross-lab-only 0.032), RMSE matched 25.6 vs unmatched 56.6 K, row-cal cov. matched 0.861; YoungModulus share 0.185 (cross-lab-only 0.016), RMSE matched 2.01 vs unmatched 18.47 GPa, row-cal cov. matched 0.979; Microhardness share 0.047 (cross-lab-only 0.043), RMSE matched 1.163 vs unmatched 1.319 GPa, row-cal cov. matched 0.774; Tliquidus share 0.169 (cross-lab-only 0.034), RMSE matched 18.8 vs unmatched 83.3 K, row-cal cov. matched 0.988.

**Selective prediction and locally adaptive intervals** (normalised score |r|/RF tree SD; target-test (a)): RMSE reduction on the 50 % most confident (ID → target), width-stratified coverage (worst width quartile) and abstention rate (share of target candidates whose recalibrated adaptive interval is wider than the in-distribution conformal width).

| Property | RMSE reduction @50 % ID → target | adaptive source-cal cov. | adaptive recal m cov. | worst width-quartile cov. (recal) | abstention |
|---|---|---|---|---|---|
| Tg | 48.2 % → 34.2 % | 0.819 [0.775, 0.857] | 0.915 [0.844, 0.970] | 0.846 | 0.71 |
| YoungModulus | 57.8 % → 40.1 % | 0.733 [0.679, 0.804] | 0.908 [0.805, 0.995] | 0.806 | 0.76 |
| Microhardness | 55.1 % → 23.7 % | 0.808 [0.756, 0.841] | 0.914 [0.753, 0.977] | 0.844 | 0.86 |
| Tliquidus | 54.1 % → 55.8 % | 0.795 [0.713, 0.859] | 0.883 [0.779, 0.980] | 0.785 | 0.64 |


## 4. TERTIARY — nominal → chemically analysed composition (protocol + laboratory shift)

**Accuracy and marginal coverage on the target-test half (a) (same records for every method).** Coverage mean [2.5–97.5 pct]; normalised width = mean width / IQR of the target property values.

| Property | RMSE source-test → target-test | ID coverage row-cal (source-test) | (i) row-cal cov. | (ii) lab-cal cov. | weighted conformal cov. (inf. share) | (iii-a) recal m cov. | norm. width (i) / (ii) / (iii-a) | width ratio (ii)/(i), (iii-a)/(i) |
|---|---|---|---|---|---|---|---|---|
| Tg (m=30) | 29.2 → 60.2 K | 0.902 [0.881, 0.925] | 0.681 [0.630, 0.727] | 0.809 [0.745, 0.856] | 0.758 [0.707, 0.809] (0.00) | 0.905 [0.804, 0.979] | 0.81 / 1.28 / 2.19 | 1.59, 2.71 |
| YoungModulus (m=30) | 5.99 → 7.45 GPa | 0.903 [0.875, 0.938] | 0.761 [0.681, 0.836] | 0.885 [0.695, 0.975] | 0.919 [0.876, 0.963] (0.08) | 0.886 [0.771, 0.986] | 1.44 / 2.86 / 2.70 | 1.99, 1.89 |
| Microhardness (m=30) | 0.765 → 1.492 GPa | 0.903 [0.872, 0.950] | 0.786 [0.709, 0.862] | 0.872 [0.757, 0.924] | 0.853 [0.756, 0.910] (0.06) | 0.891 [0.762, 0.976] | 2.10 / 3.37 / 4.81 | 1.62, 2.28 |
| Tliquidus (m=30) | 54.4 → 70.7 K | 0.902 [0.890, 0.915] | 0.800 [0.759, 0.838] | 0.928 [0.864, 0.971] | 0.842 [0.777, 0.897] (0.04) | 0.914 [0.833, 0.972] | 0.92 / 1.65 / 1.53 | 1.78, 1.66 |

**Further metrics on target-test (a)** (interval score and WIS in property units; CE = mean |empirical − nominal| over levels 0.50–0.95; worst-cluster = lowest coverage among k-means clusters of the evaluated compositions).

| Property | method | interval score | WIS | CE mean | worst-cluster cov. |
|---|---|---|---|---|---|
| Tg | row | 394.6 | 30.8 | 0.249 | 0.155 |
| Tg | lab | 330.1 | 28.9 | 0.100 | 0.362 |
| Tg | recal m (a) | 322.9 | 28.4 | 0.067 | 0.578 |
| YoungModulus | row | 42.60 | 3.67 | 0.216 | 0.518 |
| YoungModulus | lab | 41.88 | 3.72 | 0.078 | 0.754 |
| YoungModulus | recal m (a) | 40.07 | 3.44 | 0.061 | 0.770 |
| Microhardness | row | 8.885 | 0.670 | 0.155 | 0.728 |
| Microhardness | lab | 8.355 | 0.666 | 0.064 | 0.826 |
| Microhardness | recal m (a) | 8.845 | 0.656 | 0.061 | 0.868 |
| Tliquidus | row | 361.2 | 34.3 | 0.153 | 0.512 |
| Tliquidus | lab | 358.0 | 35.2 | 0.046 | 0.729 |
| Tliquidus | recal m (a) | 348.6 | 33.1 | 0.052 | 0.669 |

**Target recalibration k-sweep on the target-test half.** (a) record-level halves (composition-grouped); (b) laboratory-grouped halves (recalibration and test records from different laboratories), recal records spread over the pool's labs or clustered (laboratory order). Coverage mean [2.5–97.5 pct]; normalised width in parentheses. k = 0 is the row-calibrated source quantile.

| Property | variant | k=0 | k=9 | k=10 | k=20 | k=30 | k=40 | headline m |
|---|---|---|---|---|---|---|---|---|
| Tg | (a) records | 0.681 [0.630, 0.727] (0.81) | 0.897 [0.714, 0.984] (2.26) | 0.911 [0.714, 0.985] (2.45) | 0.896 [0.738, 0.983] (2.21) | 0.905 [0.804, 0.979] (2.19) | 0.906 [0.804, 0.976] (2.23) | 0.905 [0.804, 0.979] (2.19; <0.90 in 0.40) |
| Tg | (b) labs, spread | 0.684 [0.608, 0.775] (0.81) | 0.898 [0.699, 0.995] (2.38) | 0.904 [0.699, 0.995] (2.44) | 0.900 [0.718, 0.994] (2.23) | 0.901 [0.793, 0.990] (2.08) | 0.895 [0.766, 0.990] (1.96) | 0.901 [0.793, 0.990] (2.08; <0.90 in 0.45) |
| Tg | (b) labs, clustered | 0.684 [0.608, 0.775] (0.81) | 0.821 [0.567, 0.983] (1.67) | 0.822 [0.567, 0.985] (1.70) | 0.805 [0.600, 0.977] (1.51) | 0.808 [0.687, 0.982] (1.41) | 0.824 [0.637, 0.977] (1.57) | 0.808 [0.687, 0.982] (1.41; <0.90 in 0.90) |
| YoungModulus | (a) records | 0.761 [0.681, 0.836] (1.44) | 0.870 [0.641, 0.995] (2.95) | 0.895 [0.736, 0.995] (3.11) | 0.879 [0.736, 0.992] (2.76) | 0.886 [0.771, 0.986] (2.70) | 0.894 [0.813, 0.971] (2.59) | 0.886 [0.771, 0.986] (2.70; <0.90 in 0.60) |
| YoungModulus | (b) labs, spread | 0.756 [0.671, 0.875] (1.44) | 0.866 [0.677, 0.989] (2.76) | 0.881 [0.720, 0.997] (2.94) | 0.883 [0.750, 0.995] (2.79) | 0.856 [0.753, 0.987] (2.44) | 0.867 [0.760, 0.982] (2.50) | 0.856 [0.753, 0.987] (2.44; <0.90 in 0.70) |
| YoungModulus | (b) labs, clustered | 0.756 [0.671, 0.875] (1.44) | 0.798 [0.484, 1.000] (2.29) | 0.818 [0.555, 1.000] (2.40) | 0.802 [0.540, 1.000] (2.28) | 0.819 [0.553, 1.000] (2.31) | 0.839 [0.707, 1.000] (2.30) | 0.819 [0.553, 1.000] (2.31; <0.90 in 0.75) |
| Microhardness | (a) records | 0.786 [0.709, 0.862] (2.10) | 0.890 [0.751, 0.990] (5.25) | 0.909 [0.799, 0.995] (6.09) | 0.900 [0.762, 0.986] (5.17) | 0.891 [0.762, 0.976] (4.81) | 0.882 [0.789, 0.962] (4.08) | 0.891 [0.762, 0.976] (4.81; <0.90 in 0.50) |
| Microhardness | (b) labs, spread | 0.792 [0.660, 0.901] (2.10) | 0.871 [0.609, 1.000] (5.65) | 0.871 [0.609, 1.000] (5.65) | 0.885 [0.627, 1.000] (5.30) | 0.909 [0.730, 1.000] (5.89) | 0.903 [0.677, 1.000] (5.31) | 0.909 [0.730, 1.000] (5.89; <0.90 in 0.35) |
| Microhardness | (b) labs, clustered | 0.792 [0.660, 0.901] (2.10) | 0.834 [0.352, 0.995] (3.95) | 0.838 [0.352, 0.995] (4.24) | 0.846 [0.559, 1.000] (5.12) | 0.881 [0.654, 1.000] (5.72) | 0.900 [0.717, 1.000] (6.62) | 0.881 [0.654, 1.000] (5.72; <0.90 in 0.40) |
| Tliquidus | (a) records | 0.800 [0.759, 0.838] (0.92) | 0.920 [0.716, 0.997] (1.88) | 0.921 [0.716, 0.997] (1.88) | 0.910 [0.805, 0.979] (1.56) | 0.914 [0.833, 0.972] (1.53) | 0.896 [0.810, 0.948] (1.37) | 0.914 [0.833, 0.972] (1.53; <0.90 in 0.45) |
| Tliquidus | (b) labs, spread | 0.791 [0.714, 0.847] (0.92) | 0.919 [0.743, 0.998] (1.94) | 0.921 [0.766, 0.998] (1.94) | 0.874 [0.708, 0.988] (1.42) | 0.884 [0.727, 0.981] (1.40) | 0.888 [0.752, 0.983] (1.40) | 0.884 [0.727, 0.981] (1.40; <0.90 in 0.50) |
| Tliquidus | (b) labs, clustered | 0.791 [0.714, 0.847] (0.92) | 0.725 [0.256, 0.979] (1.06) | 0.752 [0.256, 0.990] (1.27) | 0.791 [0.395, 0.990] (1.25) | 0.780 [0.389, 0.988] (1.15) | 0.806 [0.441, 0.982] (1.17) | 0.780 [0.389, 0.988] (1.15; <0.90 in 0.70) |

Pool sizes (records; labs in pool / test, mean): Tg (a) 473, (b) 473 (83 / 80 labs), labs in the m clustered records 5.5; YoungModulus (a) 174, (b) 172 (18 / 18 labs), labs in the m clustered records 4.3; Microhardness (a) 104, (b) 104 (17 / 21 labs), labs in the m clustered records 6.3; Tliquidus (a) 578, (b) 580 (26 / 26 labs), labs in the m clustered records 3.0.

**Per-laboratory conditional coverage** (laboratories with ≥ 20 records in the evaluated set; median over labs / share of labs with coverage < 0.80; mean over evaluations). 'exact' = share expected < 0.80 if every lab were covered at exactly 0.90 (binomial, given its n).

| Property | set | labs | row median / <0.8 | lab median / <0.8 | recal m (a) median / <0.8 | exact <0.8 |
|---|---|---|---|---|---|---|
| Tg | whole target, ≥ 20 | 12.0 | 0.82 / 0.47 | 0.97 / 0.33 | – | 0.05 |
| Tg | target-test (a), ≥ 20 | 1.4 | 0.29 / 0.82 | 0.46 / 0.82 | 0.65 / 0.62 | 0.04 |
| Tg | whole target, ≥ 10 | 27.0 | 0.84 / 0.47 | 0.97 / 0.28 | – | 0.08 |
| Tg | target-test (a), ≥ 10 | 10.7 | 0.83 / 0.44 | 0.97 / 0.29 | 1.00 / 0.21 | 0.09 |
| YoungModulus | whole target, ≥ 20 | 5.0 | 0.88 / 0.23 | 0.97 / 0.09 | – | 0.04 |
| YoungModulus | target-test (a), ≥ 20 | 1.6 | 0.83 / 0.23 | 0.93 / 0.10 | 0.94 / 0.03 | 0.05 |
| YoungModulus | whole target, ≥ 10 | 12.0 | 0.86 / 0.36 | 0.97 / 0.13 | – | 0.06 |
| YoungModulus | target-test (a), ≥ 10 | 5.5 | 0.88 / 0.28 | 0.98 / 0.11 | 0.97 / 0.12 | 0.08 |
| Microhardness | whole target, ≥ 20 | 2.0 | 0.99 / 0.00 | 1.00 / 0.00 | – | 0.05 |
| Microhardness | target-test (a), ≥ 20 | 0.0 | – / – | – / – | – / – | – |
| Microhardness | whole target, ≥ 10 | 6.0 | 0.99 / 0.23 | 1.00 / 0.17 | – | 0.06 |
| Microhardness | target-test (a), ≥ 10 | 2.0 | 0.97 / 0.06 | 0.99 / 0.04 | 0.98 / 0.04 | 0.11 |
| Tliquidus | whole target, ≥ 20 | 15.0 | 0.78 / 0.50 | 0.96 / 0.16 | – | 0.03 |
| Tliquidus | target-test (a), ≥ 20 | 6.0 | 0.79 / 0.49 | 0.96 / 0.08 | 0.93 / 0.20 | 0.03 |
| Tliquidus | whole target, ≥ 10 | 23.0 | 0.80 / 0.47 | 0.97 / 0.13 | – | 0.05 |
| Tliquidus | target-test (a), ≥ 10 | 14.9 | 0.78 / 0.51 | 0.96 / 0.15 | 0.94 / 0.23 | 0.07 |

**(iv) Lab-specific recalibration** ('a new laboratory measures 10 glasses'): per target laboratory with ≥ 20 records, k = 10 of its own records (composition-grouped; 10 draws per evaluation) recalibrate the row-cal model; coverage on the lab's remaining records (excluding the fold-level recalibration records). All methods on the same records. Median over labs / share of labs < 0.80.

| Property | labs | lab-specific k=10 | row | lab | recal m (a) | width lab-specific / row-cal (median) | lab-specific p10 over draws (median lab) |
|---|---|---|---|---|---|---|---|
| Tg | 12.0 | 0.91 / 0.01 | 0.82 / 0.47 | 0.97 / 0.33 | 1.00 / 0.24 | 1.29 | 0.79 |
| YoungModulus | 5.0 | 0.91 / 0.01 | 0.88 / 0.22 | 0.97 / 0.09 | 0.96 / 0.12 | 1.44 | 0.79 |
| Microhardness | 2.0 | 0.92 / 0.00 | 0.99 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.58 | 0.78 |
| Tliquidus | 15.0 | 0.91 / 0.01 | 0.79 / 0.50 | 0.96 / 0.15 | 0.94 / 0.23 | 1.51 | 0.81 |

**Covariate versus label/measurement shift** (A2 §2.5). Source-cal residual RMSE → the same residuals density-ratio reweighted to the target compositions (cross-fitted RF domain classifier) → actual target RMSE. Covariate share = (rw² − cal²)/(target² − cal²), a rough indicator. **Full exclusion rule** (the trimmed column): a split is dropped if (1) ESS/n < 0.05, (2) the target RMSE does not exceed the source-cal RMSE (no excess error to decompose), or (3) the reweighted RMSE exceeds the target RMSE (A2's caveat that the indicator is then meaningless). Rules (2) and (3) condition on the outcome, so the 'ESS-filter only' column (rules 1 and 2, i.e. no selection on the size of the estimate) and the per-reason exclusion counts are given alongside; the trimmed column can only be the lower of the two. A split may fail more than one condition, so the three counts do not add up to the number of splits removed (compare the n of the two share columns). 'Reweighted cov.' = coverage the row-cal quantile would have if only covariates shifted; 'actual' = row-cal coverage on the whole target.

| Property | domain AUC | ESS/n | RMSE cal → reweighted → target | covariate share, trimmed (n) | covariate share, ESS filter only: mean / median / max (n) | splits excluded: ESS / no excess / rw > target | reweighted cov. → actual cov. | novel-composition subset of target-test (a): share / row-cal / lab-cal / recal m |
|---|---|---|---|---|---|---|---|---|
| Tg | 0.939 | 0.27 | 29.5 → 35.8 → 60.0 | 0.15 (n=20 of 20) | 0.15 / 0.15 / 0.32 (n=20 of 20) | 0 / 0 / 0 | 0.863 → 0.680 | 0.99 / 0.677 / 0.806 / 0.903 |
| YoungModulus | 0.970 | 0.29 | 6.15 → 8.74 → 7.40 | 0.83 (n=2 of 20) | 20.46 / 2.40 / 309.28 (n=19 of 20) | 0 / 1 / 18 | 0.825 → 0.768 | 0.99 / 0.758 / 0.884 / 0.885 |
| Microhardness | 0.931 | 0.35 | 0.768 → 0.906 → 1.494 | 0.15 (n=20 of 20) | 0.15 / 0.13 / 0.39 (n=20 of 20) | 0 / 0 / 0 | 0.848 → 0.792 | 0.97 / 0.782 / 0.870 / 0.888 |
| Tliquidus | 0.977 | 0.10 | 53.7 → 55.6 → 71.8 | 0.08 (n=20 of 20) | 0.08 / 0.08 / 0.50 (n=20 of 20) | 0 / 0 / 0 | 0.898 → 0.797 | 0.99 / 0.798 / 0.927 / 0.913 |

Where the trimmed column rests on few splits, it is **not** evidence that the covariate share is small: for YoungModulus the reweighted residuals *exceed* the target residuals in 18 of 20 splits, so the indicator is uninformative there (reweighting over-explains the excess error rather than under-explaining it). Such properties are excluded from any range quoted in § 0 and the reason is stated there.

Identical compositions (1e-3) in the target whose composition occurs in the source training set: Tg share 0.014 (cross-lab-only 0.011), RMSE matched 23.2 vs unmatched 60.3 K, row-cal cov. matched 0.943; YoungModulus share 0.009 (cross-lab-only 0.009), RMSE matched – vs unmatched 7.43 GPa, row-cal cov. matched –; Microhardness share 0.020 (cross-lab-only 0.009), RMSE matched – vs unmatched 1.508 GPa, row-cal cov. matched –; Tliquidus share 0.011 (cross-lab-only 0.006), RMSE matched 31.5 vs unmatched 72.1 K, row-cal cov. matched 0.961.

**Selective prediction and locally adaptive intervals** (normalised score |r|/RF tree SD; target-test (a)): RMSE reduction on the 50 % most confident (ID → target), width-stratified coverage (worst width quartile) and abstention rate (share of target candidates whose recalibrated adaptive interval is wider than the in-distribution conformal width).

| Property | RMSE reduction @50 % ID → target | adaptive source-cal cov. | adaptive recal m cov. | worst width-quartile cov. (recal) | abstention |
|---|---|---|---|---|---|
| Tg | 51.2 % → 31.0 % | 0.707 [0.671, 0.737] | 0.893 [0.774, 0.968] | 0.845 | 0.92 |
| YoungModulus | 64.2 % → 21.4 % | 0.899 [0.859, 0.928] | 0.877 [0.692, 0.960] | 0.817 | 0.67 |
| Microhardness | 51.0 % → 18.0 % | 0.792 [0.752, 0.849] | 0.883 [0.795, 0.976] | 0.805 | 0.87 |
| Tliquidus | 54.8 % → 33.7 % | 0.850 [0.819, 0.889] | 0.901 [0.763, 0.981] | 0.866 | 0.68 |


## 5. Identical compositions: intra- versus inter-laboratory error

Reference = ungrouped random-row 60/20/20 split (not the protocol split; used only here). Records whose composition (1e-3 atomic fraction) occurs in the training set, split by whether a training record of that composition comes from the same laboratory. 'Lookup' = |y − mean y of the matching training records| (model-free). Leave-lab-out matches are cross-laboratory by construction.

| Property | random rows: same-lab match n / RMSE model / RMSE lookup | random rows: cross-lab-only match n / RMSE model / RMSE lookup | random rows: matched / unmatched RMSE | lab-out: matched frac / RMSE model / RMSE lookup / row-cal cov. | lab-out: unmatched RMSE / cov. | lab-out ÷ random rows, RMSE: matched / unmatched | implied lab component √(lab-out² − same-lab²) (model) |
|---|---|---|---|---|---|---|---|
| Tg | 116 / 12.5 / 9.1 | 88 / 24.1 / 26.7 | 18.4 / 32.6 | 0.056 / 24.0 / 27.3 / 0.937 | 44.7 / 0.809 | 1.30 / 1.37 | 20.4 |
| YoungModulus | 77 / 1.21 / 0.82 | 30 / 2.81 / 2.73 | 1.86 / 6.42 | 0.031 / 4.13 / 4.04 / 0.955 | 9.32 / 0.780 | 2.22 / 1.45 | 3.95 |
| Microhardness | 18 / 0.475 / 0.403 | 18 / 0.786 / 0.839 | 0.681 / 0.821 | 0.039 / 0.752 / 0.732 / 0.884 | 1.154 / 0.773 | 1.10 / 1.41 | 0.583 |
| Tliquidus | 171 / 20.9 / 8.4 | 55 / 26.1 / 21.6 | 22.4 / 55.4 | 0.032 / 28.5 / 25.6 / 0.971 | 79.9 / 0.764 | 1.27 / 1.44 | 19.3 |

Reading the last two columns together: repeating an identical composition in another laboratory already costs a large part of the error (same-lab versus cross-lab-only RMSE in the first two columns), but the *shift penalty* of holding out whole laboratories is not confined to compositions the model has already seen — relative to the ungrouped random-row reference the RMSE rises by a similar factor on matched and on unmatched records (ratios above). Inter-laboratory disagreement at fixed composition and genuine novelty therefore contribute comparably; § 0 states the claim in that form.

Ungrouped random-row reference, for comparison with the composition-grouped source-test: Tg coverage 0.900, RMSE 31.1; YoungModulus coverage 0.901, RMSE 6.00; Microhardness coverage 0.900, RMSE 0.812; Tliquidus coverage 0.898, RMSE 52.2.


## 6. SENSITIVITY — phosphorus design shift (R1) with publication-grouped recalibration

Source split, model and target region identical to `R1_core` (`common.design_split`, same seeds; the composition-grouped recalibration reproduces R1 per split, see last column). Publication-grouped: the target region is split into halves of whole publications; recalibration records (headline m) come from the pool publications and the target-test contains only other publications. 'spread' = the m records are spread over the pool's publications; 'clustered' = taken in publication order (few publications). Coverage mean [2.5–97.5 pct] (share of splits < 0.90).

| Property (m) | target pubs (Kod) | composition-grouped (R1 protocol) | test rows sharing a publication with the recal pool / with the m records | Kod-grouped, spread | Kod-grouped, clustered | Author+Year-grouped, spread | Author+Year-grouped, clustered | source-cal on Kod-grouped test | norm. width comp → Kod spread | max abs. diff. vs R1 per split |
|---|---|---|---|---|---|---|---|---|---|---|
| Tg (30) | 65 | 0.930 [0.841, 0.989] (0.20) | 0.83 / 0.50 | 0.894 [0.679, 0.992] (0.45) | 0.825 [0.495, 0.988] (0.65) | 0.894 [0.703, 0.992] (0.45) | 0.813 [0.567, 0.996] (0.65) | 0.405 [0.287, 0.523] | 2.72 → 2.64 | 0 |
| YoungModulus (22) | 15 | 0.921 [0.755, 1.000] (0.25) | 0.90 / 0.80 | 0.878 [0.483, 1.000] (0.40) | 0.847 [0.476, 1.000] (0.45) | 0.878 [0.490, 1.000] (0.40) | 0.820 [0.172, 1.000] (0.45) | 0.240 [0.047, 0.413] | 3.66 → 3.60 | 0 |
| Microhardness (23) | 11 | 0.933 [0.842, 1.000] (0.30) | 0.94 / 0.90 | 0.891 [0.578, 1.000] (0.35) | 0.830 [0.344, 1.000] (0.40) | 0.891 [0.578, 1.000] (0.35) | 0.830 [0.344, 1.000] (0.40) | 0.188 [0.000, 0.465] | 5.14 → 4.83 | 0 |
| Tliquidus (30) | 26 | 0.898 [0.752, 0.985] (0.40) | 0.87 / 0.77 | 0.828 [0.383, 1.000] (0.40) | 0.792 [0.259, 1.000] (0.45) | 0.828 [0.383, 1.000] (0.40) | 0.792 [0.259, 1.000] (0.45) | 0.386 [0.104, 0.587] | 2.83 → 2.92 | 0 |

**The two publication proxies are not always independent.** For Microhardness (11 groups under both proxies, identical target halves in 20/20 seeds); Tliquidus (26 groups under both proxies, identical target halves in 20/20 seeds) — the Kod and Author+Year partitions of the design region are the *same* partition, so the corresponding column pairs above are identical by construction, not by coincidence. For Tg (identical in 0/20 seeds; 65 Kod groups vs 61 Author+Year groups; per-split max |Kod − Author+Year| coverage difference 0.255 spread, 0.357 clustered; the two 'spread' means are 0.8935 and 0.8941, equal only after rounding to three decimals); YoungModulus (identical in 1/20 seeds; 15 Kod groups vs 14 Author+Year groups; per-split max |Kod − Author+Year| coverage difference 0.039 spread, 0.394 clustered; the two 'spread' means are 0.8785 and 0.8777, equal only after rounding to three decimals) — the splits do differ; where the two means nevertheless agree to three decimals the per-split differences cancel, which is a coincidence of rounding and not a copy error.

**Verdict on the split audit A1 (§ 6.3).** A1 reported, for the *submitted* Table-3 protocol (70/30 source, Author+Year publication proxy, whole publications drawn into the recalibration pool, 10 seeds), that publication-grouped recalibration lowers the recalibrated coverage from Tg 0.93 → 0.91, YoungModulus 0.95 → 0.91, **Microhardness 0.94 → 0.80**, Tliquidus 0.95 → 0.92, with the share of splits below 0.90 rising from Tg 20% → 30%, YoungModulus 10% → 30%, Microhardness 20% → 80%, Tliquidus 10% → 30%. Because A1 drew whole publications into the pool, the arm of this table that corresponds to it is the **clustered** ordering (the m records come from the pool's first publications); the 'spread' arm is a weaker, more favourable perturbation. Under the unified protocol of this script (composition-grouped source 60/20/20, 20 seeds, Kod proxy):

| Property | R4 composition-grouped | R4 Kod-grouped, clustered (A1's arm) | share of splits < 0.90 (comp. → clustered) | A1 (10 seeds, Author+Year, Table-3 protocol) | verdict |
|---|---|---|---|---|---|
| Tg | 0.930 [0.841, 0.989] | 0.825 [0.495, 0.988] | 0.20 → 0.65 | 0.930 → 0.911 (30% of splits < 0.90) | direction confirmed, magnitude larger |
| YoungModulus | 0.921 [0.755, 1.000] | 0.847 [0.476, 1.000] | 0.25 → 0.45 | 0.951 → 0.911 (30% of splits < 0.90) | direction confirmed, magnitude larger |
| Microhardness | 0.933 [0.842, 1.000] | 0.830 [0.344, 1.000] | 0.30 → 0.40 | 0.938 → 0.804 (80% of splits < 0.90) | direction confirmed, magnitude milder |
| Tliquidus | 0.898 [0.752, 0.985] | 0.792 [0.259, 1.000] | 0.40 → 0.45 | 0.947 → 0.923 (30% of splits < 0.90) | direction confirmed, magnitude larger |

For the property A1 singled out, microhardness, the answer is therefore: **direction confirmed, magnitude milder.** Here 0.933 [0.842, 1.000] → 0.830 [0.344, 1.000] clustered (40% of splits below 0.90) and 0.891 [0.578, 1.000] spread (35% below 0.90), against A1's 0.94 → 0.80 with 80 % of splits below 0.90. A1 used the submitted 70/30 source split, m = 23 and 10 seeds, whereas this table uses the unified protocol, so the two magnitudes are not expected to match exactly. The qualitative conclusion is A1's: recalibration records drawn from the same publications as the evaluated records overstate the recovery.


## 7. 45S5 Bioglass illustration (Tg)

Rule: |x_ox - x_ref| <= 0.01 mole fraction for SiO2 0.461, Na2O 0.244, CaO 0.269, P2O5 0.026; sum of the four >= 0.98 (as A2).

| n records | labs | publications | analysed | years | Tg min / median / max (K) | 5–95 % (K) | SD (K) | robust SD 1.4826·MAD (K) | SD of lab means (K) |
|---|---|---|---|---|---|---|---|---|---|
| 43 | 33 | 36 | 1 | 1985–2016 | 728.1 / 805.1 / 887.1 | 738.2–842.1 | 28.4 | 17.8 | 22.9 |

Model (20 seeds): the RF is trained without any record of the ±1 mol% window **and** without any record sharing a composition group with it, so none of the 45S5 records above is in the training set. This is **not** an extrapolation test: glasses chemically close to 45S5 but outside the window remain in the training data (10 further Tg records lie within ±2 mol% per oxide, and the nearest training composition is 0.012 away in summed |Δ atomic fraction|, with 25 training records within 0.05). The table gives the source-calibrated 90 % interval at the 45S5 composition. Two coverages are reported: 'own-composition prediction' centres each record's interval on the model's prediction for **that record's** composition (the records differ slightly within the window), 'nominal-centred' centres every interval on the prediction at the nominal 46.1–24.4–26.9–2.6 composition.

| Calibration | prediction at nominal 45S5 (K) | interval half-width q (K) | interval at nominal (K) | coverage of the 45S5 records (own-composition prediction) | coverage, nominal-centred | mean y − prediction (K) | RMSE on records (K) |
|---|---|---|---|---|---|---|---|
| row-cal (i.i.d.) | 797.2 [792.4, 803.6] | 46.1 [43.8, 50.4] | 751.1–843.4 | 0.890 [0.860, 0.907] | 0.873 [0.860, 0.884] | 7.1 [1.4, 12.1] | 27.4 [25.1, 29.3] |
| lab-cal (whole labs) | 799.1 [791.3, 811.4] | 71.4 [59.7, 84.8] | 727.7–870.4 | 0.952 [0.907, 0.989] | 0.944 [0.884, 0.977] | 5.7 [-3.7, 12.4] | 27.5 [24.4, 33.8] |

For comparison, ±1.645 × SD = ±46.7 K; ±1.645 × robust SD = ±29.3 K; ±1.645 × SD of lab means = ±37.7 K.


## 8. Notes and deviations

* Laboratory = normalised first-author key (A2 §2.1: ASCII-fold, surname + first initial, transliteration folding). First author is a proxy for laboratory; over-merging (e.g. common surnames) makes leave-lab-out more conservative.
* Fold assignment: labs processed roughly largest-first with a U(0.5, 1.5) jitter on size, each to the lightest fold (A2's greedy made repeats nearly identical for the large labs; the jitter makes the 4 repeats differ in which large labs are held out together). Fold sizes are listed in the JSON (`data.<prop>.A_folds`).
* Grouping: every source split and every recalibration/test split is composition-grouped (common.composition_groups, 1e-6) as the protocol requires. The laboratory-grouped source split and the laboratory/publication-grouped target halves additionally drop cal/test records whose composition occurs on the other side (counts in `sizes`). Between source and target the grouping unit is the shift variable (laboratory, period, protocol): identical compositions measured by a source laboratory and a target laboratory are kept, because they are the object of the identical-composition check; the 'novel-composition' column shows coverage on target-test records whose composition never occurs in the source.
* Recalibration uses the row-cal model (trained on the composition-grouped 60 % of the source); only the calibration residuals change. Headline m = clip(|T|/3, 10, 30) = 30 in every family. k = 9 is the smallest finite k at α = 0.1; for 9 ≤ k ≤ 18 the conformal quantile is the maximum of the k residuals.
* The (b) variant keeps recalibration and test records in different laboratories; 'spread' orders the pool by a composition-grouped permutation (the m records come from many pool labs), 'clustered' in laboratory order (the m records then come from the pool's first few laboratories, 3-6 on average; see `recal_b_clustered.n_labs_in_recal_m`).
* Weighted conformal (Tibshirani et al. 2019) uses the same cross-fitted domain classifier as the decomposition (unlabelled target compositions only). An infinite weighted quantile is counted as covered and reported as 'inf. share'; every weighted-conformal coverage in this document, including those in § 0, is therefore an upper bound on the coverage of the finite intervals.
* Covariate-share exclusion rule, in full: a split contributes to the trimmed column only if ESS/n ≥ 0.05, the target RMSE exceeds the source-cal RMSE, and the reweighted RMSE does not exceed the target RMSE. The last two conditions select on the outcome, so the untrimmed (ESS-filter-only) mean, median and maximum and the per-reason exclusion counts are printed beside it in §§ 2–4; § 0 quotes the untrimmed range.
* Calibration schemes: § 1 defines row-cal, lab-cal (which also refits the model on the laboratory-grouped 60 %, so it is not a pure calibration comparison) and recent-cal, and gives the realised source fractions, which depart from 60/20/20 because laboratories are indivisible.
* Normalised widths divide the mean width by the IQR of **all** labels of the evaluated pool, including the target-test labels. This follows PROTOCOL.md and R1 so that the numbers are comparable with R1_core; it is a scale constant, not a fitted quantity, but the split audit A1 (item 11) flagged the practice, so it is stated here. Coverage, width and interval score never use target-test labels except for evaluation.
* Several per-laboratory cells rest on very few laboratories (the laboratory count is printed in every such table, and the ≥ 10-record threshold is reported beside the ≥ 20-record one for that reason). Any per-laboratory median quoted elsewhere must be quoted with its laboratory count; in the temporal and analysed families the target-test halves are often too small for any laboratory to reach 20 records.
* The recent-cal scheme (temporal only) trains on the earliest 60 % of the source by year and calibrates on the latest 20 %; the middle 20 % is its (intermediate-period) test.
* Publications: Kod = SciGlass ID // 1e8 (as instructed). The split audit A1 used Author+Year; both are reported for the P-shift sensitivity.
* Reference random-row split: ungrouped, used only for the intra- versus inter-laboratory comparison and to show the optimism of ungrouped splitting.
* All targets are retrospective literature data: genuine cross-laboratory, cross-time and cross-protocol shifts, but not in-silico → in-vitro/in-vivo shifts.

## 9. Verifier issues and how they were handled

All issues raised in the verification of the first run were adopted; the section below records where. There is no 'Verifier issues not adopted' list.

1. *The stated covariate-share exclusion rule was not the rule in the code, and the undisclosed part selected on the outcome.* The caption in §§ 2–4 now states all three conditions, every table gives the untrimmed (ESS-filter-only) mean, median and maximum and the per-reason exclusion counts, § 0 quotes the untrimmed range, and properties whose reweighted residuals exceed the target residuals in most splits are named explicitly instead of being shown as a thin 'n = 2 of 20' cell.
2. *The Kod and Author+Year columns of § 6 are not always two independent proxies.* § 6 now states, per property, in how many seeds the two proxies induce the same target partition (identical columns are then identical by construction) and, where they differ, the per-split maximum |Kod − Author+Year| coverage difference, so that coinciding means are visibly a coincidence.
3. *The requested 'confirm or correct' verdict on A1's microhardness 0.94 → 0.80 was missing.* § 6 now quotes A1 § 6.3 in full, identifies the clustered ordering as the arm that corresponds to A1's design, and gives the verdict per property.
4. *The lab-cal arm changes the trained model, not only the calibration set.* § 1 now defines row-cal, lab-cal and recent-cal explicitly, says that lab-cal and recent-cal refit the model, and gives the realised (not nominal) source fractions.
5. *Weighted-conformal coverages were quoted in § 0 without the infinite-interval caveat.* The § 0 bullets now carry it with the maximum infinite share of that family.
6. *The claim that inter-laboratory disagreement 'not extrapolation' dominates was not supported by the table under it.* § 5 now carries the random-row matched and unmatched columns and the lab-out ÷ random-row ratios, and the claim in § 0 is restated in the form the numbers support (matched and unmatched records suffer a similar shift penalty).
7. *Accuracy slips in the covering summary of the first round* (ranges that ignored a 0.00 cell, a per-laboratory median quoted for the wrong family, and the phosphorus source-calibrated coverage quoted from the publication-grouped rather than the composition-grouped test). § 0 now prints R1's own composition-grouped source-calibrated coverage, and all ranges in this document are generated from the JSON rather than typed.
8. *Two transparency gaps*: the IQR normaliser uses all target labels (now stated in § 8, with the reference to A1 item 11), and the 45S5 model is trained without the window records but not without chemically similar glasses (now stated in § 0 and § 7, with the nearest training composition and the number of records within ±2 mol%), together with which of the two coverages is being quoted.
9. *Note only (no fix required): several per-laboratory cells rest on one or two laboratories.* The laboratory counts were already printed; a sentence in § 8 now requires them to be quoted with any median taken from this document.
