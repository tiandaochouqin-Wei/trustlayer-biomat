# R6 - genuine, naturally incomplete multi-measurement fusion

Answers Reviewer 1 comment 3 ("Figure 6 divides one composition vector into two feature groups") and Reviewer 2 comment 6 ("overly idealized ... difficult to consider representative of genuine biomedical-materials multimodal missingness"), and integrates the four trust-layer components into one decision procedure for Reviewer 2 comment 4.

What is fused: **heterogeneous, naturally incomplete characterisation measurements (density, dilatometric expansion, refractive index) reported by different laboratories** in SciGlass. This is *not* imaging, spectral or mechanical multimodality: SciGlass contains no images or spectra, and that gap remains a stated limitation. The missingness is real rather than simulated - it is what was measured, reported **and captured in SciGlass** - except in the explicitly labelled 'measurement not yet made' deployment case, where measurements that WERE captured are masked at prediction time. SciGlass is a curated compilation (A2 section 1), so an absent density is not proof that the density was never measured: it may have been reported in the source paper and not captured by the curators. What the experiment needs is only that the missingness is *real and structured*, which section 1 demonstrates by computation, not that it is a pure record of laboratory intent; the curation layer is listed as a limitation in section 8.

Protocol: `revision/code/PROTOCOL.md` and `common.py`; alpha = 0.1, finite-sample conformal quantile, 20 splits per split family, every quantity computed per split and reported as **mean [2.5, 97.5 percentile over splits]**. The splits resample one finite dataset, so the interval describes split-to-split variability, not population sampling error. Script: `revision/code/exp_R6_fusion.py`; results: `revision/results/R6_fusion.json`. No figures.


## 0. Bottom line

1. **The missingness is genuine and strongly informative.** Tg (B1) co-occurs with density, dilatometric expansion and refractive index in 8 natural patterns of 207-2385 records each; which measurements exist is a proxy for glass family and for the reporting practice of the source laboratory (MNAR), not a random mask. Computed, not asserted: a B1 laboratory with at least 20 records carries 84 % of them in a single missing pattern, against 34 % when the patterns are permuted (section 1).

2. **Fusing real auxiliary measurements buys very little accuracy.** RMSE, composition-only -> fused HGB: B1 30.35 -> 29.34 K on the composition-grouped random split and 42.76 -> 42.85 K across held-out laboratories; B2 5.57 -> 5.36 GPa and 10.26 -> 9.97 GPa. The RF-with-imputation family behaves the same way. They buy almost nothing in decisions either: in the end-to-end procedure of section 7 (1525 candidates, random split), 1873 already-existing auxiliary values save 20 of 686 Tg measurements. The reportable effect of the auxiliaries is not accuracy, and not measurement savings; it is what an honest uncertainty statement has to look like when records are characterised to different depths.

3. **Pooled conformal hides pattern-conditional miscoverage; Mondrian-by-pattern narrows the spread of per-pattern coverage on i.i.d.-style splits, and does not repair it across held-out laboratories.** Three distinct statistics are involved and all three are given on both split families, so that they are never mixed. Over patterns with >= 20 test records: (a) the range of the MEAN per-pattern coverage (average each pattern over the splits first, then take max - min: the stable statistic, but it has no per-split value and therefore no paired test); (b) the mean over splits of the per-split RANGE (max - min within a split, so it carries the binomial noise of the small patterns, but it can be tested pairwise); (c) the mean over splits of the per-split WORST-PATTERN MINIMUM, the noisiest of the three. (b) and (c) are always quoted with their percentile interval and their paired split-to-split difference. (a) and (b) need not move in the same direction - (b) is a max - min of noisy per-split estimates, (a) averages each pattern first - which is exactly why they are named apart and never substituted for one another.

   (a) Range of the mean per-pattern coverage, fused model, pooled -> Mondrian: B1 random split 0.838 (nD) to 0.946 (rho+nD) -> 0.885 (rho) to 0.922 (rho+CTE+nD); B1 laboratory-grouped 0.855 (CTE+nD) to 0.952 (rho+nD) -> 0.843 (rho+CTE+nD) to 0.927 (CTE); B2 random split 0.738 (comp-only) to 0.949 (rho+CTE) -> 0.880 (comp-only) to 0.918 (CTE); B2 laboratory-grouped 0.751 (comp-only) to 0.912 (rho+CTE) -> 0.774 (CTE+Tg) to 0.926 (CTE).

   (b) Mean per-split range, pooled -> Mondrian: B1 random 0.152 -> 0.145 (paired difference -0.006, SD 0.049, 12 of 20 splits worse / 8 better, Wilcoxon p = 0.956); B1 laboratory 0.214 -> 0.262 (paired difference +0.048, SD 0.126, 12 of 20 splits worse / 8 better, Wilcoxon p = 0.097); B2 random 0.229 -> 0.135 (paired difference -0.094, SD 0.067, 2 of 20 splits worse / 18 better, Wilcoxon p < 0.001); B2 laboratory 0.262 -> 0.322 (paired difference +0.060, SD 0.138, 14 of 20 splits worse / 5 better, Wilcoxon p = 0.044). On both random splits this statistic narrows; on both laboratory-grouped splits it widens, i.e. the pattern-conditional quantiles make the per-split spread of coverage worse, not better, once the laboratories are held out.

   (c) Worst-pattern minimum, pooled -> Mondrian: B1 random 0.812 [0.718, 0.871] -> 0.820 [0.744, 0.871] (paired difference +0.008, SD 0.041, 12 of 20 splits better / 7 worse, Wilcoxon p = 0.421); B1 laboratory 0.765 [0.515, 0.903] -> 0.724 [0.500, 0.896] (paired difference -0.041, SD 0.125, 8 of 20 splits better / 12 worse, Wilcoxon p = 0.154); B2 random 0.733 [0.650, 0.822] -> 0.826 [0.737, 0.890] (paired difference +0.092, SD 0.053, 19 of 20 splits better / 1 worse, Wilcoxon p < 0.001); B2 laboratory 0.705 [0.491, 0.961] -> 0.670 [0.355, 0.918] (paired difference -0.035, SD 0.136, 9 of 20 splits better / 10 worse, Wilcoxon p = 0.243). The sampling-noise reference for this statistic is 0.846 (B1 laboratory split).

   Read together: the worst-pattern minimum moves significantly (paired Wilcoxon p < 0.05) in 1 of the 4 design x split families (B2_E, composition-grouped random split); everywhere else it is inside split noise, in either direction. The per-split range moves significantly in B2_E, composition-grouped random split (narrows), B2_E, laboratory-grouped split (widens). **On the laboratory-grouped families the evidence supports *'Mondrian does not repair per-pattern coverage across laboratories'*.** It does not support a claim that Mondrian measurably lowers the worst-pattern minimum there (both paired tests are far from significant); the only significant laboratory-split movement of any of the three statistics is the widening of B2's per-split range, which points the same way as the absence of a repair.

   The mechanism is nevertheless visible per pattern. Mondrian narrows the interval of a well-characterised stratum (B1, rho+CTE: 98 K versus 135 K pooled), and under a laboratory shift that narrower interval then under-covers (0.868 versus 0.940). Pattern-conditional calibration is not a substitute for handling the cross-laboratory shift.

4. **The width differences between patterns are population differences, not information gain.** For B1's best-characterised pattern (rho+CTE) the fused intervals are 0.60x the width of composition-only glasses, but a model that never sees an auxiliary value already gives 0.64x on the same glasses; the measured values themselves contribute only 0.92x. Across held-out laboratories the population term accounts for the width difference entirely (information ratio 1.00 [0.86, 1.15]), so this item is the one place where the laboratory-grouped split strengthens rather than weakens the statement. The paper must not say that the auxiliary measurements 'halve' the uncertainty (section 5).

5. **'Not yet measured' is a different regime from 'naturally missing', and it is fixable.** Masking every candidate's auxiliaries at prediction time costs the native-NaN model 33.2 K versus 30.4 K for a composition-only model on the same glasses; modality-dropout training removes the penalty (30.6 K). Every calibration number in this item is for the **modality-dropout model**, on B1's random split. Calibrating on identically masked calibration records gives 0.897 coverage at 88 K; borrowing the quantile of naturally composition-only glasses gives 0.934 at 110 K (valid but needlessly wide); reusing the pooled natural quantile gives 0.892 at 86 K for the same model, and 0.860 for the native-NaN model, whose accuracy the masking does degrade - the naive recipe is the one that turns a model-quality problem into an invalid interval. The calibration set must reproduce the deployment-time missingness, not the historical one (full table in section 6).

6. **End to end, the layer's guarantee is marginal, and it degrades exactly where the data stop being exchangeable.** With the specification Tg >= median, interval-based acceptance keeps the overall false-accept proportion at 0.014 (random split), but per pattern it reaches 0.273; conformal selection was tested against its nominal q = 0.10 on both split families with a one-sided t over the splits (JSON `fdp_vs_q`); the verdicts below are that test, not point estimates. Composition-grouped random split - guarantee upheld: realised FDP 0.085-0.097 as a ratio of totals over the 6 threshold x scope combinations; **none of the 6 is above q = 0.10** by a one-sided t over the splits. Laboratory-grouped split - guarantee violated: realised FDP 0.112-0.235 as a ratio of totals over the 6 threshold x scope combinations; **4 of 6 are above q = 0.10** by a one-sided t over the splits, the strongest being *conformal selection, BH inside each pattern, tau = q75*: 0.223 [0.091, 0.385] per split, 0.235 as a ratio of totals, one-sided t vs q = 0.10 p < 0.001; *conformal selection, Mondrian p-values, one BH, tau = q75*: 0.176 [0.036, 0.300] per split, 0.201 as a ratio of totals, one-sided t vs q = 0.10 p < 0.001. Per pattern the realised FDP reaches 0.545 in one pattern (laboratory split). Section 7 and section 8 give every arm.


## 1. Designs, measurements and natural missing patterns


**B1_Tg** - target Tg (7623 records, 7025 unique compositions, 756 laboratories, 1271 publications), inputs = 25 element atomic fractions + auxiliaries Density293K (3699 records), CTEbelowTg (4629 records), RefractiveIndex (1019 records). IQR(Tg) = 115 K.

| pattern (what was measured) | n | share | Tg mean +/- SD | labs | publications | median year |
|---|---|---|---|---|---|---|
| rho+CTE | 2385 | 31.3 % | 870.6 +/- 74.5 | 162 | 303 | 2005 |
| comp-only | 1807 | 23.7 % | 872.1 +/- 91.5 | 314 | 408 | 2008 |
| CTE | 1681 | 22.1 % | 833.3 +/- 74 | 214 | 315 | 1999 |
| rho | 731 | 9.6 % | 858.7 +/- 79.3 | 115 | 144 | 2012 |
| rho+CTE+nD | 356 | 4.7 % | 847.1 +/- 85.2 | 49 | 69 | 2000 |
| nD | 229 | 3.0 % | 802.9 +/- 55.8 | 37 | 51 | 2010 |
| rho+nD | 227 | 3.0 % | 810.2 +/- 50.7 | 18 | 21 | 2011 |
| CTE+nD | 207 | 2.7 % | 834.6 +/- 97.7 | 24 | 31 | 2002 |

Auxiliary screen (counts and Pearson r recomputed here; verdicts fixed a priori, A2 section 4.1):

| candidate auxiliary | co-occurring records | r with target | verdict |
|---|---|---|---|
| Density293K | 3699 | -0.12 | used |
| CTEbelowTg | 4629 | -0.38 | used (dilatometry; the same run can also give a dilatometric Tg) |
| RefractiveIndex | 1019 | -0.07 | used |
| YoungModulus | 1473 | +0.41 | not used (4th modality makes patterns sparse) |
| Microhardness | 741 | +0.37 | not used (sparse) |
| CTE433K | 1466 | -0.44 | not used (alternative CTE definition) |
| CTE483K | 1599 | -0.51 | not used (alternative CTE definition) |
| Tliquidus | 1388 | +0.62 | not used (another characteristic temperature) |
| CrystallizationPeak | 905 | +0.73 | excluded (same DTA/DSC thermogram as Tg) |
| CrystallizationOnset | 383 | +0.80 | excluded (same DTA/DSC thermogram as Tg) |
| TAnnealing | 125 | +0.91 | excluded (viscosity iso-point, near-label leakage) |
| Tstrain | 230 | +0.97 | excluded (viscosity iso-point, near-label leakage) |
| T12 | 230 | +0.97 | excluded (Tg ~ 10^12 Pa s temperature, near-label leakage) |
| T11 | 309 | +0.77 | excluded (viscosity iso-point, near-label leakage) |
| Tsoft | 404 | +0.80 | excluded (same viscosity curve) |
| TdilatometricSoftening | 1115 | +0.82 | excluded (same dilatometric curve) |
| TLittletons | 207 | +0.80 | excluded (same viscosity curve) |


**B2_E** - target YoungModulus (3812 records, 3444 unique compositions, 259 laboratories, 431 publications), inputs = 25 element atomic fractions + auxiliaries Density293K (3053 records), CTEbelowTg (2276 records), Tg (1473 records). IQR(YoungModulus) = 15.2 GPa.

| pattern (what was measured) | n | share | YoungModulus mean +/- SD | labs | publications | median year |
|---|---|---|---|---|---|---|
| rho+CTE | 981 | 25.7 % | 74.09 +/- 9.88 | 61 | 101 | 2013 |
| rho+CTE+Tg | 956 | 25.1 % | 78.37 +/- 10.1 | 62 | 102 | 2012 |
| rho | 884 | 23.2 % | 86.36 +/- 14.1 | 69 | 107 | 2012 |
| comp-only | 336 | 8.8 % | 80.47 +/- 19.7 | 54 | 65 | 2008 |
| rho+Tg | 232 | 6.1 % | 77.82 +/- 12 | 36 | 43 | 2013 |
| CTE+Tg | 201 | 5.3 % | 78.66 +/- 11.1 | 23 | 32 | 2003 |
| CTE | 138 | 3.6 % | 80.11 +/- 8.91 | 31 | 36 | 2003 |
| Tg | 84 | 2.2 % | 83.58 +/- 9.88 | 13 | 13 | 2000 |

Auxiliary screen (counts and Pearson r recomputed here; verdicts fixed a priori, A2 section 4.1):

| candidate auxiliary | co-occurring records | r with target | verdict |
|---|---|---|---|
| Density293K | 3053 | +0.45 | used |
| CTEbelowTg | 2276 | -0.12 | used |
| Tg | 1473 | +0.41 | used |
| ShearModulus | 1105 | +0.87 | excluded (elastically coupled to E) |
| PoissonRatio | 1105 | +0.15 | excluded (usually derived from E and G) |
| RefractiveIndex | 199 | +0.66 | not used (too few co-occurring records) |
| Microhardness | 655 | +0.32 | not used (not in the A2 design) |

The patterns are strongly non-random (MNAR): pattern-specific target means differ by up to 69 K for B1. That the pattern also tracks who produced the record is **computed here, not asserted**: for every laboratory or publication with at least 20 records, the share of its records carried by its single most frequent missing pattern, against the same statistic after permuting the pattern labels over all records (50 permutations).

| design | unit | groups | records in them | mean share of the modal pattern | groups >= 80 % one pattern | permuted baseline |
|---|---|---|---|---|---|---|
| B1_Tg | laboratories (>= 20 records) | 90 | 4382 | 84.3 % | 63.3 % | 34.3 % |
| B1_Tg | publications (>= 20 records) | 68 | 2310 | 98.0 % | 95.6 % | 35.0 % |
| B2_E | laboratories (>= 20 records) | 44 | 2701 | 89.7 % | 81.8 % | 31.7 % |
| B2_E | publications (>= 20 records) | 44 | 1897 | 95.3 % | 95.5 % | 32.0 % |

A laboratory therefore reports one characterisation set and stays with it, and a single publication almost always reports exactly one - which is why the missing pattern behaves like a grouping variable and why pattern-conditional calibration matters below. (A2 section 4.2 illustrates the same point with patent and industrial series reporting density and expansion and optical papers reporting the refractive index. That attribution is an illustration from the A2 scoping study; it is **not** recomputed here. A2's corporate-name string pattern, re-applied to B1, matches 74 records resolving to 1 normalised author key, which is far too thin to establish the attribution, so this report does not use it as evidence.)


## 2. Splits and the dependence they do (not) remove

| design | split family | train / cal / test | test records whose laboratory also occurs in cal | ... in train | test records whose publication occurs in cal | test laboratories |
|---|---|---|---|---|---|---|
| B1_Tg | composition-grouped random | 4574 / 1524 / 1525 | 0.87 [0.85, 0.89] | 0.96 [0.95, 0.97] | 0.77 [0.75, 0.79] | 443 [421, 458] |
| B1_Tg | laboratory-grouped | 4472 / 1593 / 1558 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 150 [62, 218] |
| B2_E | composition-grouped random | 2287 / 763 / 762 | 0.91 [0.90, 0.94] | 0.98 [0.96, 0.98] | 0.85 [0.84, 0.87] | 164 [155, 173] |
| B2_E | laboratory-grouped | 2155 / 804 / 853 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 62 [28, 87] |

In the composition-grouped random split a composition never straddles a split, but most test records still share a laboratory and a publication with calibration records - the dependence the A1 audit (section 6.3) found between recalibration and test records. The laboratory-grouped split removes it completely (the grouping unit is a connected component of the laboratory-composition graph, so laboratories that published an identical composition are held out together; largest unit 15 % of B1 records). **Which split a number comes from is stated everywhere, and every table reports both.** The laboratory-grouped family is the deployment-realistic one and carries the negative conclusions (sections 3, 7, 8); the random split is the optimistic bound and is where the positive calibration effect of section 3 and the trust-layer arms of section 7 are legible at all. Section 5's width decomposition is quoted from the random split in section 0 because that is the conservative direction for its claim - the laboratory-grouped split makes the 'information' contribution of the measurements smaller still, not larger.

Mondrian strata: a pattern with fewer than 30 calibration records is merged into its parent pattern with one fewer measurement (rule fixed a priori, calibration counts only). Merges actually applied, over all splits:

| design | split family | merges (child -> parent: number of splits out of 20) |
|---|---|---|
| B1_Tg | composition-grouped random | CTE+nD->CTE: 1 |
| B1_Tg | laboratory-grouped | CTE+nD->CTE: 3, nD->comp-only: 10, rho+CTE+nD->rho+CTE: 1, rho+nD->rho: 10 |
| B2_E | composition-grouped random | CTE->comp-only: 13, Tg->comp-only: 20 |
| B2_E | laboratory-grouped | CTE+Tg->CTE: 8, CTE+Tg->Tg: 1, CTE->comp-only: 10, Tg->comp-only: 14, rho+CTE->rho: 1, rho+Tg->rho: 5 |

After a merge the coverage statement is marginal over the merged stratum, not conditional on the pattern; the per-pattern tables name the stratum each pattern used.

One further case has to be recorded: if the merged composition-only ROOT itself ends up with fewer than 30 calibration records, its members (composition-only plus everything merged into it) fall back to the pooled quantile, so that split's 'Mondrian' arm is partly pooled. This happened in 1 of 20 B2_E laboratory-grouped splits (whole laboratory-composition units can strip the root); everywhere else the root kept its own quantile. The occurrence is in the JSON as `strata.root_pooled_fallback`.


## 3. Accuracy and calibration, pooled versus Mondrian-by-pattern


**B1_Tg, composition-grouped random split** (20 splits)

| model | RMSE (K) | pooled cov. | pooled width (K) | Mondrian cov. | Mondrian width (K) | pooled interval score | Mondrian interval score | mean per-SPLIT per-pattern cov. range, pooled -> Mondrian |
|---|---|---|---|---|---|---|---|---|
| HGB composition-only | 30.35 [29.17, 32.63] | 0.898 [0.882, 0.909] | 88.4 [83.2, 93.6] | 0.901 [0.877, 0.912] | 89.9 [85.9, 93.2] | 148.9 | 147.0 | 0.135 -> 0.129 |
| HGB fused (native NaN) | 29.34 [28.15, 31.60] | 0.896 [0.878, 0.910] | 84.4 [80.9, 89.4] | 0.899 [0.879, 0.916] | 86.4 [81.2, 92.1] | 144.4 | 142.3 | 0.152 -> 0.145 |
| HGB fused + modality dropout | 30.10 [28.73, 32.69] | 0.896 [0.878, 0.912] | 86.4 [81.4, 91.3] | 0.900 [0.878, 0.911] | 87.9 [84.1, 92.1] | 147.3 | 145.0 | 0.158 -> 0.169 |
| RF composition-only | 31.95 [30.24, 34.26] | 0.897 [0.877, 0.908] | 91.6 [85.1, 97.7] | 0.898 [0.875, 0.910] | 92.5 [88.1, 98.7] | 158.8 | 156.8 | 0.157 -> 0.153 |
| RF fused (median imputation + indicators) | 31.48 [29.76, 33.63] | 0.896 [0.880, 0.909] | 89.2 [81.9, 97.1] | 0.897 [0.878, 0.908] | 90.4 [86.1, 96.8] | 156.7 | 154.2 | 0.155 -> 0.153 |

The last column is the max - min of per-pattern coverage **within a split**, averaged over splits (so it carries the binomial noise of the small patterns). The range of the per-pattern coverage MEANS, which averages first and is therefore smaller, is in section 0 item 3(a) and in the per-pattern tables of section 4; the two can move in opposite directions and are never quoted interchangeably. Paired test of the per-split range (Mondrian minus pooled), fused HGB: paired difference -0.006, SD 0.049, 12 of 20 splits worse / 8 better, Wilcoxon p = 0.956.

Worst-pattern coverage for the same models - the noisiest statistic in this report, so it is given with its percentile interval and with the PAIRED split-to-split difference (Mondrian minus pooled over the same 20 splits):

| model | worst-pattern cov. pooled | worst-pattern cov. Mondrian | paired difference |
|---|---|---|---|
| HGB composition-only | 0.832 [0.763, 0.871] | 0.837 [0.726, 0.889] | paired difference +0.004, SD 0.045, 10 of 20 splits better / 7 worse, Wilcoxon p = 0.435 |
| HGB fused (native NaN) | 0.812 [0.718, 0.871] | 0.820 [0.744, 0.871] | paired difference +0.008, SD 0.041, 12 of 20 splits better / 7 worse, Wilcoxon p = 0.421 |
| HGB fused + modality dropout | 0.811 [0.707, 0.869] | 0.801 [0.606, 0.889] | paired difference -0.010, SD 0.077, 9 of 20 splits better / 9 worse, Wilcoxon p = 0.744 |
| RF composition-only | 0.812 [0.741, 0.853] | 0.814 [0.731, 0.869] | paired difference +0.002, SD 0.046, 12 of 20 splits better / 8 worse, Wilcoxon p = 0.756 |
| RF fused (median imputation + indicators) | 0.808 [0.732, 0.854] | 0.811 [0.715, 0.881] | paired difference +0.003, SD 0.058, 10 of 20 splits better / 8 worse, Wilcoxon p = 0.616 |

Reference for those two tables: with these calibration and test counts, exact calibration would give a worst-pattern coverage of 0.850 (pooled) and 0.836 (Mondrian) by sampling noise alone; restricted to patterns with >= 50 test records the observed values are 0.840 -> 0.858 for the fused model against a reference of 0.858 (paired difference +0.017, SD 0.027, 15 of 20 splits better / 4 worse, Wilcoxon p = 0.016). A worst-pattern difference whose paired Wilcoxon p is well above 0.05 is split noise and must not be read as a measured improvement or a measured degradation; the coverage-range column above is the stable statistic.


**B1_Tg, laboratory-grouped split** (20 splits)

| model | RMSE (K) | pooled cov. | pooled width (K) | Mondrian cov. | Mondrian width (K) | pooled interval score | Mondrian interval score | mean per-SPLIT per-pattern cov. range, pooled -> Mondrian |
|---|---|---|---|---|---|---|---|---|
| HGB composition-only | 42.76 [31.22, 50.14] | 0.901 [0.826, 0.973] | 134.9 [107.2, 164.7] | 0.896 [0.836, 0.957] | 140.4 [100.7, 174.5] | 207.9 | 213.9 | 0.214 -> 0.255 |
| HGB fused (native NaN) | 42.85 [30.77, 49.54] | 0.904 [0.828, 0.969] | 134.7 [108.6, 155.3] | 0.895 [0.844, 0.951] | 138.7 [106.9, 170.6] | 207.1 | 211.3 | 0.214 -> 0.262 |
| HGB fused + modality dropout | 42.31 [30.35, 50.20] | 0.904 [0.848, 0.969] | 133.3 [107.9, 169.7] | 0.899 [0.861, 0.959] | 138.0 [104.0, 177.0] | 204.9 | 209.9 | 0.220 -> 0.252 |
| RF composition-only | 45.59 [35.45, 54.33] | 0.902 [0.853, 0.960] | 145.1 [119.1, 174.7] | 0.891 [0.805, 0.958] | 148.5 [109.6, 186.1] | 219.4 | 227.8 | 0.256 -> 0.285 |
| RF fused (median imputation + indicators) | 45.44 [35.45, 54.23] | 0.907 [0.865, 0.961] | 146.0 [123.0, 171.9] | 0.892 [0.841, 0.938] | 149.0 [115.9, 184.7] | 217.9 | 226.1 | 0.251 -> 0.322 |

The last column is the max - min of per-pattern coverage **within a split**, averaged over splits (so it carries the binomial noise of the small patterns). The range of the per-pattern coverage MEANS, which averages first and is therefore smaller, is in section 0 item 3(a) and in the per-pattern tables of section 4; the two can move in opposite directions and are never quoted interchangeably. Paired test of the per-split range (Mondrian minus pooled), fused HGB: paired difference +0.048, SD 0.126, 12 of 20 splits worse / 8 better, Wilcoxon p = 0.097.

Worst-pattern coverage for the same models - the noisiest statistic in this report, so it is given with its percentile interval and with the PAIRED split-to-split difference (Mondrian minus pooled over the same 20 splits):

| model | worst-pattern cov. pooled | worst-pattern cov. Mondrian | paired difference |
|---|---|---|---|
| HGB composition-only | 0.758 [0.455, 0.945] | 0.721 [0.371, 0.893] | paired difference -0.037, SD 0.177, 9 of 20 splits better / 11 worse, Wilcoxon p = 0.596 |
| HGB fused (native NaN) | 0.765 [0.515, 0.903] | 0.724 [0.500, 0.896] | paired difference -0.041, SD 0.125, 8 of 20 splits better / 12 worse, Wilcoxon p = 0.154 |
| HGB fused + modality dropout | 0.754 [0.489, 0.926] | 0.731 [0.342, 0.889] | paired difference -0.024, SD 0.180, 8 of 20 splits better / 10 worse, Wilcoxon p = 0.913 |
| RF composition-only | 0.720 [0.403, 0.939] | 0.697 [0.435, 0.862] | paired difference -0.023, SD 0.158, 9 of 20 splits better / 7 worse, Wilcoxon p = 0.918 |
| RF fused (median imputation + indicators) | 0.728 [0.395, 0.924] | 0.661 [0.355, 0.862] | paired difference -0.067, SD 0.182, 9 of 20 splits better / 8 worse, Wilcoxon p = 0.554 |

Reference for those two tables: with these calibration and test counts, exact calibration would give a worst-pattern coverage of 0.855 (pooled) and 0.846 (Mondrian) by sampling noise alone; restricted to patterns with >= 50 test records the observed values are 0.824 -> 0.786 for the fused model against a reference of 0.860 (paired difference -0.038, SD 0.053, 4 of 20 splits better / 13 worse, Wilcoxon p = 0.005). A worst-pattern difference whose paired Wilcoxon p is well above 0.05 is split noise and must not be read as a measured improvement or a measured degradation; the coverage-range column above is the stable statistic.


**B2_E, composition-grouped random split** (20 splits)

| model | RMSE (GPa) | pooled cov. | pooled width (GPa) | Mondrian cov. | Mondrian width (GPa) | pooled interval score | Mondrian interval score | mean per-SPLIT per-pattern cov. range, pooled -> Mondrian |
|---|---|---|---|---|---|---|---|---|
| HGB composition-only | 5.57 [4.89, 6.75] | 0.897 [0.872, 0.923] | 13.9 [12.7, 15.3] | 0.899 [0.876, 0.913] | 14.6 [12.4, 16.8] | 27.7 | 26.9 | 0.214 -> 0.157 |
| HGB fused (native NaN) | 5.36 [4.65, 6.45] | 0.898 [0.875, 0.926] | 13.1 [11.9, 15.1] | 0.898 [0.877, 0.921] | 13.5 [11.6, 15.5] | 26.6 | 25.5 | 0.229 -> 0.135 |
| HGB fused + modality dropout | 5.48 [4.73, 6.79] | 0.899 [0.879, 0.923] | 13.1 [11.7, 14.6] | 0.899 [0.881, 0.916] | 13.8 [12.4, 14.9] | 27.1 | 26.3 | 0.219 -> 0.156 |
| RF composition-only | 6.00 [5.00, 7.24] | 0.902 [0.880, 0.922] | 13.7 [12.5, 15.5] | 0.902 [0.886, 0.916] | 14.7 [12.9, 16.8] | 29.6 | 28.9 | 0.198 -> 0.142 |
| RF fused (median imputation + indicators) | 5.82 [4.80, 6.97] | 0.901 [0.890, 0.929] | 13.1 [11.4, 14.8] | 0.902 [0.888, 0.919] | 14.1 [12.3, 16.6] | 28.6 | 27.7 | 0.232 -> 0.138 |

The last column is the max - min of per-pattern coverage **within a split**, averaged over splits (so it carries the binomial noise of the small patterns). The range of the per-pattern coverage MEANS, which averages first and is therefore smaller, is in section 0 item 3(a) and in the per-pattern tables of section 4; the two can move in opposite directions and are never quoted interchangeably. Paired test of the per-split range (Mondrian minus pooled), fused HGB: paired difference -0.094, SD 0.067, 2 of 20 splits worse / 18 better, Wilcoxon p < 0.001.

Worst-pattern coverage for the same models - the noisiest statistic in this report, so it is given with its percentile interval and with the PAIRED split-to-split difference (Mondrian minus pooled over the same 20 splits):

| model | worst-pattern cov. pooled | worst-pattern cov. Mondrian | paired difference |
|---|---|---|---|
| HGB composition-only | 0.744 [0.648, 0.836] | 0.806 [0.697, 0.884] | paired difference +0.062, SD 0.060, 16 of 20 splits better / 3 worse, Wilcoxon p = 0.001 |
| HGB fused (native NaN) | 0.733 [0.650, 0.822] | 0.826 [0.737, 0.890] | paired difference +0.092, SD 0.053, 19 of 20 splits better / 1 worse, Wilcoxon p < 0.001 |
| HGB fused + modality dropout | 0.737 [0.672, 0.808] | 0.809 [0.675, 0.879] | paired difference +0.072, SD 0.064, 18 of 20 splits better / 2 worse, Wilcoxon p < 0.001 |
| RF composition-only | 0.764 [0.698, 0.836] | 0.830 [0.739, 0.893] | paired difference +0.066, SD 0.030, 20 of 20 splits better / 0 worse, Wilcoxon p < 0.001 |
| RF fused (median imputation + indicators) | 0.734 [0.637, 0.820] | 0.832 [0.729, 0.893] | paired difference +0.099, SD 0.039, 20 of 20 splits better / 0 worse, Wilcoxon p < 0.001 |

Reference for those two tables: with these calibration and test counts, exact calibration would give a worst-pattern coverage of 0.843 (pooled) and 0.832 (Mondrian) by sampling noise alone; restricted to patterns with >= 50 test records the observed values are 0.738 -> 0.849 for the fused model against a reference of 0.863 (paired difference +0.111, SD 0.052, 20 of 20 splits better / 0 worse, Wilcoxon p < 0.001). A worst-pattern difference whose paired Wilcoxon p is well above 0.05 is split noise and must not be read as a measured improvement or a measured degradation; the coverage-range column above is the stable statistic.


**B2_E, laboratory-grouped split** (20 splits)

| model | RMSE (GPa) | pooled cov. | pooled width (GPa) | Mondrian cov. | Mondrian width (GPa) | pooled interval score | Mondrian interval score | mean per-SPLIT per-pattern cov. range, pooled -> Mondrian |
|---|---|---|---|---|---|---|---|---|
| HGB composition-only | 10.26 [6.12, 14.55] | 0.874 [0.749, 0.975] | 25.1 [14.1, 36.6] | 0.878 [0.798, 0.957] | 32.1 [22.8, 45.2] | 52.3 | 57.8 | 0.264 -> 0.326 |
| HGB fused (native NaN) | 9.97 [6.01, 14.15] | 0.881 [0.743, 0.970] | 24.7 [14.5, 35.5] | 0.877 [0.796, 0.952] | 31.5 [22.1, 46.0] | 50.5 | 56.2 | 0.262 -> 0.322 |
| HGB fused + modality dropout | 10.10 [6.19, 14.15] | 0.878 [0.737, 0.967] | 25.1 [13.8, 39.9] | 0.880 [0.803, 0.944] | 31.5 [20.9, 44.5] | 51.5 | 56.9 | 0.245 -> 0.310 |
| RF composition-only | 10.08 [5.58, 14.19] | 0.877 [0.752, 0.972] | 24.9 [14.9, 34.2] | 0.876 [0.779, 0.958] | 32.0 [22.6, 46.0] | 51.8 | 57.8 | 0.250 -> 0.312 |
| RF fused (median imputation + indicators) | 9.76 [5.53, 14.05] | 0.880 [0.747, 0.970] | 23.5 [14.5, 30.9] | 0.884 [0.783, 0.961] | 31.0 [19.8, 47.3] | 49.8 | 55.8 | 0.258 -> 0.294 |

The last column is the max - min of per-pattern coverage **within a split**, averaged over splits (so it carries the binomial noise of the small patterns). The range of the per-pattern coverage MEANS, which averages first and is therefore smaller, is in section 0 item 3(a) and in the per-pattern tables of section 4; the two can move in opposite directions and are never quoted interchangeably. Paired test of the per-split range (Mondrian minus pooled), fused HGB: paired difference +0.060, SD 0.138, 14 of 20 splits worse / 5 better, Wilcoxon p = 0.044.

Worst-pattern coverage for the same models - the noisiest statistic in this report, so it is given with its percentile interval and with the PAIRED split-to-split difference (Mondrian minus pooled over the same 20 splits):

| model | worst-pattern cov. pooled | worst-pattern cov. Mondrian | paired difference |
|---|---|---|---|
| HGB composition-only | 0.699 [0.405, 0.958] | 0.669 [0.374, 0.906] | paired difference -0.030, SD 0.102, 8 of 20 splits better / 11 worse, Wilcoxon p = 0.159 |
| HGB fused (native NaN) | 0.705 [0.491, 0.961] | 0.670 [0.355, 0.918] | paired difference -0.035, SD 0.136, 9 of 20 splits better / 10 worse, Wilcoxon p = 0.243 |
| HGB fused + modality dropout | 0.717 [0.478, 0.951] | 0.680 [0.382, 0.881] | paired difference -0.037, SD 0.109, 8 of 20 splits better / 12 worse, Wilcoxon p = 0.165 |
| RF composition-only | 0.718 [0.401, 0.958] | 0.678 [0.346, 0.923] | paired difference -0.040, SD 0.074, 5 of 20 splits better / 12 worse, Wilcoxon p = 0.031 |
| RF fused (median imputation + indicators) | 0.712 [0.461, 0.948] | 0.699 [0.342, 0.914] | paired difference -0.012, SD 0.116, 10 of 20 splits better / 8 worse, Wilcoxon p = 0.557 |

Reference for those two tables: with these calibration and test counts, exact calibration would give a worst-pattern coverage of 0.858 (pooled) and 0.846 (Mondrian) by sampling noise alone; restricted to patterns with >= 50 test records the observed values are 0.716 -> 0.678 for the fused model against a reference of 0.854 (paired difference -0.038, SD 0.130, 7 of 20 splits better / 12 worse, Wilcoxon p = 0.184). A worst-pattern difference whose paired Wilcoxon p is well above 0.05 is split noise and must not be read as a measured improvement or a measured degradation; the coverage-range column above is the stable statistic.


Full metric set (Reviewer 2 comment 12) for the fused HGB model:

| design | split | calibration | coverage | width | width / IQR | interval score | WIS | calib. error (mean) | worst width-bin cov. | worst k-means cluster cov. | rel. AURC | RMSE reduction on 50 % most confident |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_Tg | comp | pooled | 0.896 [0.878, 0.910] | 84.4 | 0.73 | 144.4 | 12.8 | 0.011 | 0.896 | 0.749 | n/a (constant width) | n/a (constant width) |
| B1_Tg | comp | mondrian | 0.899 [0.879, 0.916] | 86.4 | 0.75 | 142.3 | 12.7 | 0.011 | 0.814 | 0.763 | 0.879 | 13.7 % |
| B1_Tg | lab | pooled | 0.904 [0.828, 0.969] | 134.7 | 1.17 | 207.1 | 19.7 | 0.046 | 0.904 | 0.699 | n/a (constant width) | n/a (constant width) |
| B1_Tg | lab | mondrian | 0.895 [0.844, 0.951] | 138.7 | 1.21 | 211.3 | 19.9 | 0.039 | 0.809 | 0.710 | 0.945 | 4.5 % |
| B2_E | comp | pooled | 0.898 [0.875, 0.926] | 13.1 | 0.86 | 26.6 | 2.1 | 0.016 | 0.898 | 0.704 | n/a (constant width) | n/a (constant width) |
| B2_E | comp | mondrian | 0.898 [0.877, 0.921] | 13.5 | 0.89 | 25.5 | 2.1 | 0.016 | 0.777 | 0.682 | 0.746 | 32.8 % |
| B2_E | lab | pooled | 0.881 [0.743, 0.970] | 24.7 | 1.62 | 50.5 | 4.2 | 0.066 | 0.881 | 0.490 | n/a (constant width) | n/a (constant width) |
| B2_E | lab | mondrian | 0.877 [0.796, 0.952] | 31.5 | 2.07 | 56.2 | 4.3 | 0.055 | 0.672 | 0.389 | 0.921 | 9.3 % |

The pooled scheme gives every candidate the same width, so it cannot rank candidates at all (no risk-coverage curve); the Mondrian scheme ranks them only at the resolution of the missing pattern.


## 4. Per-pattern results (the reviewers' question)


**B1_Tg, composition-grouped random split** - fused HGB unless stated; the composition-only model's Mondrian width is in parentheses.

| pattern | n test | stratum used | RMSE comp-only -> fused (K) | pooled cov. | Mondrian cov. | Mondrian width (K) | pooled width (K) | pooled interval score | Mondrian interval score |
|---|---|---|---|---|---|---|---|---|---|
| comp-only | 361 | comp-only | 35.3 -> 35.0 | 0.857 [0.816, 0.894] | 0.903 [0.869, 0.944] | 105.9 (106.8) | 84.4 | 176.9 | 173.4 |
| rho | 143 | rho | 32.1 -> 31.0 | 0.864 [0.806, 0.922] | 0.885 [0.842, 0.938] | 95.0 (96.1) | 84.4 | 155.0 | 152.3 |
| CTE | 337 | CTE | 33.0 -> 31.6 | 0.882 [0.845, 0.917] | 0.898 [0.854, 0.925] | 93.4 (98.2) | 84.4 | 156.8 | 156.5 |
| rho+CTE | 480 | rho+CTE | 24.6 -> 23.2 | 0.942 [0.922, 0.961] | 0.896 [0.861, 0.923] | 62.7 (68.3) | 84.4 | 116.1 | 111.3 |
| nD | 45 | nD | 32.1 -> 32.3 | 0.838 [0.718, 0.944] | 0.893 [0.744, 0.980] | 109.8 (106.5) | 84.4 | 170.5 | 168.5 |
| rho+nD | 48 | rho+nD | 20.6 -> 20.3 | 0.946 [0.891, 0.991] | 0.903 [0.814, 0.980] | 69.2 (69.7) | 84.4 | 115.0 | 115.7 |
| CTE+nD | 41 | varies | 23.4 -> 24.0 | 0.907 [0.825, 0.985] | 0.905 [0.784, 1.000] | 87.9 (89.9) | 84.4 | 113.9 | 120.5 |
| rho+CTE+nD | 70 | rho+CTE+nD | 25.1 -> 22.8 | 0.927 [0.847, 0.985] | 0.922 [0.821, 0.994] | 89.6 (99.4) | 84.4 | 110.4 | 116.5 |

**B1_Tg, laboratory-grouped split** - fused HGB unless stated; the composition-only model's Mondrian width is in parentheses.

| pattern | n test | stratum used | RMSE comp-only -> fused (K) | pooled cov. | Mondrian cov. | Mondrian width (K) | pooled width (K) | pooled interval score | Mondrian interval score |
|---|---|---|---|---|---|---|---|---|---|
| comp-only | 362 | comp-only | 50.0 -> 50.9 | 0.866 [0.777, 0.947] | 0.899 [0.783, 0.969] | 160.5 (161.7) | 134.7 | 262.2 | 262.7 |
| rho | 134 | rho | 49.1 -> 48.9 | 0.866 [0.759, 0.996] | 0.870 [0.714, 1.000] | 145.0 (142.7) | 134.7 | 247.8 | 257.8 |
| CTE | 351 | CTE | 40.7 -> 40.0 | 0.914 [0.809, 0.981] | 0.927 [0.831, 0.984] | 152.5 (157.2) | 134.7 | 194.2 | 201.9 |
| rho+CTE | 485 | rho+CTE | 35.5 -> 35.4 | 0.940 [0.841, 0.985] | 0.868 [0.751, 0.959] | 98.1 (98.7) | 134.7 | 174.9 | 170.3 |
| nD | 52 | varies | 42.8 -> 44.9 | 0.865 [0.576, 1.000] | 0.875 [0.667, 1.000] | 152.7 (152.6) | 134.7 | 193.6 | 203.4 |
| rho+nD | 63 | varies | 31.0 -> 32.2 | 0.952 [0.777, 1.000] | 0.914 [0.616, 1.000] | 137.3 (139.0) | 134.7 | 173.4 | 188.0 |
| CTE+nD | 40 | varies | 37.8 -> 40.2 | 0.855 [0.503, 1.000] | 0.872 [0.500, 1.000] | 147.8 (148.9) | 135.1 | 208.3 | 223.9 |
| rho+CTE+nD | 75 | varies | 43.7 -> 43.0 | 0.878 [0.722, 1.000] | 0.843 [0.555, 1.000] | 143.8 (141.3) | 134.7 | 199.8 | 233.6 |

A pattern that has no test record in a given split contributes nothing to that split, and its row is the mean over the splits in which it does appear: CTE+nD absent in 2 of 20 splits (`n_missing` in the JSON).


**B2_E, composition-grouped random split** - fused HGB unless stated; the composition-only model's Mondrian width is in parentheses.

| pattern | n test | stratum used | RMSE comp-only -> fused (GPa) | pooled cov. | Mondrian cov. | Mondrian width (GPa) | pooled width (GPa) | pooled interval score | Mondrian interval score |
|---|---|---|---|---|---|---|---|---|---|
| comp-only | 67 | comp-only | 9.5 -> 9.6 | 0.738 [0.650, 0.836] | 0.880 [0.742, 0.960] | 26.2 (27.0) | 13.1 | 60.0 | 52.0 |
| rho | 175 | rho | 5.5 -> 5.3 | 0.883 [0.835, 0.922] | 0.888 [0.826, 0.938] | 13.7 (15.1) | 13.1 | 27.3 | 27.4 |
| CTE | 28 | varies | 6.6 -> 6.6 | 0.819 [0.677, 0.960] | 0.918 [0.791, 1.000] | 25.8 (26.6) | 13.1 | 38.9 | 39.0 |
| rho+CTE | 200 | rho+CTE | 3.7 -> 3.3 | 0.949 [0.919, 0.974] | 0.905 [0.870, 0.941] | 8.4 (8.7) | 13.1 | 17.8 | 16.5 |
| Tg | 16 | comp-only | 5.2 -> 4.1 | 0.923 [0.703, 1.000] | 0.970 [0.836, 1.000] | 26.2 (27.0) | 13.1 | 24.0 | 31.4 |
| rho+Tg | 44 | rho+Tg | 7.1 -> 6.8 | 0.822 [0.753, 0.894] | 0.888 [0.768, 0.978] | 19.6 (21.9) | 13.1 | 35.0 | 33.8 |
| CTE+Tg | 40 | CTE+Tg | 5.5 -> 5.6 | 0.882 [0.764, 0.976] | 0.895 [0.768, 0.988] | 15.5 (14.1) | 13.1 | 28.6 | 29.4 |
| rho+CTE+Tg | 191 | rho+CTE+Tg | 4.2 -> 3.8 | 0.943 [0.912, 0.976] | 0.901 [0.862, 0.920] | 9.6 (11.4) | 13.1 | 19.7 | 18.8 |

**B2_E, laboratory-grouped split** - fused HGB unless stated; the composition-only model's Mondrian width is in parentheses.

| pattern | n test | stratum used | RMSE comp-only -> fused (GPa) | pooled cov. | Mondrian cov. | Mondrian width (GPa) | pooled width (GPa) | pooled interval score | Mondrian interval score |
|---|---|---|---|---|---|---|---|---|---|
| comp-only | 77 | varies | 15.7 -> 16.5 | 0.751 [0.567, 0.939] | 0.898 [0.770, 1.000] | 62.3 (58.7) | 24.7 | 103.0 | 109.8 |
| rho | 193 | rho | 11.2 -> 10.5 | 0.885 [0.693, 0.973] | 0.895 [0.709, 1.000] | 34.7 (36.9) | 24.7 | 57.4 | 66.0 |
| CTE | 31 | varies | 9.0 -> 9.5 | 0.836 [0.516, 1.000] | 0.926 [0.727, 1.000] | 45.1 (42.1) | 24.7 | 54.8 | 57.7 |
| rho+CTE | 233 | varies | 6.8 -> 6.5 | 0.912 [0.589, 0.991] | 0.859 [0.517, 0.986] | 20.1 (22.1) | 24.7 | 34.7 | 36.5 |
| Tg | 18 | varies | 9.1 -> 8.6 | 0.786 [0.297, 1.000] | 0.906 [0.412, 1.000] | 56.2 (50.3) | 24.4 | 44.7 | 68.1 |
| rho+Tg | 63 | varies | 11.3 -> 9.6 | 0.823 [0.462, 1.000] | 0.819 [0.438, 1.000] | 27.2 (29.2) | 24.7 | 49.5 | 52.3 |
| CTE+Tg | 43 | varies | 10.4 -> 10.5 | 0.759 [0.238, 1.000] | 0.774 [0.095, 1.000] | 35.9 (32.2) | 24.7 | 66.0 | 75.6 |
| rho+CTE+Tg | 197 | rho+CTE+Tg | 8.0 -> 7.6 | 0.897 [0.649, 0.996] | 0.820 [0.384, 1.000] | 21.2 (22.6) | 24.7 | 43.5 | 50.5 |

A pattern that has no test record in a given split contributes nothing to that split, and its row is the mean over the splits in which it does appear: Tg absent in 2 of 20 splits (`n_missing` in the JSON).



## 5. Width differences between patterns are mostly population differences

Ratios of Mondrian widths, relative to the composition-only glasses of the same split. *Total* = fused model, pattern p versus composition-only glasses (what a reader would see); *population* = the same ratio for the composition-only MODEL, which uses none of the auxiliary values and therefore measures only how homogeneous that sub-population is; *information* = fused / composition-only model within the pattern, the part actually contributed by the measured values. The last column is the share of the log width difference explained by the population term.


**B1_Tg, composition-grouped random split**

| pattern | total ratio | population ratio | information ratio | population share of log difference |
|---|---|---|---|---|
| rho | 0.90 [0.70, 1.02] | 0.91 [0.67, 1.12] | 0.99 [0.88, 1.09] | 98 % |
| CTE | 0.89 [0.71, 1.09] | 0.93 [0.73, 1.11] | 0.95 [0.90, 1.02] | 67 % |
| rho+CTE | 0.60 [0.49, 0.69] | 0.64 [0.51, 0.78] | 0.92 [0.86, 1.01] | 86 % |
| nD | 1.04 [0.73, 1.41] | 1.00 [0.73, 1.41] | 1.04 [0.80, 1.41] | n/a (no width difference) |
| rho+nD | 0.65 [0.38, 1.10] | 0.66 [0.36, 1.16] | 1.00 [0.58, 1.51] | 97 % |
| CTE+nD | 0.84 [0.50, 1.38] | 0.85 [0.43, 1.40] | 1.00 [0.74, 1.43] | 93 % |
| rho+CTE+nD | 0.85 [0.58, 1.27] | 0.93 [0.63, 1.43] | 0.91 [0.72, 1.17] | 50 % |

**B1_Tg, laboratory-grouped split**

| pattern | total ratio | population ratio | information ratio | population share of log difference |
|---|---|---|---|---|
| rho | 0.91 [0.59, 1.29] | 0.88 [0.62, 1.12] | 1.04 [0.78, 1.40] | 128 % |
| CTE | 0.96 [0.66, 1.43] | 0.99 [0.66, 1.72] | 0.98 [0.85, 1.22] | 79 % |
| rho+CTE | 0.63 [0.41, 0.91] | 0.63 [0.42, 1.02] | 1.00 [0.86, 1.15] | 101 % |
| nD | 0.96 [0.56, 1.36] | 0.96 [0.46, 1.47] | 1.01 [0.83, 1.13] | 115 % |
| rho+nD | 0.87 [0.39, 1.41] | 0.88 [0.34, 1.74] | 1.02 [0.65, 1.27] | 107 % |
| CTE+nD | 0.95 [0.39, 1.69] | 0.96 [0.44, 1.90] | 0.99 [0.73, 1.19] | 94 % |
| rho+CTE+nD | 0.94 [0.43, 1.84] | 0.90 [0.45, 1.57] | 1.01 [0.80, 1.23] | 109 % |

**B2_E, composition-grouped random split**

| pattern | total ratio | population ratio | information ratio | population share of log difference |
|---|---|---|---|---|
| rho | 0.56 [0.32, 0.93] | 0.59 [0.37, 0.80] | 0.91 [0.75, 1.10] | 89 % |
| CTE | 1.02 [0.50, 1.84] | 1.00 [0.54, 1.69] | 0.99 [0.77, 1.45] | n/a (no width difference) |
| rho+CTE | 0.34 [0.19, 0.50] | 0.34 [0.21, 0.56] | 0.98 [0.76, 1.31] | 100 % |
| Tg | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.99 [0.72, 1.45] | n/a (no width difference) |
| rho+Tg | 0.80 [0.44, 1.31] | 0.86 [0.41, 1.51] | 0.92 [0.68, 1.30] | 76 % |
| CTE+Tg | 0.61 [0.30, 0.82] | 0.56 [0.30, 0.96] | 1.10 [0.79, 1.64] | 119 % |
| rho+CTE+Tg | 0.39 [0.22, 0.56] | 0.44 [0.27, 0.60] | 0.85 [0.71, 0.96] | 86 % |

**B2_E, laboratory-grouped split**

| pattern | total ratio | population ratio | information ratio | population share of log difference |
|---|---|---|---|---|
| rho | 0.85 [0.10, 2.61] | 0.99 [0.11, 3.01] | 0.93 [0.73, 1.10] | 74 % |
| CTE | 0.84 [0.16, 1.26] | 0.87 [0.16, 1.60] | 1.09 [0.83, 1.66] | 100 % |
| rho+CTE | 0.47 [0.08, 1.17] | 0.55 [0.09, 1.19] | 0.91 [0.73, 1.19] | 83 % |
| Tg | 0.90 [0.33, 1.00] | 0.87 [0.27, 1.00] | 1.19 [0.83, 1.82] | 145 % |
| rho+Tg | 0.65 [0.10, 2.02] | 0.73 [0.10, 2.05] | 0.94 [0.69, 1.21] | 81 % |
| CTE+Tg | 0.68 [0.16, 1.40] | 0.64 [0.16, 1.28] | 1.15 [0.90, 1.82] | 109 % |
| rho+CTE+Tg | 0.45 [0.08, 1.19] | 0.51 [0.10, 1.18] | 0.94 [0.62, 1.27] | 86 % |

The laboratory-grouped split is the stronger version of this result, not the weaker one: for B1's rho+CTE pattern the information ratio is 1.00 [0.86, 1.15] across held-out laboratories against 0.92 [0.86, 1.01] on the random split - i.e. once the laboratory is held out the measured values contribute nothing at all to the width.


**The paper must not claim that the auxiliary measurements 'halve' the uncertainty.** Richly characterised glasses are simply more homogeneous families; a composition-only model produces nearly the same narrow intervals for them. The honest statement is that pattern-conditional calibration makes the interval width track what is known about a glass, and that the measurements themselves contribute the small 'information' column.


## 6. 'The measurement has not been made yet' is not the same as 'naturally missing'

All auxiliaries of the test records are masked at prediction time. `masked calibration` calibrates on calibration records masked identically; the two naive recipes reuse natural calibration residuals.


**B1_Tg, composition-grouped random split** - every test record masked

| fused model | RMSE natural | RMSE masked | RMSE of the composition-only model | masked cal.: cov. / width | naive borrowed comp-only quantile: cov. / width | naive pooled natural quantile: cov. / width |
|---|---|---|---|---|---|---|
| HGB fused (native NaN) | 29.34 | 33.17 [31.86, 35.44] | 30.35 | 0.896 [0.871, 0.908] / 98.9 | 0.909 [0.880, 0.940] / 105.9 | 0.860 [0.834, 0.882] / 84.4 |
| HGB fused + modality dropout | 30.10 | 30.59 [29.28, 33.22] | 30.35 | 0.897 [0.879, 0.910] / 88.3 | 0.934 [0.918, 0.950] / 110.5 | 0.892 [0.874, 0.908] / 86.4 |
| RF fused (median imputation + indicators) | 31.48 | 34.47 [32.79, 36.53] | 31.95 | 0.895 [0.875, 0.911] / 100.3 | 0.908 [0.886, 0.933] / 108.6 | 0.871 [0.853, 0.889] / 89.2 |

**B1_Tg, laboratory-grouped split** - every test record masked

| fused model | RMSE natural | RMSE masked | RMSE of the composition-only model | masked cal.: cov. / width | naive borrowed comp-only quantile: cov. / width | naive pooled natural quantile: cov. / width |
|---|---|---|---|---|---|---|
| HGB fused (native NaN) | 42.85 | 43.56 [31.27, 52.22] | 42.76 | 0.904 [0.834, 0.968] / 139.3 | 0.927 [0.832, 0.988] / 160.5 | 0.898 [0.820, 0.962] / 134.7 |
| HGB fused + modality dropout | 42.31 | 42.49 [30.99, 50.54] | 42.76 | 0.904 [0.845, 0.968] / 133.7 | 0.929 [0.858, 0.991] / 156.6 | 0.904 [0.847, 0.968] / 133.3 |
| RF fused (median imputation + indicators) | 45.44 | 46.14 [36.27, 55.50] | 45.59 | 0.906 [0.857, 0.961] / 148.5 | 0.926 [0.872, 0.987] / 167.8 | 0.903 [0.859, 0.957] / 146.0 |

**B2_E, composition-grouped random split** - every test record masked

| fused model | RMSE natural | RMSE masked | RMSE of the composition-only model | masked cal.: cov. / width | naive borrowed comp-only quantile: cov. / width | naive pooled natural quantile: cov. / width |
|---|---|---|---|---|---|---|
| HGB fused (native NaN) | 5.36 | 7.18 [6.49, 8.03] | 5.57 | 0.896 [0.868, 0.919] / 20.2 | 0.924 [0.835, 0.977] / 26.2 | 0.795 [0.729, 0.843] / 13.1 |
| HGB fused + modality dropout | 5.48 | 5.57 [4.81, 6.86] | 5.57 | 0.899 [0.875, 0.919] / 13.5 | 0.960 [0.925, 0.984] / 26.2 | 0.895 [0.872, 0.918] / 13.1 |
| RF fused (median imputation + indicators) | 5.82 | 7.20 [6.14, 8.36] | 6.00 | 0.895 [0.870, 0.924] / 18.8 | 0.942 [0.903, 0.988] / 28.4 | 0.821 [0.799, 0.851] / 13.1 |

**B2_E, laboratory-grouped split** - every test record masked

| fused model | RMSE natural | RMSE masked | RMSE of the composition-only model | masked cal.: cov. / width | naive borrowed comp-only quantile: cov. / width | naive pooled natural quantile: cov. / width |
|---|---|---|---|---|---|---|
| HGB fused (native NaN) | 9.97 | 11.21 [6.79, 14.77] | 10.26 | 0.879 [0.744, 0.977] / 29.2 | 0.946 [0.817, 1.000] / 62.3 | 0.829 [0.647, 0.961] / 24.7 |
| HGB fused + modality dropout | 10.10 | 10.18 [6.23, 14.22] | 10.26 | 0.882 [0.751, 0.967] / 25.5 | 0.951 [0.855, 1.000] / 57.6 | 0.879 [0.747, 0.969] / 25.1 |
| RF fused (median imputation + indicators) | 9.76 | 10.68 [6.67, 14.49] | 10.08 | 0.886 [0.778, 0.972] / 27.7 | 0.940 [0.798, 1.000] / 56.7 | 0.848 [0.688, 0.969] / 23.5 |

Per pattern (A2's design: only the records of one natural pattern are masked, calibration on the records of that pattern's stratum masked the same way), for both designs and both split families:


**B1_Tg, composition-grouped random split**

| pattern | n test | RMSE with the measurements | RMSE masked, no augmentation -> with modality dropout | RMSE of the composition-only model | masked cal. cov. / width (dropout model) | borrowed comp-only quantile cov. / width (dropout model) |
|---|---|---|---|---|---|---|
| rho | 143 | 31.0 | 33.8 -> 31.6 | 32.1 | 0.892 [0.814, 0.944] / 93.5 | 0.925 [0.892, 0.960] / 110.5 |
| CTE | 337 | 31.6 | 37.2 -> 32.6 | 33.0 | 0.897 [0.847, 0.942] / 96.6 | 0.922 [0.888, 0.961] / 110.5 |
| rho+CTE | 480 | 23.2 | 29.2 -> 24.3 | 24.6 | 0.894 [0.859, 0.914] / 67.0 | 0.963 [0.947, 0.981] / 110.5 |
| nD | 45 | 32.3 | 34.2 -> 31.6 | 32.1 | 0.893 [0.784, 0.976] / 104.8 | 0.913 [0.798, 0.978] / 110.5 |
| rho+nD | 48 | 20.3 | 22.9 -> 19.9 | 20.6 | 0.916 [0.767, 0.980] / 66.0 | 0.974 [0.941, 1.000] / 110.5 |
| CTE+nD | 41 | 24.0 | 28.3 -> 25.0 | 23.4 | 0.875 [0.690, 0.984] / 88.7 | 0.952 [0.881, 1.000] / 110.5 |
| rho+CTE+nD | 70 | 22.8 | 32.5 -> 24.8 | 25.1 | 0.924 [0.829, 1.000] / 96.6 | 0.953 [0.879, 0.994] / 110.5 |

**B1_Tg, laboratory-grouped split**

| pattern | n test | RMSE with the measurements | RMSE masked, no augmentation -> with modality dropout | RMSE of the composition-only model | masked cal. cov. / width (dropout model) | borrowed comp-only quantile cov. / width (dropout model) |
|---|---|---|---|---|---|---|
| rho | 134 | 48.9 | 49.2 -> 48.7 | 49.1 | 0.863 [0.748, 1.000] / 141.9 | 0.885 [0.766, 1.000] / 156.6 |
| CTE | 351 | 40.0 | 42.2 -> 40.3 | 40.7 | 0.926 [0.841, 0.990] / 152.1 | 0.938 [0.858, 1.000] / 156.6 |
| rho+CTE | 485 | 35.4 | 36.0 -> 35.5 | 35.5 | 0.880 [0.776, 0.978] / 102.0 | 0.951 [0.849, 0.996] / 156.6 |
| nD | 52 | 44.9 | 44.5 -> 44.8 | 42.8 | 0.894 [0.685, 1.000] / 153.8 | 0.897 [0.705, 1.000] / 156.6 |
| rho+nD | 63 | 32.2 | 31.1 -> 31.4 | 31.0 | 0.905 [0.503, 1.000] / 134.5 | 0.960 [0.809, 1.000] / 156.6 |
| CTE+nD | 40 | 40.2 | 39.2 -> 36.2 | 37.8 | 0.899 [0.627, 1.000] / 134.9 | 0.933 [0.750, 1.000] / 155.7 |
| rho+CTE+nD | 75 | 43.0 | 44.2 -> 42.1 | 43.7 | 0.844 [0.545, 1.000] / 136.8 | 0.902 [0.593, 1.000] / 156.6 |

**B2_E, composition-grouped random split**

| pattern | n test | RMSE with the measurements | RMSE masked, no augmentation -> with modality dropout | RMSE of the composition-only model | masked cal. cov. / width (dropout model) | borrowed comp-only quantile cov. / width (dropout model) |
|---|---|---|---|---|---|---|
| rho | 175 | 5.3 | 7.5 -> 5.5 | 5.5 | 0.902 [0.857, 0.936] / 14.6 | 0.954 [0.917, 0.989] / 26.2 |
| CTE | 28 | 6.6 | 8.0 -> 6.8 | 6.6 | 0.912 [0.790, 1.000] / 26.4 | 0.919 [0.790, 1.000] / 26.2 |
| rho+CTE | 200 | 3.3 | 6.0 -> 3.7 | 3.7 | 0.896 [0.837, 0.931] / 8.4 | 0.984 [0.955, 0.998] / 26.2 |
| Tg | 16 | 4.1 | 5.0 -> 5.0 | 5.2 | 0.951 [0.836, 1.000] / 26.4 | 0.944 [0.836, 1.000] / 26.2 |
| rho+Tg | 44 | 6.8 | 9.4 -> 7.0 | 7.1 | 0.884 [0.696, 1.000] / 20.5 | 0.934 [0.867, 1.000] / 26.2 |
| CTE+Tg | 40 | 5.6 | 6.1 -> 5.4 | 5.5 | 0.886 [0.758, 0.976] / 14.2 | 0.964 [0.918, 1.000] / 26.2 |
| rho+CTE+Tg | 191 | 3.8 | 6.1 -> 4.1 | 4.2 | 0.905 [0.844, 0.955] / 10.9 | 0.975 [0.954, 1.000] / 26.2 |

**B2_E, laboratory-grouped split**

| pattern | n test | RMSE with the measurements | RMSE masked, no augmentation -> with modality dropout | RMSE of the composition-only model | masked cal. cov. / width (dropout model) | borrowed comp-only quantile cov. / width (dropout model) |
|---|---|---|---|---|---|---|
| rho | 193 | 10.5 | 12.1 -> 10.9 | 11.2 | 0.893 [0.685, 1.000] / 36.2 | 0.942 [0.828, 1.000] / 57.6 |
| CTE | 31 | 9.5 | 10.0 -> 9.2 | 9.0 | 0.905 [0.544, 1.000] / 40.6 | 0.951 [0.708, 1.000] / 57.6 |
| rho+CTE | 233 | 6.5 | 8.6 -> 6.9 | 6.8 | 0.871 [0.546, 0.991] / 21.0 | 0.973 [0.830, 1.000] / 57.6 |
| Tg | 18 | 8.6 | 8.8 -> 9.1 | 9.1 | 0.875 [0.362, 1.000] / 49.7 | 0.916 [0.435, 1.000] / 59.8 |
| rho+Tg | 63 | 9.6 | 11.3 -> 11.0 | 11.3 | 0.836 [0.397, 1.000] / 30.3 | 0.900 [0.525, 1.000] / 57.6 |
| CTE+Tg | 43 | 10.5 | 10.5 -> 10.7 | 10.4 | 0.785 [0.190, 1.000] / 31.2 | 0.882 [0.238, 1.000] / 57.6 |
| rho+CTE+Tg | 197 | 7.6 | 8.9 -> 7.9 | 8.0 | 0.822 [0.403, 1.000] / 23.1 | 0.944 [0.647, 1.000] / 57.6 |


## 7. The trust layer as one decision procedure (Reviewer 2 comment 4)

Candidates = the test records of each split; specification Tg >= tau with tau fixed a priori as the median (845.1 K) and the 75th percentile (913.1 K) of Tg over the whole B1 dataset; `trustlayer.TrustLayer` (imported unchanged) with the missing pattern as the Mondrian group label, alpha = 0.1, conformal selection at q = 0.1, novelty level 0.05. FDP = false accepts / accepted (a false accept is an accepted candidate whose true Tg is below tau); the table gives the ratio of totals over all splits. 'Measurements' counts candidates sent to a Tg measurement (decision 'measure' or 'not certified'); 'auxiliary measurements' counts the density / expansion / refractive-index values the arm consumed.


**composition-grouped random split, tau = median** (fused HGB predictor; 0.50 of candidates actually satisfy the specification)

Counts are **per split** (mean over 20 splits); the two FDP columns are **ratios of totals over all 20 splits**, and the n in the worst-pattern column is that pattern's accepted total over all splits, not a per-split count.

| arm | accepted (per split) | FDP among accepted (all splits) | power (per split) | measured (per split) | auxiliary measurements (per split) | worst-pattern FDP (all splits, n = accepted total) | interval miscoverage (per split) |
|---|---|---|---|---|---|---|---|
| interval, pooled calibration | 499 | 0.016 | 0.637 | 636 | 1873 | 0.192 (nD, n=52) | 0.104 |
| interval, Mondrian by pattern | 493 | 0.014 | 0.631 | 666 | 1873 | 0.273 (nD, n=33) | 0.101 |
| interval, Mondrian + shift check | 480 | 0.014 | 0.615 | 714 | 1873 | 0.273 (nD, n=33) | 0.090 |
| conformal selection, pooled (BH q=0.10) | 793 | 0.097 | 0.930 | 342 | 1873 | 0.321 (nD, n=196) | 0.104 |
| conformal selection, Mondrian p-values, one BH | 779 | 0.092 | 0.918 | 380 | 1873 | 0.246 (nD, n=122) | 0.101 |
| conformal selection, BH inside each pattern | 767 | 0.095 | 0.902 | 392 | 1873 | 0.207 (rho+nD, n=29) | 0.101 |
| not yet measured: interval, masked calibration | 468 | 0.017 | 0.598 | 762 | 0 | 0.194 (nD, n=36) | 0.104 |
| not yet measured: selection, masked calibration | 769 | 0.104 | 0.895 | 461 | 0 | 0.344 (nD, n=157) | 0.104 |
| not yet measured: naive, borrowed comp-only quantile | 451 | 0.016 | 0.576 | 806 | 0 | 0.233 (nD, n=30) | 0.091 |
| not yet measured: naive, pooled natural quantile | 510 | 0.020 | 0.649 | 667 | 0 | 0.200 (nD, n=50) | 0.140 |

**composition-grouped random split, tau = q75** (fused HGB predictor; 0.26 of candidates actually satisfy the specification)

Counts are **per split** (mean over 20 splits); the two FDP columns are **ratios of totals over all 20 splits**, and the n in the worst-pattern column is that pattern's accepted total over all splits, not a per-split count.

| arm | accepted (per split) | FDP among accepted (all splits) | power (per split) | measured (per split) | auxiliary measurements (per split) | worst-pattern FDP (all splits, n = accepted total) | interval miscoverage (per split) |
|---|---|---|---|---|---|---|---|
| interval, pooled calibration | 171 | 0.018 | 0.428 | 419 | 1873 | 0.049 (CTE, n=527) | 0.104 |
| interval, Mondrian by pattern | 171 | 0.018 | 0.429 | 419 | 1873 | 0.042 (CTE, n=477) | 0.101 |
| interval, Mondrian + shift check | 167 | 0.016 | 0.420 | 476 | 1873 | 0.039 (CTE, n=466) | 0.090 |
| conformal selection, pooled (BH q=0.10) | 378 | 0.095 | 0.873 | 212 | 1873 | 0.139 (CTE+nD, n=208) | 0.104 |
| conformal selection, Mondrian p-values, one BH | 359 | 0.085 | 0.839 | 231 | 1873 | 0.800 (rho+nD, n=25) | 0.101 |
| conformal selection, BH inside each pattern | 366 | 0.095 | 0.845 | 224 | 1873 | 0.160 (CTE+nD, n=150) | 0.101 |
| not yet measured: interval, masked calibration | 158 | 0.023 | 0.393 | 494 | 0 | 0.054 (CTE, n=534) | 0.104 |
| not yet measured: selection, masked calibration | 366 | 0.098 | 0.842 | 286 | 0 | 0.156 (CTE+nD, n=224) | 0.104 |
| not yet measured: naive, borrowed comp-only quantile | 147 | 0.020 | 0.367 | 526 | 0 | 0.053 (CTE, n=508) | 0.091 |
| not yet measured: naive, pooled natural quantile | 183 | 0.026 | 0.454 | 423 | 0 | 0.063 (CTE, n=589) | 0.140 |

**laboratory-grouped split, tau = median** (fused HGB predictor; 0.52 of candidates actually satisfy the specification)

Counts are **per split** (mean over 20 splits); the two FDP columns are **ratios of totals over all 20 splits**, and the n in the worst-pattern column is that pattern's accepted total over all splits, not a per-split count.

| arm | accepted (per split) | FDP among accepted (all splits) | power (per split) | measured (per split) | auxiliary measurements (per split) | worst-pattern FDP (all splits, n = accepted total) | interval miscoverage (per split) |
|---|---|---|---|---|---|---|---|
| interval, pooled calibration | 363 | 0.028 | 0.438 | 1020 | 1929 | 0.049 (rho, n=817) | 0.096 |
| interval, Mondrian by pattern | 401 | 0.026 | 0.474 | 990 | 1929 | 0.066 (rho, n=760) | 0.105 |
| interval, Mondrian + shift check | 388 | 0.024 | 0.458 | 1037 | 1929 | 0.068 (rho, n=735) | 0.091 |
| conformal selection, pooled (BH q=0.10) | 765 | 0.112 | 0.820 | 618 | 1929 | 0.510 (nD, n=202) | 0.096 |
| conformal selection, Mondrian p-values, one BH | 769 | 0.123 | 0.811 | 623 | 1929 | 0.515 (rho+nD, n=268) | 0.105 |
| conformal selection, BH inside each pattern | 746 | 0.129 | 0.780 | 646 | 1929 | 0.545 (rho+nD, n=213) | 0.105 |
| not yet measured: interval, masked calibration | 360 | 0.029 | 0.434 | 1052 | 0 | 0.049 (CTE, n=990) | 0.096 |
| not yet measured: selection, masked calibration | 761 | 0.101 | 0.825 | 651 | 0 | 0.483 (nD, n=180) | 0.096 |
| not yet measured: naive, borrowed comp-only quantile | 309 | 0.027 | 0.377 | 1141 | 0 | 0.047 (CTE, n=899) | 0.073 |
| not yet measured: naive, pooled natural quantile | 370 | 0.029 | 0.449 | 1029 | 0 | 0.050 (CTE, n=1018) | 0.102 |

**laboratory-grouped split, tau = q75** (fused HGB predictor; 0.27 of candidates actually satisfy the specification)

Counts are **per split** (mean over 20 splits); the two FDP columns are **ratios of totals over all 20 splits**, and the n in the worst-pattern column is that pattern's accepted total over all splits, not a per-split count.

| arm | accepted (per split) | FDP among accepted (all splits) | power (per split) | measured (per split) | auxiliary measurements (per split) | worst-pattern FDP (all splits, n = accepted total) | interval miscoverage (per split) |
|---|---|---|---|---|---|---|---|
| interval, pooled calibration | 87 | 0.035 | 0.230 | 721 | 1929 | 0.043 (CTE, n=234) | 0.096 |
| interval, Mondrian by pattern | 84 | 0.045 | 0.214 | 772 | 1929 | 0.088 (rho, n=159) | 0.105 |
| interval, Mondrian + shift check | 81 | 0.045 | 0.207 | 844 | 1929 | 0.097 (rho, n=145) | 0.091 |
| conformal selection, pooled (BH q=0.10) | 336 | 0.156 | 0.665 | 472 | 1929 | 0.181 (rho+CTE+nD, n=127) | 0.096 |
| conformal selection, Mondrian p-values, one BH | 335 | 0.201 | 0.633 | 522 | 1929 | 0.925 (rho+nD, n=80) | 0.105 |
| conformal selection, BH inside each pattern | 362 | 0.235 | 0.659 | 497 | 1929 | 0.938 (rho+nD, n=112) | 0.105 |
| not yet measured: interval, masked calibration | 94 | 0.038 | 0.241 | 754 | 0 | 0.055 (CTE, n=309) | 0.096 |
| not yet measured: selection, masked calibration | 286 | 0.144 | 0.588 | 562 | 0 | 0.194 (rho+CTE+nD, n=124) | 0.096 |
| not yet measured: naive, borrowed comp-only quantile | 74 | 0.041 | 0.193 | 882 | 0 | 0.052 (CTE, n=251) | 0.073 |
| not yet measured: naive, pooled natural quantile | 99 | 0.039 | 0.256 | 725 | 0 | 0.055 (CTE, n=327) | 0.102 |

Per-pattern false-accept proportions (ratio of totals over splits, patterns with at least 20 accepted candidates in total), fused HGB, tau = median:


**composition-grouped random split**

| arm | rho+CTE | comp-only | CTE | rho | rho+CTE+nD | nD | rho+nD | CTE+nD |
|---|---|---|---|---|---|---|---|---|
| interval, pooled calibration | 0.008 (n=4107) | 0.021 (n=2736) | 0.025 (n=1324) | 0.015 (n=1041) | 0.013 (n=399) | 0.192 (n=52) | 0.000 (n=79) | 0.009 (n=234) |
| interval, Mondrian by pattern | 0.009 (n=4452) | 0.015 (n=2388) | 0.024 (n=1266) | 0.011 (n=989) | 0.010 (n=381) | 0.273 (n=33) | 0.018 (n=109) | 0.017 (n=233) |
| conformal selection, pooled (BH q=0.10) | 0.068 (n=6254) | 0.104 (n=4396) | 0.149 (n=2426) | 0.085 (n=1447) | 0.072 (n=665) | 0.321 (n=196) | 0.142 (n=190) | 0.140 (n=292) |
| conformal selection, Mondrian p-values, one BH | 0.066 (n=6237) | 0.089 (n=4223) | 0.144 (n=2403) | 0.100 (n=1473) | 0.072 (n=628) | 0.246 (n=122) | 0.179 (n=196) | 0.138 (n=290) |
| conformal selection, BH inside each pattern | 0.083 (n=6394) | 0.109 (n=4406) | 0.100 (n=2207) | 0.098 (n=1473) | 0.072 (n=583) | n/a | 0.207 (n=29) | 0.099 (n=243) |

**laboratory-grouped split**

| arm | rho+CTE | comp-only | CTE | rho | rho+CTE+nD | nD | rho+nD | CTE+nD |
|---|---|---|---|---|---|---|---|---|
| interval, pooled calibration | 0.010 (n=2765) | 0.037 (n=2420) | 0.042 (n=956) | 0.049 (n=817) | 0.020 (n=199) | n/a | n/a | 0.000 (n=79) |
| interval, Mondrian by pattern | 0.011 (n=3957) | 0.037 (n=2142) | 0.039 (n=870) | 0.066 (n=760) | 0.014 (n=214) | n/a | n/a | 0.000 (n=55) |
| conformal selection, pooled (BH q=0.10) | 0.049 (n=6145) | 0.101 (n=4280) | 0.267 (n=2255) | 0.080 (n=1362) | 0.061 (n=586) | 0.510 (n=202) | 0.460 (n=265) | 0.069 (n=202) |
| conformal selection, Mondrian p-values, one BH | 0.072 (n=6522) | 0.108 (n=4143) | 0.261 (n=2109) | 0.107 (n=1391) | 0.089 (n=608) | 0.330 (n=100) | 0.515 (n=268) | 0.179 (n=235) |
| conformal selection, BH inside each pattern | 0.077 (n=6566) | 0.121 (n=4179) | 0.266 (n=1731) | 0.128 (n=1361) | 0.133 (n=533) | 0.398 (n=83) | 0.545 (n=213) | 0.235 (n=251) |

Reading of section 7. (i) Interval-based acceptance is very conservative overall - it accepts only when the whole interval clears tau - so its false-accept proportion stays far below alpha, but that is a marginal statement: per pattern it reaches the values in the table below, which is exactly the coverage-versus-FDR gap documented in `trustlayer.py` (G1). (ii) Conformal selection accepts far more candidates at a controlled error rate; against q = 0.10 its guarantee is **upheld** on the composition-grouped split and **violated** on the laboratory-grouped one (the table below gives every arm). (iii) Where it fails, it fails because a candidate from an unseen laboratory is not exchangeable with the calibration records of its stratum. Running BH inside each pattern makes it worse, not better - the realised FDP of the per-stratum scope is above the pooled scope's at both thresholds on the laboratory-grouped split - because the smaller strata carry the laboratory offset undiluted.


Because a realised FDP is itself a noisy quantity, every selection arm is tested against its nominal q with a one-sided one-sample t over the 20 splits (JSON `fdp_vs_q`). 'Upheld' and 'violated' anywhere in this report mean this test, not a point estimate:

| split family | tau | arm | realised FDP (mean [2.5, 97.5] over splits) | ratio of totals | one-sided t vs q = 0.10 |
|---|---|---|---|---|---|
| composition-grouped random | median | conformal selection, pooled (BH q=0.10) | 0.097 [0.075, 0.125] | 0.097 | p = 0.801 (not distinguishable from q) |
| composition-grouped random | median | conformal selection, Mondrian p-values, one BH | 0.091 [0.067, 0.116] | 0.092 | p = 0.976 (not distinguishable from q) |
| composition-grouped random | median | conformal selection, BH inside each pattern | 0.094 [0.072, 0.121] | 0.095 | p = 0.941 (not distinguishable from q) |
| composition-grouped random | q75 | conformal selection, pooled (BH q=0.10) | 0.093 [0.046, 0.132] | 0.095 | p = 0.860 (not distinguishable from q) |
| composition-grouped random | q75 | conformal selection, Mondrian p-values, one BH | 0.084 [0.042, 0.113] | 0.085 | p = 0.996 (not distinguishable from q) |
| composition-grouped random | q75 | conformal selection, BH inside each pattern | 0.094 [0.055, 0.133] | 0.095 | p = 0.870 (not distinguishable from q) |
| laboratory-grouped | median | conformal selection, pooled (BH q=0.10) | 0.120 [0.029, 0.275] | 0.112 | p = 0.128 (not distinguishable from q) |
| laboratory-grouped | median | conformal selection, Mondrian p-values, one BH | 0.130 [0.047, 0.262] | 0.123 | p = 0.036 (above q) |
| laboratory-grouped | median | conformal selection, BH inside each pattern | 0.138 [0.045, 0.285] | 0.129 | p = 0.019 (above q) |
| laboratory-grouped | q75 | conformal selection, pooled (BH q=0.10) | 0.130 [0.028, 0.286] | 0.156 | p = 0.055 (not distinguishable from q) |
| laboratory-grouped | q75 | conformal selection, Mondrian p-values, one BH | 0.176 [0.036, 0.300] | 0.201 | p < 0.001 (above q) |
| laboratory-grouped | q75 | conformal selection, BH inside each pattern | 0.223 [0.091, 0.385] | 0.235 | p < 0.001 (above q) |
(iv) The auxiliary measurements barely pay for themselves in decisions either.

On the composition-grouped random split (tau = median, Mondrian intervals), the composition-only model sends 686 of 1525 candidates to a Tg measurement; the fused model sends 666 while consuming 1873 existing auxiliary values, i.e. it saves 20.2 Tg measurements in exchange, and its power moves from 0.619 to 0.631. The modality-dropout model deciding with nothing measured yet (masked calibration) needs 667 measurements at power 0.626, against 670 and 0.624 with the auxiliaries in hand.

On the laboratory-grouped split (tau = median, Mondrian intervals), the composition-only model sends 991 of 1558 candidates to a Tg measurement; the fused model sends 990 while consuming 1929 existing auxiliary values, i.e. it saves 1.2 Tg measurements on average, i.e. effectively none in exchange, and its power moves from 0.464 to 0.474. The modality-dropout model deciding with nothing measured yet (masked calibration) needs 1004 measurements at power 0.451, against 982 and 0.481 with the auxiliaries in hand.


Shift check inside the same procedure: the conformal novelty flag fires on 0.046 of candidates (composition-grouped random), 0.064 of candidates (laboratory-grouped). Composition-and-auxiliary novelty is a covariate-space test: it does not detect a laboratory offset, which is a label shift (A2 section 2.5), so it cannot rescue the laboratory-grouped coverage on its own.


Predictor families other than the fused HGB behave the same way; the JSON holds every arm for HGB composition-only, HGB fused (native NaN), HGB fused + modality dropout, RF fused (median imputation + indicators).


## 8. What the manuscript may and may not claim from this experiment

**May be claimed.**

* The fusion experiment uses *heterogeneous, naturally incomplete characterisation measurements (density, dilatometric expansion, refractive index) reported by different laboratories*, with 8 missing patterns that occur in the literature, not a composition vector split into two halves and not a simulated mask (answers R1-3 and R2-6 on the 'idealized' character of the old Figure 6).

* Missingness is informative (MNAR): the pattern predicts the property and tracks the reporting practice of the source laboratory (section 1, computed against a permutation baseline), so a single pooled conformal quantile over-covers well-characterised glasses and under-covers composition-only ones; Mondrian-by-pattern calibration narrows the spread of per-pattern coverage on i.i.d.-style splits and makes the width track what was measured. Say 'measured, reported and captured in SciGlass', not 'chosen to measure'.

* The deployment case matters: a model with native NaN handling has learned 'missing means a different glass family', so masking a measurement that simply has not been made yet degrades it; modality-dropout training plus calibration on identically masked calibration records restores both accuracy and valid, tight intervals.

* The four components run as one procedure on one object (uncertainty, shift check, decision/acquisition, missing-modality strata), and the experiment reports what that procedure actually delivers, including where the guarantee does not hold.


**Must not be claimed.**

* Not imaging, spectral or histological multimodality. SciGlass has no images or spectra. State it as a limitation, and do not generalise from three scalar characterisation measurements to multimodal biomedical data.

* Not an accuracy story: the fused model gains 3 % RMSE on B1's random split and nothing across laboratories.

* The auxiliaries do not 'halve' or otherwise materially shrink the uncertainty of a given glass. Almost all of the between-pattern width difference is a population difference (section 5).

* Pattern-conditional calibration is not a general fix. Per-SPLIT per-pattern coverage range (the statistic that can be tested pairwise): on the composition-grouped random split it shows no significant change for B1 (paired difference -0.006, SD 0.049, 12 of 20 splits worse / 8 better, Wilcoxon p = 0.956) and narrows significantly for B2 (paired difference -0.094, SD 0.067, 2 of 20 splits worse / 18 better, Wilcoxon p < 0.001); across held-out laboratories it shows no significant change for B1 (paired difference +0.048, SD 0.126, 12 of 20 splits worse / 8 better, Wilcoxon p = 0.097) and widens significantly for B2 (paired difference +0.060, SD 0.138, 14 of 20 splits worse / 5 better, Wilcoxon p = 0.044). The range of the per-pattern coverage MEANS on the laboratory split is B1 0.855 (CTE+nD) to 0.952 (rho+nD) pooled versus 0.843 (rho+CTE+nD) to 0.927 (CTE) Mondrian, B2 0.751 (comp-only) to 0.912 (rho+CTE) versus 0.774 (CTE+Tg) to 0.926 (CTE) - a different statistic from the per-split range (it averages each pattern first), and on B1 it moves the other way, which is why neither is quoted without its name. The worst-pattern minimum does not improve either, and its laboratory-split movement is not individually significant (B1 paired difference -0.041, SD 0.125, 8 of 20 splits better / 12 worse, Wilcoxon p = 0.154; B2 paired difference -0.035, SD 0.136, 9 of 20 splits better / 10 worse, Wilcoxon p = 0.243). **The defensible claim is that across laboratories Mondrian does not bring the worst pattern back to the nominal level - not that it measurably lowers that minimum** (the one significant laboratory-split movement of any of these statistics is the widening of B2's per-split range, section 0 item 3(b)). Whatever it does to the range there, the worst pattern stays at 0.724 (B1) and 0.670 (B2) against a nominal 0.90.

* The FDR guarantee of the selection arm is violated once whole laboratories are held out, and upheld when they are not; the verdict is a one-sided t of the realised FDP against q over the 20 splits, not a point estimate. Composition-grouped random split: realised FDP 0.085-0.097 as a ratio of totals over the 6 threshold x scope combinations; **none of the 6 is above q = 0.10** by a one-sided t over the splits. Laboratory-grouped split: realised FDP 0.112-0.235 as a ratio of totals over the 6 threshold x scope combinations; **4 of 6 are above q = 0.10** by a one-sided t over the splits.

  **Quote the arms with the strongest evidence, not the mildest one**, in that order: *conformal selection, BH inside each pattern, tau = q75*: 0.223 [0.091, 0.385] per split, 0.235 as a ratio of totals, one-sided t vs q = 0.10 p < 0.001; *conformal selection, Mondrian p-values, one BH, tau = q75*: 0.176 [0.036, 0.300] per split, 0.201 as a ratio of totals, one-sided t vs q = 0.10 p < 0.001; *conformal selection, BH inside each pattern, tau = median*: 0.138 [0.045, 0.285] per split, 0.129 as a ratio of totals, one-sided t vs q = 0.10 p = 0.019; *conformal selection, Mondrian p-values, one BH, tau = median*: 0.130 [0.047, 0.262] per split, 0.123 as a ratio of totals, one-sided t vs q = 0.10 p = 0.036.

  Arms whose realised FDP exceeds q but which are **not individually distinguishable from q** at 20 splits must be qualified as such wherever they are quoted: conformal selection, pooled (BH q=0.10), tau = q75: 0.130 [0.028, 0.286] per split, 0.156 as a ratio of totals, one-sided t vs q = 0.10 p = 0.055; conformal selection, pooled (BH q=0.10), tau = median: 0.120 [0.029, 0.275] per split, 0.112 as a ratio of totals, one-sided t vs q = 0.10 p = 0.128.

  Cross-laboratory shift is a label/measurement shift (A2 section 2.5) and needs the target-recalibration machinery of the other experiments, not a Mondrian stratum.

* Nothing here supports the retrospective '65-90 % fewer measurements' claim of the old Figure 11; that is a separate, retrospective replay experiment.


**Suggested wording.** "We fuse composition with heterogeneous, naturally incomplete characterisation measurements - density, dilatometric expansion and refractive index - reported by different laboratories, which co-occur with the glass-transition temperature in 8 natural missingness patterns. Fusion changes accuracy only marginally; what changes is the reliability of the uncertainty statement. A single pooled conformal quantile over-covers richly characterised glasses (0.94) and under-covers glasses described by composition alone (0.86), while calibrating separately within each missingness pattern brings every pattern close to the nominal level (0.885 (rho) to 0.922 (rho+CTE+nD)). The narrower intervals of well-characterised glasses are, however, mostly a property of those sub-populations rather than information contributed by the measurements themselves, and when whole laboratories are held out the pattern-conditional calibration no longer equalises coverage at all."


**Limitations of this experiment.** (0) **SciGlass is a curated compilation, so the missingness is 'measured, reported and captured', not 'measured and reported'.** A missing density may mean the source paper reported one that the curators did not transcribe; A2 section 1 documents the compilation character, and A2 section 3 finds that 27-32 % of cross-laboratory duplicate pairs report an identical Tg, consistent with re-reported or compiled values. Nothing in this experiment requires the missingness to be a pure record of laboratory intent - only that it is real, structured and correlated with the property, which section 1 establishes by computation - but the paper must not describe the pattern as what a laboratory *chose* to measure. (i) The laboratory is a normalised first-author key, a proxy. (ii) The dilatometric expansion coefficient can come from the same instrument run as a dilatometric Tg, so that auxiliary is not fully instrument-independent; the viscosity iso-points and crystallisation temperatures that are near-label leakage were excluded a priori (section 1). (iii) Patterns are unbalanced, so worst-pattern statistics carry binomial noise; the reference columns quantify it. (iv) All splits resample one finite retrospective database; the percentile intervals describe split-to-split variability only. (v) The specification thresholds are dataset quantiles fixed a priori, not a clinical requirement, and none of the conformal statements is a safety statement (trustlayer.py, G1-G3).


## 9. Fix round: what an independent verification changed

The numerical pipeline was independently re-implemented and reproduced exactly (same split indices, same per-split values); nothing in the computation changed in this round. What changed is what the report claims from it:

1. **Cross-model number substitution repaired (severe).** Bottom line 5 quoted two masked-calibration numbers for the modality-dropout model and the third for the native-NaN model, and called it 'invalid'. It now names the model for every number and gives both models' pooled-natural-quantile coverage.

2. **Worst-pattern coverage is no longer printed without dispersion.** Section 3 now shows the per-pattern coverage RANGE in the main table and the worst-pattern minimum in a separate table with percentile intervals and the paired split-to-split difference (mean, SD, splits better / worse, Wilcoxon). The bottom line was rewritten accordingly: section 0 now names the design x split families in which the minimum actually moves, and the laboratory-split changes, being inside split noise, support the ABSENCE of a repair rather than a measured degradation.

3. **One named statistic per sentence.** Bottom line 3 previously quoted the coverage range for the random split and the worst-pattern minimum for the laboratory split. Three statistics are now defined explicitly - the range of the per-pattern coverage means, the mean per-split range, and the per-split worst-pattern minimum - and all three are given for all four design x split families. The second and third disagree in direction on B1's laboratory split, which is precisely why the original mixing was a defect.

4. **The root-fallback statement was wrong and is corrected.** The script's docstring claimed the merged composition-only root never fell below the minimum calibration size. Section 2 above now reports the actual occurrences from `strata.root_pooled_fallback`.

5. **The missingness mechanism is no longer over-attributed.** SciGlass is a curated compilation, so the wording is 'measured, reported and captured in SciGlass'; database curation is limitation (0).

6. **An illustrative claim is no longer presented as a finding.** The 'patent and industrial series report density and expansion; optical papers report the refractive index' sentence was not computed anywhere. It is replaced by a computed laboratory/publication concentration statistic with a permutation baseline (section 1), and A2's attribution is explicitly marked as an illustration.

7. **The FDR violation is demonstrated with the decisive cases.** Every selection arm now carries a one-sided t against the nominal q (section 7 table); section 8 orders the laboratory-split evidence by the strength of that test, quotes the decisive arms first, and explicitly qualifies any arm whose realised FDP exceeds q but is not individually distinguishable from it. Every 'upheld' / 'violated' verdict in this report is now emitted by the test rather than written into the prose.

8. **Cosmetic fixes:** the split-provenance promise in section 2 now matches what sections 0 and 5 actually quote; the leftover caption fragment before the per-pattern masking tables is gone; the measurement-saving sentence no longer reads 'saves 1 Tg measurements'; per-pattern tables footnote the patterns that are absent from some splits' test sets; and the section 7 arm tables state which columns are per-split means and which are totals over all splits.

9. **Two sources of run-to-run drift found while re-running, and removed.** The Beta-binomial reference for the worst-pattern coverage drew its random numbers in `set` iteration order, which varies with the process hash seed and moved that reference by about 0.001 between runs; and RandomForest's parallel tree accumulation is not bit-reproducible across processes (about 2e-13), which is enough to flip a width-stratified-coverage bin on an exact tie. Both are fixed (sorted iteration, RF `n_jobs = 1`), and the JSON is now bit-exact when the script is run twice. No reported digit changed.


**Verifier issues not adopted: none.** Every issue raised was accepted and fixed. Two were corrections of fact (items 1 and 4), three were over-claims relative to the dispersion or the provenance of the evidence (items 2, 5 and 7), one replaced an uncomputed assertion with a computation (item 6), and the rest were presentational. The verifier also listed inaccuracies confined to the implementer's summary message rather than to these outputs; those are corrected in the summary returned with this run.

