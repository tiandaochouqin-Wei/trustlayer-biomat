# R7 - the trust layer as one decision procedure (Reviewer 2, comments 2, 4 and 7)

Script: `revision/code/exp_R7_pipeline.py`; library: `revision/code/trustlayer.py`. Protocol: `revision/code/PROTOCOL.md` (grouped 60/20/20 source split, target pool halves, alpha = 0.10, finite-sample conformal quantile, 20 seeds). Numbers are mean [2.5-97.5 percentile] over 20 splits; the splits resample one finite dataset, so the interval describes split-to-split variability, not population sampling error. Runtime 3.2 min.

Candidate pool = source-test U target-test. Specification: acceptable iff y >= tau. **Headline tau = median of y over the split's recalibration pool** (protocol-compliant: no target-test label enters it; it is used by no trust-layer component). Sensitivity thresholds: see the deviations section. FDR = mean over splits of the false-accept proportion among accepted (FDP = 0 when nothing is accepted). 'FDP if any' = mean over the splits with at least one accept in that subset. 'Pooled FDP' = total false accepts / total accepts over the 20 splits (more stable than the FDR when a subgroup has fewer than one accept per split). 'Any' = share of splits with at least one accept in that subset. 'Measurements' = measure + not_certified + recalibration budget m (m counted for A3, A4, A4s, D3 and D4). 'Uncond. FA' = false accepts / pool size. A subset metric is averaged over the splits in which that subset is non-empty (e.g. 'target not flagged' is empty in some polymer splits), so the ratio of two subset means is not a share; shares of counts in the text are computed from pooled totals. Each summary in the JSON carries its own `n` of splits.


## What is and is not guaranteed (theory, stated once)

Let C(x) = [mu(x) - q, mu(x) + q] be the split-conformal interval of a stratum and let a
candidate be 'accepted' by the interval rule iff the whole interval satisfies the
specification (y >= tau means lo >= tau).

* **G1, coverage.** If a candidate is exchangeable with the calibration records of the
  stratum used for it, P(Y in C(X)) >= 1 - alpha (marginal over calibration and candidate).
  An interval-accept of a candidate that violates the specification requires Y < tau <= lo,
  i.e. Y outside C(X). Hence **P(accept AND violate) <= P(Y not in C(X)) <= alpha.** (The
  per-split inequality 'false accepts / n <= miscoverage share' holds by construction and is
  asserted in the code only as a consistency check of the implementation; it is not
  empirical evidence for G1. The empirical evidence is the realised miscoverage per stratum.)
* **G1 does not bound the false-accept rate among accepted candidates.**
  P(violate | accept) = P(accept AND violate) / P(accept); with alpha = 0.1 and only 10 % of
  candidates accepted this ratio can reach 1. The self-test in `trustlayer._selftest`
  (exchangeable synthetic data, all assumptions true) gives a concrete case:
  interval-accept keeps P(accept AND violate) = 0.029 <= 0.10 but its FDR among
  accepted is 0.262.
* **G2, FDR.** Conformal selection (Jin & Candes 2023) with p_j =
  (1 + #{i : y_i <= tau, mu_i >= mu_j}) / (n + 1) (derived from their clipped score
  V(x, y) = M 1{y > tau} + tau - mu(x); numerically identical to the generic formula,
  max |diff| = 0.0) and Benjamini-Hochberg at q bounds E[FDP] <= q when each candidate
  is exchangeable with the calibration records of its stratum. Same synthetic case:
  FDR = 0.029 <= 0.10, at the price of accepting 2.2 instead of
  45.1 candidates per batch. The bound is for the set BH is run on: joint BH
  bounds the FDR of the whole accepted set, not of a subgroup (e.g. the design-region
  candidates); per-stratum BH bounds it within each stratum, not for the union.
* **G3, shift check.** Conformal novelty p-values (Bates et al. 2023) are super-uniform for
  in-domain candidates, so an in-domain candidate is flagged with probability <= 0.05
  (synthetic check: 0.049). There is no guarantee that a shifted candidate is
  flagged; a shifted candidate that is NOT flagged is decided with source quantiles, for
  which neither G1 nor G2 holds.
* **Correct routing is not enough.** Even a shifted candidate routed to the target stratum
  only has G1 (a bound on P(accept AND violate)) or, under joint BH, G2 for the whole
  accepted set. Neither bounds the error rate among the (few) accepted design-region
  candidates; only BH run inside the target stratum does (the oracle-routing diagnostics D3
  and D4 below show this empirically).
* **None of G1-G3 is a clinical safety or reliability guarantee.** They are statements about
  a measured material property under an exchangeability assumption that the design-region
  shift itself violates for the source stratum. Safety of a biomedical material requires
  biological evidence (ISO 10993 biocompatibility, in-vitro and in-vivo testing); the most a
  trust layer can do is decide which candidates to send to such testing and bound the
  error rate of the statistical screen that precedes it.

## Findings (RF, headline tau = recalibration-pool median, unless stated)

1. **One object, one pass per candidate; what that does and does not cover.** Every arm is the same `TrustLayer` object with a component switched off (A1: shift check off; A2: no target records; A5: shift check and target records off) or its accept rule swapped (A4/A4s: conformal selection). R7 therefore demonstrates manuscript components (1) distribution-free UQ and (2) shift robustness (novelty shift check plus budgeted target recalibration), joined by one accept/reject/measure/not-certified rule. It does NOT demonstrate component (4), incomplete-modality handling: Mondrian groups keyed by a missing-measurement pattern are implemented in `trustlayer.py` but every call here uses groups=None, and only `_selftest` checks their routing; that evaluation belongs to the missing-pattern experiment. Component (3), cost/risk-aware SEQUENTIAL experimentation (manuscript Sec. 4.3), is represented only by a static 'measure' label and a fixed recalibration budget m; there is no sequential acquisition policy here.
2. **Shift check: its false-flag guarantee holds; its power is partial.** Source candidates flagged: 0.014-0.050 (guarantee: <= 0.05). Target candidates flagged: glass 0.21-0.54, steel 0.75, polymer 0.99; AUROC of the novelty p-value for target membership 0.82-0.99. A BH-corrected check would flag 0.00 of glass/steel target candidates (polymer 0.37), which is why the per-candidate rule is used (the shift check does not depend on tau).
3. **Without the shift check, calibrated intervals still accept failing design-region candidates.** Among accepted TARGET candidates the false-accept proportion is 0.35-0.51 for point prediction (A0) and 0.20-0.51 for source-calibrated conformal (A1) on the four glass properties (RF; HistGB 0.38-0.51 and 0.23-0.51). On glass, A1's FDR over the whole candidate pool is only 0.02-0.07, because target candidates are just 4 %-11 % of the glass pool: a pooled error rate hides the design-region failure. This dilution argument is specific to glass. On steel A1 accepts 0.25 candidates per split, so its FDR of 0.00 says nothing. On polymer, target candidates are 36 % of the pool, and A1's target FDR is itself only 0.08 (overall 0.04).
4. **The drop in design-region false accepts comes from the shift check plus abstention (A2), not from target recalibration.** False accepts among target candidates per split (RF, glass): A0 14.3-47.8, A1 5.2-34.2, A2 2.5-6.0, A3 2.8-6.5. Reduction relative to A0 on glass: A2 (no target data) 72 %-87 %, A3 70 %-86 % (HistGB: A2 65 %-88 %, A3 64 %-85 %; polymer RF: A2 100 %, A3 99 %). By construction A3's accept and reject sets contain A2's (in-domain candidates get the same source interval; A2 leaves flagged ones uncertified), so recalibration can only add decisions. It adds target false accepts in 8 of 8 glass dataset-model pairs (equal in 0), and costs m = 22-30 extra measurements, partly offset by the flagged candidates it decides: net extra measurements per split 16.2-26.6 on glass (RF). The design-region accepts that recalibration adds, as false/added per split (pooled false share over 20 splits): glass_Tg RF 0.55/4.65 (12 %); glass_Tg HistGB 0.95/8.15 (12 %); glass_E RF 0.20/0.50 (40 %); glass_E HistGB 1.30/2.05 (63 %); glass_HV RF 0.70/0.90 (78 %); glass_HV HistGB 0.30/0.35 (86 %); glass_Tliq RF 0.20/1.25 (16 %); glass_Tliq HistGB 0.15/0.75 (20 %); polymer_Tg RF 0.30/1.30 (23 %); polymer_Tg HistGB 0.55/5.30 (10 %). So target recalibration with m <= 30 buys very few design-region acceptances, and on some glass properties most of them are false.
5. **Missed shifted candidates carry most of the COUNT of residual false accepts; correct routing does not make the design-region error RATE small.** Under A3, 80 %-95 % of the target false accepts on glass (RF) come from target candidates the shift check missed, which are decided with source quantiles (totals over the 20 splits: glass_Tg 120 of 131; glass_E 51 of 55; glass_HV 56 of 70; glass_Tliq 81 of 85). With oracle routing (D3, diagnostic only) the count falls to 0.20-0.80 per split (HistGB 0.35-1.35). But the few design-region accepts that remain are often false. D3 target FDR: glass_Tg RF 0.06 (9.20 accepts/split, at least one in 100 % of splits, pooled 0.09 = 16/184); glass_Tg HistGB 0.06 (17.85 accepts/split, at least one in 100 % of splits, pooled 0.06 = 22/357); glass_E RF 0.15 (0.95 accepts/split, at least one in 65 % of splits, pooled 0.21 = 4/19); glass_E HistGB 0.45 (2.55 accepts/split, at least one in 90 % of splits, pooled 0.53 = 27/51); glass_HV RF 0.22 (1.20 accepts/split, at least one in 30 % of splits, pooled 0.62 = 15/24); glass_HV HistGB 0.18 (0.85 accepts/split, at least one in 35 % of splits, pooled 0.47 = 8/17); glass_Tliq RF 0.13 (1.85 accepts/split, at least one in 45 % of splits, pooled 0.24 = 9/37); glass_Tliq HistGB 0.12 (1.20 accepts/split, at least one in 35 % of splits, pooled 0.29 = 7/24). D4 (joint-BH selection with oracle routing) target FDR: glass_Tg RF 0.15 (35.20 accepts/split, at least one in 100 % of splits, pooled 0.17 = 121/704); glass_Tg HistGB 0.16 (43.20 accepts/split, at least one in 100 % of splits, pooled 0.17 = 147/864); glass_E RF 0.50 (5.30 accepts/split, at least one in 95 % of splits, pooled 0.57 = 60/106); glass_E HistGB 0.66 (4.20 accepts/split, at least one in 100 % of splits, pooled 0.62 = 52/84); glass_HV RF 0.59 (5.05 accepts/split, at least one in 85 % of splits, pooled 0.51 = 52/101); glass_HV HistGB 0.53 (5.05 accepts/split, at least one in 100 % of splits, pooled 0.43 = 43/101); glass_Tliq RF 0.17 (9.75 accepts/split, at least one in 85 % of splits, pooled 0.24 = 47/195); glass_Tliq HistGB 0.22 (6.40 accepts/split, at least one in 90 % of splits, pooled 0.30 = 38/128). D4's target FDR is higher than A4's (non-oracle) in 4 of 8 glass pairs: glass_E RF (0.50 vs 0.29), glass_E HistGB (0.66 vs 0.29), glass_HV RF (0.59 vs 0.25), glass_HV HistGB (0.53 vs 0.22). This is what the theory predicts. Under D3 the design-region accepts come from target-stratum intervals, and G1 bounds only P(accept AND violate) among target candidates, not the rate among the few accepted. Under D4, joint BH bounds the FDR of the whole accepted set. That set is dominated by in-domain accepts, so the BH threshold q|selected|/n is 0.052-0.093 on glass, well above the smallest target-stratum p-value 1/(m+1) = 0.032-0.043: a target candidate is selected as soon as only one or two of the m target records that fail the specification have a prediction at least as high as its own. Only BH run inside the target stratum (A4s, finding 8) targets the design-region subgroup. Several of these subgroup FDRs rest on fewer than one accept per split, so their percentile ranges span [0, 1] (tables below).
6. **Target recalibration restores coverage only for FLAGGED candidates and rarely produces a decision.** With m = 13-30 target records, A3's miscoverage on flagged candidates is 0.05-0.09 (RF; HistGB 0.04-0.10; nominal 0.10). Across ALL true target candidates A3's miscoverage on glass is 0.23-0.46, because unflagged target candidates keep source quantiles, whose miscoverage on them is 0.34-0.56 (steel 0.76). With oracle routing (D3) the target miscoverage is 0.05-0.10. The target-stratum half-width is 1.04-2.57 x the target IQR (source stratum 0.28-0.70), so A3 decides (accepts or rejects) only 4 %-11 % of flagged candidates on glass, 0 % on steel and 11 % on polymer; the rest go to measurement. Measurements used, including m, are 33 %-67 % of the pool for A3, 28 %-47 % for A2 (no target data) and 20 %-41 % for A1. Recalibration with m <= 30 thus buys a valid 'measure' decision for flagged design-region candidates, rarely a certified acceptance. This is the cost of target data raised in Reviewer 2 comment 7.
7. **FDR-controlled acceptance with one joint BH (A4) controls the accepted set as a whole, not the design-region subgroup.** At the headline tau, A4's FDR over all accepted candidates is 0.000-0.097 (RF; HistGB 0.000-0.101); the largest is glass_Tliq HistGB 0.101, z = 0.2, where z = (FDR - q) / SE and SE is the split-to-split sd divided by sqrt(20). 1 of 12 dataset-model pairs exceed q = 0.10 nominally, none by 2 SE or more. Measurements fall to 10 %-23 % of the pool on glass. Among accepted TARGET candidates, however, A4's FDR is 0.25-0.31 on glass (RF; HistGB 0.22-0.34). At the stricter tau = 75th percentile of the recalibration pool, A4's overall FDR on glass is 0.096-0.118. It exceeds 0.10 in 5 of 8 glass pairs, but by 2 SE or more only for glass_HV RF 0.118 (z = 2.3), glass_HV HistGB 0.117 (z = 2.5), glass_Tliq HistGB 0.111 (z = 3.2); the rest are within noise. Replacing the flag by true membership (D4) gives 0.086-0.101 at this threshold (no pair 2 SE or more above q). So the excess of A4 is consistent with unflagged target candidates being routed to the source stratum, whose calibration records they are not exchangeable with.
8. **BH within each stratum (A4s) controls the flagged stratum, but is all-or-nothing with m <= 30 target records.** The FDR inside the flagged stratum is at most 0.076 (polymer_Tg HistGB, rpool_median) over all datasets, both models and both recalibration-pool thresholds. On polymer (99 % of target candidates flagged): RF: target candidates accepted in 5 of 20 splits, about 166 per split when any (mean over all splits 41.5); target FDP in those splits 0.190 (above q), FDR over all splits 0.048; flagged stratum 517 candidates, smallest possible p = 1/(m+1) = 0.032, so any non-empty BH selection in it has at least k = 167 members (candidates tied at the minimum p: 106.7 on average). HistGB: target candidates accepted in 8 of 20 splits, about 178 per split when any (mean over all splits 71.2); target FDP in those splits 0.185 (above q), FDR over all splits 0.074; flagged stratum 517 candidates, smallest possible p = 1/(m+1) = 0.032, so any non-empty BH selection in it has at least k = 167 members (candidates tied at the minimum p: 121.0 on average). The cause is p-value granularity: with m target records no candidate can have p below 1/(m+1), and BH at q = 0.10 over n_f flagged candidates selects nothing unless about n_f/(10(m+1)) candidates or more tie at that minimum. Per-stratum BH is therefore not a route to many certified design-region acceptances at m <= 30. The FDR of the union of both strata's selections on polymer is 0.123 (HistGB 0.135); per-stratum BH does not bound the union.
9. **Negative control.** Conformal selection calibrated on the source only (A5) has FDR 0.10-0.24 overall and 0.23-0.51 on target candidates (glass and polymer, RF; HistGB 0.10-0.27 and 0.26-0.51): an FDR procedure calibrated on the source does not protect the design region.
10. **Steel is small and its rates are noisy.** The pool has 60 candidates (21 target), the source calibration set 39 records and m = 13. Point prediction accepts 3.0 candidates per split, at FDR 0.28 [0.00, 1.00] (HistGB 0.50 [0.00, 1.00]). The calibrated arms accept almost nothing. Splits (of 20) with any accept: RF: A1 2, A2 2, A3 2, A4 0, A4s 0, A5 0; HistGB: A1 5, A2 5, A3 5, A4 0, A4s 2, A5 1. They are not uniformly cautious, though. Pooled over the 20 splits A3 rejects 70 target candidates (out of 420 target candidates in total), 70 of them unflagged candidates decided with source-quantile intervals, whose miscoverage on them is 0.76; 13 of those rejects (19 %) are false. The design region is otherwise sent to measurement.
11. **Coverage is not an error rate among accepted candidates.** In the exchangeable source subset, interval-accept (A1) has FDR 0.000-0.016 at miscoverage 0.098-0.103; it is conservative in these data, but nothing guarantees that. In the synthetic check with all assumptions true, the same rule has FDR 0.262 with P(accept and violate) = 0.029 <= 0.10. Conformal selection is tight at q on clean data (synthetic homoscedastic case: FDR 0.094; in-domain source candidates here: 0.000-0.087).
12. **How tau is defined matters for small subgroups.** The headline tau is the recalibration-pool median (no target-test label). The task's literal definition, the whole-target-pool median (which uses target-test labels), changes the FDR of any arm by at most 0.015 over all candidates of a dataset (glass and polymer) and by up to 0.050 in their subgroups (true target, flagged); on steel, where one decision moves a rate by 0.2 or more, differences reach 0.175. Largest differences outside steel, headline -> whole-target median: polymer_Tg HistGB D4 flagged 0.150 -> 0.100; glass_E HistGB D4 true_target 0.657 -> 0.702; glass_E RF A4s true_target 0.201 -> 0.157; glass_E HistGB A3 true_target 0.242 -> 0.203. With the whole-target median, A4's largest overall FDR is 0.098. Using only the recalibration-pool records NOT used for recalibration (independent of the target-stratum calibration labels; about 7-13 records on steel, glass_E and glass_HV) moves subgroup FDRs by up to 0.271 outside steel (glass_E HistGB D3 true_target 0.446 -> 0.175) and 0.300 on steel; with 7-13 records that threshold is itself noisy, which is why it is a sensitivity and not the headline. Stability of the statements above across the five definitions (number of the 12 dataset-model cells satisfying each check, in the order rpool_median/rpool_q75/rpool_holdout_median/tgt_median/tgt_q75) - A3 target false accepts <= A1's: 12/12/12/12/12 of 12; A4s FDR inside the flagged stratum <= q: 12/12/12/12/12 of 12; A5 target FDR >= A3 target FDR: 12/12/10/12/12 of 12; A4 overall FDR <= q + 2 SE: 10/7/10/10/8 of 12. The first three checks, which carry findings 4, 8 and 9, hold in essentially every cell under every definition; whether A4's overall FDR exceeds q by more than 2 SE does depend on the threshold, which is finding 7.
13. **What the manuscript can claim.** One object with one decision rule combines distribution-free UQ and shift handling (components 1-2). Incomplete modalities (4) are supported by the same object but evaluated elsewhere, and sequential experimentation (3) is not shown. Coverage (G1) and FDR (G2) hold only for candidates exchangeable with the stratum they are decided with. G2 holds only for the set BH is run on. The shift check (G3) bounds false flags but guarantees nothing about catching shifted candidates. The supported empirical claim, relative to point prediction: the shift check plus abstention (not target recalibration) removes most false accepts among design-region candidates and turns the design region into explicit measurement requests. With m <= 30 target records the layer certifies very few design-region candidates, and those it does certify through intervals or joint BH are often false, even with perfect routing. The design-region error rate is reduced, not bounded. None of this is a clinical safety guarantee.

## Deviations from the protocol and the task specification

* **Specification threshold.** The task defines tau as the median (sensitivity: 75th percentile) of y over the dataset's whole target pool. That uses target-test labels, which PROTOCOL.md forbids for anything selected from target labels. The headline therefore uses the median of the split's recalibration pool (`rpool_median`; sensitivity `rpool_q75`). The recalibration pool contains the m records used for recalibration, so tau is not independent of the target-stratum calibration labels (a small departure from the exchangeability G2 assumes). The variant `rpool_holdout_median` removes that dependence (7-13 records on steel, glass_E and glass_HV, so it is noisy there). The task's literal thresholds (`tgt_median`, `tgt_q75`) are reported as sensitivity only. No threshold is used by any trust-layer component for fitting, calibration, flagging or recalibration. Real differences: finding 12 and the sensitivity table.
* **Candidate pool** = source-test U target-test, as the task specifies. Metrics are reported pooled and split by true membership and by flag.
* **Metric set.** The full interval-metric set (interval score, WIS, calibration error, worst-cluster coverage, AURC) is not recomputed here; R1_core covers it. This script reports decision metrics (FDP/FDR, pooled FDP, power, measurements, false-reject proportion, unconditional false-accept probability) plus interval miscoverage and q/IQR.
* **Extra arms** beyond the specification: A4s (BH per stratum) shows the scope of the FDR guarantee. D3 and D4 (A3/A4 with routing by true target membership) are diagnostics only; they use membership, never property labels.
* **Measurements** include the recalibration budget m for every arm that spends it (A3, A4, A4s, D3, D4).
* **Conformal selection** uses the task's formula, which counts y_i <= tau, i.e. it tests the null Y <= tau. For the specification y >= tau this is valid, and conservative only through calibration labels exactly equal to tau (derivation and numerical check in `trustlayer.selection_pvalues_closed_form` and `_selftest`).
* **Shift check multiplicity.** Each candidate is flagged if p < 0.05, without multiplicity correction (justified in `TrustLayer.shift_check`; the BH variant is reported). Novelty settings (k = 10, level 0.05) were fixed a priori with no tuning.
* **Compute.** joblib ran 6 workers with 2 threads each (threadpoolctl; RF n_jobs = 2).

## Verifier issues (fix round): what changed

All ten issues raised by the independent verifier were adopted; none was rejected.

1. Counts vs rates (D3/D4). Finding 5 now separates the COUNT of residual false accepts (mostly shift-check misses) from the RATE among the few design-region accepts, which stays high even with oracle routing. D3/D4 subgroup FDRs are reported with accepts per split, any-accepted share and pooled FDP.
2. Attribution. Finding 4 credits the reduction to the shift check plus abstention (A2), and reports A3 minus A2 (extra accepts, their false share, extra measurements; table 'What target recalibration adds'). Finding 6 limits 'restores coverage' to flagged candidates and reports miscoverage over all target candidates. Clarification: A3 >= A2 in target false accepts is structural. A3's accept set contains A2's (asserted in the code), so recalibration can only add accepts; the question is whether they are correct.
3. A4s on polymer. Finding 8 and the granularity table report any_accepted, the count when any, fdp_if_any and the minimum BH selection size 1/(m+1)-granularity; the phrase 'many certified acceptances' is gone.
4. Protocol. The headline threshold is now the recalibration-pool median (no target-test label). The task's target-pool thresholds are reported as sensitivity with their real differences (finding 12, deviations section). The earlier claim that the threshold definition changes FDRs by at most about 0.02 was wrong and is withdrawn.
5. Scope of 'one procedure'. Findings 1 and 13 and the component mapping in the `trustlayer.py` docstring now say that R7 exercises components 1-2 plus the decision rule; component 4 is implemented but evaluated elsewhere; component 3 is not sequential here.
6. The pooled-dilution explanation (finding 3) is restricted to glass; steel and polymer are described separately.
7. Steel wording (finding 10): A3's target rejects through source-quantile intervals, any-accept counts per arm, and the small-sample caveat.
8. A4 at the 75th percentile (finding 7): exceedances are tested against split-to-split SE, and D4 is cited as the routing diagnostic.
9. Subgroup tables now carry n accepted, any-accepted share, FDP given any accept and pooled FDP beside every subgroup FDR.
10. The header counts m in measurements for A3, A4, A4s, D3 and D4. The deviations are listed in the md. The G1 per-split inequality is described as a code-consistency check, not as evidence.

### Verifier issues not adopted

None.

## What target recalibration adds: A3 minus A2 (headline tau)

Per split, mean over 20 splits. 'Extra accepts' are candidates accepted by A3 but not by A2 (all flagged, decided with target-stratum intervals); 'pooled' = over all 20 splits.

| dataset | model | m | extra accepts (target) | false among them | pooled false share (target) | splits with any | extra rejects | false among extra rejects | net extra measurements | target FA A2 -> A3 |
|---|---|---|---|---|---|---|---|---|---|---|
| glass_Tg | RF | 30 | 4.65 | 0.55 | 0.12 (11/93) | 1.00 | 0.40 | 0.00 | 22.4 | 6.00 -> 6.55 |
| glass_Tg | HistGB | 30 | 8.15 | 0.95 | 0.12 (19/163) | 1.00 | 0.65 | 0.00 | 16.0 | 4.80 -> 5.75 |
| glass_E | RF | 22 | 0.50 | 0.20 | 0.40 (4/10) | 0.35 | 0.00 | 0.00 | 16.2 | 2.55 -> 2.75 |
| glass_E | HistGB | 22 | 2.05 | 1.30 | 0.63 (26/41) | 0.75 | 0.00 | 0.00 | 15.4 | 2.40 -> 3.70 |
| glass_HV | RF | 23 | 0.90 | 0.70 | 0.78 (14/18) | 0.30 | 0.00 | 0.00 | 21.2 | 2.80 -> 3.50 |
| glass_HV | HistGB | 23 | 0.35 | 0.30 | 0.86 (6/7) | 0.25 | 0.00 | 0.00 | 21.1 | 2.70 -> 3.00 |
| glass_Tliq | RF | 30 | 1.25 | 0.20 | 0.16 (4/25) | 0.45 | 1.20 | 0.00 | 26.6 | 4.05 -> 4.25 |
| glass_Tliq | HistGB | 30 | 0.75 | 0.15 | 0.20 (3/15) | 0.35 | 1.15 | 0.00 | 27.1 | 6.40 -> 6.55 |
| steel_yield | RF | 13 | 0.00 | 0.00 | n/a (0/0) | 0.00 | 0.00 | 0.00 | 13.0 | 0.00 -> 0.00 |
| steel_yield | HistGB | 13 | 0.00 | 0.00 | n/a (0/0) | 0.00 | 0.00 | 0.00 | 13.0 | 0.00 -> 0.00 |
| polymer_Tg | RF | 30 | 1.30 | 0.30 | 0.23 (6/26) | 0.45 | 54.50 | 5.25 | -25.8 | 0.00 -> 0.30 |
| polymer_Tg | HistGB | 30 | 5.30 | 0.55 | 0.10 (11/106) | 0.60 | 52.70 | 2.15 | -28.0 | 0.00 -> 0.55 |

## Per-stratum BH (A4s) inside the flagged stratum: p-value granularity (headline tau)

p_min = 1/(n_target + 1) is the smallest attainable selection p-value; any non-empty BH selection at q = 0.10 among n_f flagged candidates has at least k = ceil(n_f p_min / q) members (asserted per split).

| dataset | model | n_target | p_min | n flagged | k needed | k / n flagged | at p_min | splits with any A4s selection | A4s selected (flagged) | A4 (joint BH) selected (flagged) |
|---|---|---|---|---|---|---|---|---|---|---|
| glass_Tg | RF | 30 | 0.032 | 124.4 | 40.6 | 0.33 | 18.6 | 0.05 | 2.4 | 28.7 |
| glass_Tg | HistGB | 30 | 0.032 | 124.4 | 40.6 | 0.33 | 21.1 | 0.15 | 7.0 | 35.0 |
| glass_E | RF | 22 | 0.043 | 52.0 | 23.1 | 0.45 | 11.4 | 0.00 | 0.0 | 12.9 |
| glass_E | HistGB | 22 | 0.043 | 52.0 | 23.1 | 0.45 | 7.2 | 0.00 | 0.0 | 9.9 |
| glass_HV | RF | 23 | 0.042 | 31.3 | 13.6 | 0.44 | 3.1 | 0.00 | 0.0 | 4.7 |
| glass_HV | HistGB | 23 | 0.042 | 31.3 | 13.6 | 0.44 | 4.0 | 0.00 | 0.0 | 5.8 |
| glass_Tliq | RF | 30 | 0.032 | 91.5 | 30.1 | 0.33 | 10.6 | 0.05 | 1.6 | 12.2 |
| glass_Tliq | HistGB | 30 | 0.032 | 91.5 | 30.1 | 0.33 | 7.6 | 0.00 | 0.0 | 8.6 |
| steel_yield | RF | 13 | 0.071 | 16.2 | 11.9 | 0.74 | 1.2 | 0.00 | 0.0 | 0.0 |
| steel_yield | HistGB | 13 | 0.071 | 16.2 | 11.9 | 0.74 | 0.7 | 0.00 | 0.0 | 0.0 |
| polymer_Tg | RF | 30 | 0.032 | 517.1 | 167.3 | 0.32 | 106.7 | 0.15 | 42.8 | 0.0 |
| polymer_Tg | HistGB | 30 | 0.032 | 517.1 | 167.3 | 0.32 | 121.0 | 0.30 | 72.3 | 0.0 |

## Joint-BH FDR (A4, and D4 with oracle routing) versus q = 0.10, all candidates

SE = split-to-split sd / sqrt(number of splits); z = (FDR - 0.10) / SE.

| dataset | model | tau | A4 FDR | A4 SE | A4 z | D4 FDR | D4 z |
|---|---|---|---|---|---|---|---|
| glass_Tg | RF | rpool_median | 0.096 | 0.003 | -1.3 | 0.090 | -3.4 |
| glass_Tg | RF | rpool_q75 | 0.100 | 0.004 | 0.1 | 0.092 | -1.9 |
| glass_Tg | RF | tgt_median | 0.097 | 0.003 | -1.0 | 0.091 | -3.3 |
| glass_Tg | RF | tgt_q75 | 0.106 | 0.004 | 1.7 | 0.097 | -0.8 |
| glass_Tg | HistGB | rpool_median | 0.096 | 0.003 | -1.4 | 0.092 | -2.4 |
| glass_Tg | HistGB | rpool_q75 | 0.097 | 0.003 | -0.9 | 0.091 | -2.6 |
| glass_Tg | HistGB | tgt_median | 0.098 | 0.002 | -1.0 | 0.093 | -3.2 |
| glass_Tg | HistGB | tgt_q75 | 0.105 | 0.004 | 1.5 | 0.099 | -0.3 |
| glass_E | RF | rpool_median | 0.075 | 0.006 | -4.1 | 0.077 | -3.8 |
| glass_E | RF | rpool_q75 | 0.096 | 0.006 | -0.7 | 0.088 | -2.2 |
| glass_E | RF | tgt_median | 0.090 | 0.004 | -2.4 | 0.092 | -1.9 |
| glass_E | RF | tgt_q75 | 0.107 | 0.005 | 1.4 | 0.099 | -0.2 |
| glass_E | HistGB | rpool_median | 0.077 | 0.007 | -3.5 | 0.078 | -3.5 |
| glass_E | HistGB | rpool_q75 | 0.099 | 0.005 | -0.1 | 0.093 | -1.6 |
| glass_E | HistGB | tgt_median | 0.089 | 0.005 | -2.4 | 0.091 | -1.9 |
| glass_E | HistGB | tgt_q75 | 0.110 | 0.005 | 1.8 | 0.102 | 0.4 |
| glass_HV | RF | rpool_median | 0.065 | 0.003 | -10.9 | 0.063 | -11.9 |
| glass_HV | RF | rpool_q75 | 0.118 | 0.008 | 2.3 | 0.098 | -0.4 |
| glass_HV | RF | tgt_median | 0.066 | 0.003 | -11.0 | 0.064 | -12.4 |
| glass_HV | RF | tgt_q75 | 0.129 | 0.007 | 3.9 | 0.105 | 0.6 |
| glass_HV | HistGB | rpool_median | 0.064 | 0.003 | -10.8 | 0.061 | -11.8 |
| glass_HV | HistGB | rpool_q75 | 0.117 | 0.007 | 2.5 | 0.101 | 0.2 |
| glass_HV | HistGB | tgt_median | 0.065 | 0.003 | -10.9 | 0.063 | -12.1 |
| glass_HV | HistGB | tgt_q75 | 0.123 | 0.006 | 3.9 | 0.102 | 0.3 |
| glass_Tliq | RF | rpool_median | 0.097 | 0.003 | -1.0 | 0.091 | -3.1 |
| glass_Tliq | RF | rpool_q75 | 0.104 | 0.004 | 0.9 | 0.091 | -2.3 |
| glass_Tliq | RF | tgt_median | 0.092 | 0.003 | -2.4 | 0.085 | -3.8 |
| glass_Tliq | RF | tgt_q75 | 0.100 | 0.004 | -0.1 | 0.088 | -3.7 |
| glass_Tliq | HistGB | rpool_median | 0.101 | 0.003 | 0.2 | 0.092 | -2.6 |
| glass_Tliq | HistGB | rpool_q75 | 0.111 | 0.003 | 3.2 | 0.086 | -3.8 |
| glass_Tliq | HistGB | tgt_median | 0.092 | 0.003 | -3.1 | 0.084 | -4.6 |
| glass_Tliq | HistGB | tgt_q75 | 0.108 | 0.004 | 2.0 | 0.083 | -3.9 |
| steel_yield | RF | rpool_median | 0.000 | 0.000 | n/a | 0.000 | n/a |
| steel_yield | RF | rpool_q75 | 0.000 | 0.000 | n/a | 0.000 | n/a |
| steel_yield | RF | tgt_median | 0.000 | 0.000 | n/a | 0.000 | n/a |
| steel_yield | RF | tgt_q75 | 0.000 | 0.000 | n/a | 0.000 | n/a |
| steel_yield | HistGB | rpool_median | 0.000 | 0.000 | n/a | 0.000 | n/a |
| steel_yield | HistGB | rpool_q75 | 0.000 | 0.000 | n/a | 0.000 | n/a |
| steel_yield | HistGB | tgt_median | 0.000 | 0.000 | n/a | 0.000 | n/a |
| steel_yield | HistGB | tgt_q75 | 0.000 | 0.000 | n/a | 0.000 | n/a |
| polymer_Tg | RF | rpool_median | 0.047 | 0.013 | -4.1 | 0.053 | -3.5 |
| polymer_Tg | RF | rpool_q75 | 0.030 | 0.021 | -3.3 | 0.029 | -3.5 |
| polymer_Tg | RF | tgt_median | 0.045 | 0.013 | -4.2 | 0.051 | -3.7 |
| polymer_Tg | RF | tgt_q75 | 0.054 | 0.026 | -1.8 | 0.053 | -1.9 |
| polymer_Tg | HistGB | rpool_median | 0.062 | 0.016 | -2.4 | 0.070 | -1.9 |
| polymer_Tg | HistGB | rpool_q75 | 0.003 | 0.003 | -33.0 | 0.003 | -33.0 |
| polymer_Tg | HistGB | tgt_median | 0.053 | 0.016 | -2.9 | 0.060 | -2.5 |
| polymer_Tg | HistGB | tgt_q75 | 0.009 | 0.006 | -14.2 | 0.009 | -14.2 |

## Results, model RF, headline tau (recalibration-pool median)

### glass_Tg (m = 30; pool 1447 = 1318 source + 129 target candidates; tau = 776.80 [762.90, 798.67] K; source calibration n = 1319)

Acceptable share: source candidates 0.864, target candidates 0.517. Shift check: AUROC 0.882 [0.860, 0.905]; target flagged 0.491 [0.430, 0.563]; source flagged 0.046 [0.031, 0.064]. q_source/IQR_T 0.31, q_target/IQR_T 1.36. RMSE source/target candidates 30.0/107.5.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 1300.2 [1154.9, 1382.6] | 0.087 [0.066, 0.116] | 0.087 | 0.078 [0.063, 0.093] | 0.984 [0.962, 0.994] | 0.0 [0.0, 0.0] | 0.984 [0.962, 0.994] | 0.130 [0.079, 0.186] | n/a |
| A1 conformal | 986.0 [777.3, 1114.0] | 0.043 [0.033, 0.057] | 0.043 | 0.029 [0.023, 0.034] | 0.780 [0.691, 0.835] | 439.6 [326.3, 598.2] | 1.000 [0.998, 1.000] | 0.024 [0.000, 0.149] | 0.144 |
| A2 layer, no target | 902.1 [705.9, 1028.4] | 0.014 [0.009, 0.019] | 0.014 | 0.009 [0.006, 0.012] | 0.736 [0.653, 0.791] | 527.6 [412.9, 680.6] | 1.000 [0.999, 1.000] | 0.017 [0.000, 0.102] | 0.094 |
| A3 layer (m) | 909.4 [709.9, 1037.6] | 0.015 [0.009, 0.020] | 0.014 | 0.009 [0.006, 0.012] | 0.741 [0.656, 0.798] | 550.0 [432.7, 705.1] | 1.000 [0.999, 1.000] | 0.016 [0.000, 0.102] | 0.100 |
| A4 layer + FDR sel. (joint BH) | 1278.0 [1118.4, 1371.2] | 0.096 [0.073, 0.120] | 0.096 | 0.085 [0.068, 0.109] | 0.958 [0.936, 0.977] | 181.6 [101.1, 311.8] | 1.000 [0.999, 1.000] | 0.019 [0.000, 0.111] | 0.100 |
| A4s layer + FDR sel. (BH per stratum) | 1264.5 [1115.0, 1357.5] | 0.102 [0.073, 0.130] | 0.102 | 0.089 [0.066, 0.110] | 0.941 [0.925, 0.962] | 196.2 [116.3, 304.2] | 1.000 [0.999, 1.000] | 0.018 [0.000, 0.148] | 0.100 |
| A5 sel., source only | 1376.3 [1214.3, 1446.0] | 0.127 [0.096, 0.154] | 0.127 | 0.121 [0.094, 0.142] | 0.996 [0.987, 1.000] | 50.9 [0.0, 161.1] | 1.000 [0.998, 1.000] | 0.024 [0.000, 0.167] | 0.144 |
| D3 = A3, oracle routing (diag.) | 902.5 [705.7, 1028.0] | 0.010 [0.005, 0.016] | 0.010 | 0.006 [0.003, 0.009] | 0.739 [0.659, 0.792] | 553.9 [441.8, 705.9] | 1.000 [0.998, 1.000] | 0.025 [0.000, 0.149] | 0.097 |
| D4 = A4, oracle routing (diag.) | 1277.0 [1110.0, 1367.0] | 0.090 [0.069, 0.118] | 0.090 | 0.079 [0.061, 0.109] | 0.964 [0.946, 0.983] | 179.7 [97.7, 301.5] | 1.000 [0.998, 1.000] | 0.029 [0.000, 0.179] | 0.097 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 113.45 | 1.00 | 0.423 [0.346, 0.567] | 0.423 | 0.421 (956/2269) | 47.80 | n/a | 107.20 | 1.00 | 0.381 [0.319, 0.471] | 0.381 | n/a | 12.90 of 65.7 | n/a |
| A1 conformal | 92.70 | 1.00 | 0.372 [0.302, 0.507] | 0.372 | 0.369 (684/1854) | 34.20 | 0.597 | 83.90 | 1.00 | 0.362 [0.273, 0.455] | 0.362 | 0.584 | 6.00 of 65.7 | 0.338 |
| A2 layer, no target | 42.50 | 1.00 | 0.146 [0.063, 0.284] | 0.146 | 0.141 (120/850) | 6.00 | 0.172 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 6.00 of 65.7 | 0.338 |
| A3 layer (m) | 47.15 | 1.00 | 0.144 [0.062, 0.292] | 0.144 | 0.139 (131/943) | 6.55 | 0.229 | 7.20 | 1.00 | 0.076 [0.000, 0.320] | 0.076 | 0.068 | 6.00 of 65.7 | 0.338 |
| A4 layer + FDR sel. (joint BH) | 77.35 | 1.00 | 0.278 [0.156, 0.418] | 0.278 | 0.280 (433/1547) | 21.65 | 0.229 | 28.70 | 1.00 | 0.150 [0.049, 0.318] | 0.150 | 0.068 | 17.00 of 65.7 | 0.338 |
| A4s layer + FDR sel. (BH per stratum) | 62.55 | 1.00 | 0.296 [0.139, 0.462] | 0.296 | 0.297 (372/1251) | 18.60 | 0.229 | 2.35 | 0.05 | 0.015 [0.000, 0.156] | 0.298 | 0.068 | 17.90 of 65.7 | 0.338 |
| A5 sel., source only | 122.10 | 1.00 | 0.456 [0.358, 0.589] | 0.456 | 0.454 (1109/2442) | 55.45 | 0.597 | 114.95 | 1.00 | 0.410 [0.357, 0.484] | 0.410 | 0.584 | 17.90 of 65.7 | 0.338 |
| D3 = A3, oracle routing (diag.) | 9.20 | 1.00 | 0.062 [0.000, 0.285] | 0.062 | 0.087 (16/184) | 0.80 | 0.070 | 38.35 | 1.00 | 0.063 [0.022, 0.180] | 0.063 | 0.198 | 0.25 of 65.7 | 0.025 |
| D4 = A4, oracle routing (diag.) | 35.20 | 1.00 | 0.153 [0.052, 0.289] | 0.153 | 0.172 (121/704) | 6.05 | 0.070 | 69.95 | 1.00 | 0.177 [0.091, 0.295] | 0.177 | 0.198 | 1.40 of 65.7 | 0.025 |

### glass_E (m = 22; pool 742 = 709 source + 33 target candidates; tau = 65.90 [61.40, 69.03] GPa; source calibration n = 709)

Acceptable share: source candidates 0.902, target candidates 0.494. Shift check: AUROC 0.889 [0.851, 0.917]; target flagged 0.521 [0.408, 0.606]; source flagged 0.049 [0.032, 0.068]. q_source/IQR_T 0.38, q_target/IQR_T 1.83. RMSE source/target candidates 6.0/19.7.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 686.3 [632.8, 733.0] | 0.057 [0.035, 0.089] | 0.057 | 0.053 [0.035, 0.078] | 0.987 [0.974, 0.999] | 0.0 [0.0, 0.0] | 0.987 [0.974, 0.999] | 0.152 [0.023, 0.239] | n/a |
| A1 conformal | 542.1 [431.0, 659.2] | 0.042 [0.023, 0.065] | 0.041 | 0.030 [0.020, 0.041] | 0.789 [0.674, 0.910] | 190.9 [79.4, 304.8] | 0.999 [0.996, 1.000] | 0.059 [0.000, 0.237] | 0.133 |
| A2 layer, no target | 501.1 [388.4, 613.2] | 0.015 [0.007, 0.029] | 0.015 | 0.010 [0.005, 0.016] | 0.749 [0.633, 0.858] | 232.3 [125.4, 347.4] | 0.999 [0.997, 1.000] | 0.059 [0.000, 0.212] | 0.091 |
| A3 layer (m) | 506.9 [392.5, 621.4] | 0.016 [0.008, 0.028] | 0.015 | 0.010 [0.005, 0.016] | 0.758 [0.639, 0.870] | 248.6 [139.1, 365.3] | 0.999 [0.997, 1.000] | 0.059 [0.000, 0.212] | 0.095 |
| A4 layer + FDR sel. (joint BH) | 682.0 [626.3, 710.0] | 0.075 [0.036, 0.115] | 0.075 | 0.069 [0.034, 0.104] | 0.962 [0.942, 0.973] | 76.1 [52.0, 125.3] | 0.999 [0.997, 1.000] | 0.072 [0.000, 0.216] | 0.095 |
| A4s layer + FDR sel. (BH per stratum) | 672.0 [617.3, 702.0] | 0.076 [0.037, 0.122] | 0.076 | 0.069 [0.034, 0.108] | 0.947 [0.927, 0.964] | 87.3 [62.0, 134.0] | 0.999 [0.997, 1.000] | 0.059 [0.000, 0.216] | 0.095 |
| A5 sel., source only | 723.1 [679.4, 742.0] | 0.095 [0.047, 0.144] | 0.095 | 0.092 [0.047, 0.138] | 0.998 [0.990, 1.000] | 13.9 [0.0, 50.1] | 0.999 [0.997, 1.000] | 0.054 [0.000, 0.207] | 0.133 |
| D3 = A3, oracle routing (diag.) | 511.4 [401.0, 627.1] | 0.012 [0.006, 0.021] | 0.012 | 0.009 [0.004, 0.014] | 0.767 [0.661, 0.878] | 243.7 [133.4, 356.8] | 0.999 [0.996, 1.000] | 0.059 [0.000, 0.237] | 0.102 |
| D4 = A4, oracle routing (diag.) | 693.2 [650.4, 720.8] | 0.077 [0.037, 0.116] | 0.076 | 0.071 [0.035, 0.107] | 0.976 [0.968, 0.987] | 65.0 [42.3, 100.8] | 0.999 [0.997, 1.000] | 0.063 [0.000, 0.207] | 0.102 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 33.00 | 1.00 | 0.506 [0.257, 0.713] | 0.506 | 0.506 (334/660) | 16.70 | n/a | 49.60 | 1.00 | 0.338 [0.204, 0.449] | 0.338 | n/a | 3.10 of 15.8 | n/a |
| A1 conformal | 31.70 | 1.00 | 0.510 [0.257, 0.724] | 0.510 | 0.506 (321/634) | 16.05 | 0.770 | 41.05 | 1.00 | 0.371 [0.184, 0.515] | 0.371 | 0.606 | 2.55 of 15.8 | 0.561 |
| A2 layer, no target | 14.90 | 1.00 | 0.179 [0.000, 0.388] | 0.179 | 0.171 (51/298) | 2.55 | 0.268 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 2.55 of 15.8 | 0.561 |
| A3 layer (m) | 15.40 | 1.00 | 0.189 [0.000, 0.388] | 0.189 | 0.179 (55/308) | 2.75 | 0.330 | 5.75 | 0.95 | 0.053 [0.000, 0.294] | 0.056 | 0.056 | 2.55 of 15.8 | 0.561 |
| A4 layer + FDR sel. (joint BH) | 19.05 | 1.00 | 0.293 [0.023, 0.563] | 0.293 | 0.297 (113/381) | 5.65 | 0.330 | 12.95 | 1.00 | 0.208 [0.000, 0.552] | 0.208 | 0.056 | 3.10 of 15.8 | 0.561 |
| A4s layer + FDR sel. (BH per stratum) | 15.80 | 1.00 | 0.201 [0.000, 0.393] | 0.201 | 0.196 (62/316) | 3.10 | 0.330 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.056 | 3.10 of 15.8 | 0.561 |
| A5 sel., source only | 33.00 | 1.00 | 0.506 [0.257, 0.713] | 0.506 | 0.506 (334/660) | 16.70 | 0.770 | 51.05 | 1.00 | 0.346 [0.204, 0.446] | 0.346 | 0.606 | 3.10 of 15.8 | 0.561 |
| D3 = A3, oracle routing (diag.) | 0.95 | 0.65 | 0.150 [0.000, 1.000] | 0.231 | 0.211 (4/19) | 0.20 | 0.079 | 24.75 | 1.00 | 0.064 [0.000, 0.206] | 0.064 | 0.319 | 0.00 of 15.8 | 0.035 |
| D4 = A4, oracle routing (diag.) | 5.30 | 0.95 | 0.502 [0.000, 1.000] | 0.529 | 0.566 (60/106) | 3.00 | 0.079 | 37.25 | 1.00 | 0.178 [0.059, 0.329] | 0.178 | 0.319 | 0.20 of 15.8 | 0.035 |

### glass_HV (m = 23; pool 330 = 295 source + 35 target candidates; tau = 4.21 [4.17, 4.26] GPa; source calibration n = 294)

Acceptable share: source candidates 0.945, target candidates 0.520. Shift check: AUROC 0.939 [0.915, 0.960]; target flagged 0.539 [0.299, 0.686]; source flagged 0.042 [0.019, 0.066]. q_source/IQR_T 0.70, q_target/IQR_T 2.57. RMSE source/target candidates 0.7/2.1.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 319.2 [313.0, 326.5] | 0.080 [0.060, 0.094] | 0.080 | 0.077 [0.057, 0.091] | 0.989 [0.973, 1.000] | 0.0 [0.0, 0.0] | 0.989 [0.973, 1.000] | 0.255 [0.000, 0.515] | n/a |
| A1 conformal | 262.9 [217.9, 290.7] | 0.074 [0.043, 0.099] | 0.074 | 0.059 [0.032, 0.076] | 0.820 [0.682, 0.905] | 65.9 [39.3, 111.1] | 0.997 [0.990, 1.000] | 0.861 [0.500, 1.000] | 0.175 |
| A2 layer, no target | 236.7 [199.4, 262.6] | 0.026 [0.014, 0.047] | 0.026 | 0.018 [0.009, 0.030] | 0.777 [0.650, 0.859] | 92.2 [67.0, 129.5] | 0.997 [0.990, 1.000] | 0.861 [0.500, 1.000] | 0.109 |
| A3 layer (m) | 238.5 [202.9, 264.5] | 0.029 [0.015, 0.058] | 0.028 | 0.020 [0.011, 0.038] | 0.780 [0.656, 0.859] | 113.4 [86.4, 149.0] | 0.997 [0.990, 1.000] | 0.861 [0.500, 1.000] | 0.114 |
| A4 layer + FDR sel. (joint BH) | 303.4 [290.0, 315.5] | 0.065 [0.044, 0.089] | 0.065 | 0.060 [0.041, 0.083] | 0.955 [0.911, 0.983] | 49.6 [37.5, 63.0] | 1.000 [1.000, 1.000] | n/a | 0.114 |
| A4s layer + FDR sel. (BH per stratum) | 298.7 [289.5, 311.6] | 0.059 [0.040, 0.083] | 0.059 | 0.054 [0.036, 0.076] | 0.946 [0.909, 0.978] | 54.3 [41.4, 63.5] | 1.000 [1.000, 1.000] | n/a | 0.114 |
| A5 sel., source only | 330.0 [330.0, 330.0] | 0.100 [0.080, 0.124] | 0.100 | 0.100 [0.080, 0.124] | 1.000 [1.000, 1.000] | 0.0 [0.0, 0.0] | 1.000 [1.000, 1.000] | n/a | 0.175 |
| D3 = A3, oracle routing (diag.) | 232.8 [192.2, 258.5] | 0.019 [0.004, 0.036] | 0.019 | 0.013 [0.003, 0.027] | 0.769 [0.632, 0.847] | 119.1 [92.4, 159.8] | 0.997 [0.990, 1.000] | 0.861 [0.500, 1.000] | 0.099 |
| D4 = A4, oracle routing (diag.) | 300.1 [295.0, 313.5] | 0.063 [0.042, 0.089] | 0.063 | 0.057 [0.038, 0.083] | 0.947 [0.927, 0.973] | 53.0 [39.5, 58.0] | 1.000 [1.000, 1.000] | n/a | 0.099 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 34.80 | 1.00 | 0.483 [0.371, 0.629] | 0.483 | 0.483 (336/696) | 16.80 | n/a | 31.25 | 1.00 | 0.510 [0.311, 0.710] | 0.510 | n/a | 2.80 of 16.1 | n/a |
| A1 conformal | 31.40 | 1.00 | 0.501 [0.336, 0.678] | 0.501 | 0.505 (317/628) | 15.85 | 0.790 | 26.25 | 1.00 | 0.525 [0.265, 0.750] | 0.525 | 0.697 | 2.80 of 16.1 | 0.520 |
| A2 layer, no target | 13.50 | 1.00 | 0.191 [0.080, 0.364] | 0.191 | 0.207 (56/270) | 2.80 | 0.253 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 2.80 of 16.1 | 0.520 |
| A3 layer (m) | 14.40 | 1.00 | 0.218 [0.084, 0.428] | 0.218 | 0.243 (70/288) | 3.50 | 0.289 | 1.80 | 0.40 | 0.153 [0.000, 0.762] | 0.383 | 0.059 | 2.80 of 16.1 | 0.520 |
| A4 layer + FDR sel. (joint BH) | 18.80 | 1.00 | 0.245 [0.139, 0.396] | 0.245 | 0.255 (96/376) | 4.80 | 0.289 | 4.65 | 0.95 | 0.479 [0.000, 1.000] | 0.504 | 0.059 | 2.80 of 16.1 | 0.520 |
| A4s layer + FDR sel. (BH per stratum) | 16.15 | 1.00 | 0.160 [0.067, 0.319] | 0.160 | 0.173 (56/323) | 2.80 | 0.289 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.059 | 2.80 of 16.1 | 0.520 |
| A5 sel., source only | 35.00 | 1.00 | 0.480 [0.371, 0.629] | 0.480 | 0.480 (336/700) | 16.80 | 0.790 | 31.30 | 1.00 | 0.508 [0.311, 0.695] | 0.508 | 0.697 | 2.80 of 16.1 | 0.520 |
| D3 = A3, oracle routing (diag.) | 1.20 | 0.30 | 0.221 [0.000, 1.000] | 0.737 | 0.625 (15/24) | 0.75 | 0.067 | 9.25 | 1.00 | 0.126 [0.000, 0.383] | 0.126 | 0.140 | 0.05 of 16.1 | 0.071 |
| D4 = A4, oracle routing (diag.) | 5.05 | 0.85 | 0.588 [0.000, 1.000] | 0.692 | 0.515 (52/101) | 2.60 | 0.067 | 15.10 | 1.00 | 0.237 [0.051, 0.370] | 0.237 | 0.140 | 0.60 of 16.1 | 0.071 |

### glass_Tliq (m = 30; pool 1617 = 1552 source + 65 target candidates; tau = 1364.15 [1324.21, 1389.59] K; source calibration n = 1552)

Acceptable share: source candidates 0.523, target candidates 0.476. Shift check: AUROC 0.816 [0.789, 0.838]; target flagged 0.208 [0.153, 0.254]; source flagged 0.050 [0.036, 0.063]. q_source/IQR_T 0.36, q_target/IQR_T 1.41. RMSE source/target candidates 49.5/190.9.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 842.8 [720.8, 1022.8] | 0.083 [0.066, 0.104] | 0.082 | 0.043 [0.035, 0.051] | 0.916 [0.877, 0.954] | 0.0 [0.0, 0.0] | 0.916 [0.877, 0.954] | 0.090 [0.060, 0.114] | n/a |
| A1 conformal | 494.8 [351.0, 694.0] | 0.020 [0.012, 0.033] | 0.020 | 0.006 [0.003, 0.009] | 0.570 [0.462, 0.677] | 659.5 [594.4, 719.8] | 0.989 [0.979, 0.995] | 0.021 [0.010, 0.039] | 0.119 |
| A2 layer, no target | 476.3 [337.8, 672.6] | 0.015 [0.009, 0.023] | 0.015 | 0.004 [0.002, 0.007] | 0.551 [0.446, 0.658] | 718.6 [654.8, 772.6] | 0.992 [0.985, 0.997] | 0.015 [0.007, 0.025] | 0.095 |
| A3 layer (m) | 478.6 [337.8, 676.6] | 0.015 [0.009, 0.024] | 0.015 | 0.004 [0.002, 0.007] | 0.553 [0.446, 0.662] | 745.2 [681.3, 801.0] | 0.992 [0.985, 0.997] | 0.014 [0.007, 0.025] | 0.098 |
| A4 layer + FDR sel. (joint BH) | 859.0 [727.9, 1057.9] | 0.097 [0.077, 0.118] | 0.098 | 0.052 [0.037, 0.069] | 0.918 [0.880, 0.960] | 364.7 [310.7, 410.8] | 0.992 [0.985, 0.997] | 0.014 [0.007, 0.025] | 0.098 |
| A4s layer + FDR sel. (BH per stratum) | 855.7 [729.8, 1054.6] | 0.101 [0.083, 0.122] | 0.102 | 0.054 [0.040, 0.073] | 0.911 [0.873, 0.946] | 368.1 [310.6, 422.6] | 0.992 [0.985, 0.997] | 0.014 [0.007, 0.025] | 0.098 |
| A5 sel., source only | 888.0 [753.4, 1096.8] | 0.107 [0.083, 0.129] | 0.108 | 0.059 [0.042, 0.082] | 0.938 [0.901, 0.975] | 266.3 [213.8, 320.1] | 0.989 [0.979, 0.995] | 0.021 [0.010, 0.039] | 0.119 |
| D3 = A3, oracle routing (diag.) | 471.9 [331.9, 657.2] | 0.011 [0.004, 0.023] | 0.010 | 0.003 [0.001, 0.005] | 0.548 [0.441, 0.647] | 722.8 [654.4, 777.3] | 0.990 [0.981, 0.995] | 0.019 [0.010, 0.037] | 0.099 |
| D4 = A4, oracle routing (diag.) | 846.9 [705.8, 1039.9] | 0.091 [0.072, 0.115] | 0.091 | 0.048 [0.031, 0.065] | 0.911 [0.865, 0.945] | 347.8 [300.9, 406.2] | 0.990 [0.981, 0.995] | 0.019 [0.010, 0.037] | 0.099 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 40.00 | 1.00 | 0.354 [0.263, 0.494] | 0.354 | 0.357 (286/800) | 14.30 | n/a | 32.95 | 1.00 | 0.251 [0.173, 0.349] | 0.251 | n/a | 10.40 of 51.5 | n/a |
| A1 conformal | 24.80 | 1.00 | 0.205 [0.091, 0.310] | 0.205 | 0.208 (103/496) | 5.15 | 0.594 | 18.50 | 1.00 | 0.149 [0.000, 0.368] | 0.149 | 0.426 | 4.05 of 51.5 | 0.520 |
| A2 layer, no target | 21.40 | 1.00 | 0.184 [0.077, 0.286] | 0.184 | 0.189 (81/428) | 4.05 | 0.412 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 4.05 of 51.5 | 0.520 |
| A3 layer (m) | 22.65 | 1.00 | 0.184 [0.077, 0.282] | 0.184 | 0.188 (85/453) | 4.25 | 0.460 | 2.25 | 0.70 | 0.080 [0.000, 0.620] | 0.114 | 0.049 | 4.05 of 51.5 | 0.520 |
| A4 layer + FDR sel. (joint BH) | 37.50 | 1.00 | 0.310 [0.201, 0.475] | 0.310 | 0.313 (235/750) | 11.75 | 0.460 | 12.15 | 1.00 | 0.112 [0.000, 0.287] | 0.112 | 0.049 | 10.95 of 51.5 | 0.520 |
| A4s layer + FDR sel. (BH per stratum) | 35.15 | 1.00 | 0.308 [0.211, 0.475] | 0.308 | 0.317 (223/703) | 11.15 | 0.460 | 1.55 | 0.05 | 0.010 [0.000, 0.102] | 0.194 | 0.049 | 11.05 of 51.5 | 0.520 |
| A5 sel., source only | 41.75 | 1.00 | 0.370 [0.264, 0.507] | 0.370 | 0.376 (314/835) | 15.70 | 0.594 | 35.55 | 1.00 | 0.279 [0.176, 0.375] | 0.279 | 0.426 | 11.05 of 51.5 | 0.520 |
| D3 = A3, oracle routing (diag.) | 1.85 | 0.45 | 0.134 [0.000, 0.715] | 0.298 | 0.243 (9/37) | 0.45 | 0.102 | 16.35 | 1.00 | 0.109 [0.000, 0.311] | 0.109 | 0.328 | 0.25 of 51.5 | 0.068 |
| D4 = A4, oracle routing (diag.) | 9.75 | 0.85 | 0.174 [0.000, 0.399] | 0.205 | 0.241 (47/195) | 2.35 | 0.102 | 31.10 | 1.00 | 0.190 [0.105, 0.316] | 0.190 | 0.328 | 1.60 of 51.5 | 0.068 |

### steel_yield (m = 13; pool 60 = 39 source + 21 target candidates; tau = 1647.09 [1472.75, 1750.66] MPa; source calibration n = 39)

Acceptable share: source candidates 0.118, target candidates 0.486. Shift check: AUROC 0.984 [0.969, 1.000]; target flagged 0.745 [0.424, 0.977]; source flagged 0.014 [0.000, 0.051]. q_source/IQR_T 0.28, q_target/IQR_T 1.54. RMSE source/target candidates 109.1/577.5.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 3.0 [0.0, 10.2] | 0.279 [0.000, 1.000] | 0.246 | 0.013 [0.000, 0.042] | 0.119 [0.000, 0.345] | 0.0 [0.0, 0.0] | 0.119 [0.000, 0.345] | 0.223 [0.075, 0.386] | n/a |
| A1 conformal | 0.2 [0.0, 2.5] | 0.000 [0.000, 0.000] | 0.000 | 0.000 [0.000, 0.000] | 0.010 [0.000, 0.097] | 14.4 [4.5, 34.5] | 0.359 [0.068, 0.638] | 0.211 [0.063, 0.432] | 0.360 |
| A2 layer, no target | 0.2 [0.0, 2.5] | 0.000 [0.000, 0.000] | 0.000 | 0.000 [0.000, 0.000] | 0.010 [0.000, 0.097] | 27.4 [16.4, 45.6] | 0.933 [0.711, 1.000] | 0.030 [0.000, 0.134] | 0.127 |
| A3 layer (m) | 0.2 [0.0, 2.5] | 0.000 [0.000, 0.000] | 0.000 | 0.000 [0.000, 0.000] | 0.010 [0.000, 0.097] | 40.4 [29.4, 58.6] | 0.933 [0.711, 1.000] | 0.030 [0.000, 0.134] | 0.143 |
| A4 layer + FDR sel. (joint BH) | 0.0 [0.0, 0.0] | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 40.6 [29.4, 61.1] | 0.933 [0.711, 1.000] | 0.030 [0.000, 0.134] | 0.143 |
| A4s layer + FDR sel. (BH per stratum) | 0.0 [0.0, 0.0] | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 40.6 [29.4, 61.1] | 0.933 [0.711, 1.000] | 0.030 [0.000, 0.134] | 0.143 |
| A5 sel., source only | 0.0 [0.0, 0.0] | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 14.7 [4.5, 37.0] | 0.359 [0.068, 0.638] | 0.211 [0.063, 0.432] | 0.360 |
| D3 = A3, oracle routing (diag.) | 0.2 [0.0, 2.5] | 0.000 [0.000, 0.000] | 0.000 | 0.000 [0.000, 0.000] | 0.010 [0.000, 0.097] | 43.6 [37.0, 58.1] | 0.976 [0.872, 1.000] | 0.014 [0.000, 0.081] | 0.081 |
| D4 = A4, oracle routing (diag.) | 0.0 [0.0, 0.0] | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 43.9 [37.0, 60.6] | 0.976 [0.872, 1.000] | 0.014 [0.000, 0.081] | 0.081 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 0.30 | 0.25 | 0.225 [0.000, 1.000] | 0.900 | 0.833 (5/6) | 0.25 | n/a | 0.40 | 0.30 | 0.175 [0.000, 1.000] | 0.583 | n/a | 0.05 of 5.6 | n/a |
| A1 conformal | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.845 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.869 | 0.00 of 5.6 | 0.758 |
| A2 layer, no target | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.186 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 0.00 of 5.6 | 0.758 |
| A3 layer (m) | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.233 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.067 | 0.00 of 5.6 | 0.758 |
| A4 layer + FDR sel. (joint BH) | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.233 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.067 | 0.00 of 5.6 | 0.758 |
| A4s layer + FDR sel. (BH per stratum) | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.233 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.067 | 0.00 of 5.6 | 0.758 |
| A5 sel., source only | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.845 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.869 | 0.00 of 5.6 | 0.758 |
| D3 = A3, oracle routing (diag.) | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.048 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.074 | 0.00 of 5.6 | 0.000 |
| D4 = A4, oracle routing (diag.) | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.048 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.074 | 0.00 of 5.6 | 0.000 |

### polymer_Tg (m = 30; pool 1346 = 866 source + 480 target candidates; tau = 511.73 [506.57, 518.77] K; source calibration n = 867)

Acceptable share: source candidates 0.075, target candidates 0.497. Shift check: AUROC 0.987 [0.981, 0.991]; target flagged 0.992 [0.960, 1.000]; source flagged 0.047 [0.036, 0.061]. q_source/IQR_T 0.67, q_target/IQR_T 1.04. RMSE source/target candidates 49.7/74.5.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 222.8 [179.5, 279.3] | 0.175 [0.111, 0.261] | 0.178 | 0.030 [0.017, 0.052] | 0.603 [0.525, 0.694] | 0.0 [0.0, 0.0] | 0.603 [0.525, 0.694] | 0.107 [0.085, 0.131] | n/a |
| A1 conformal | 18.9 [6.0, 45.8] | 0.039 [0.000, 0.139] | 0.040 | 0.001 [0.000, 0.002] | 0.059 [0.020, 0.137] | 508.4 [454.6, 555.7] | 0.934 [0.888, 0.985] | 0.024 [0.006, 0.038] | 0.148 |
| A2 layer, no target | 9.9 [6.0, 16.7] | 0.013 [0.000, 0.102] | 0.015 | 0.000 [0.000, 0.001] | 0.032 [0.019, 0.052] | 636.0 [600.3, 664.5] | 0.982 [0.972, 0.992] | 0.008 [0.004, 0.012] | 0.059 |
| A3 layer (m) | 11.2 [6.0, 18.6] | 0.037 [0.000, 0.118] | 0.040 | 0.000 [0.000, 0.001] | 0.035 [0.019, 0.057] | 610.1 [528.9, 662.9] | 0.964 [0.917, 0.992] | 0.014 [0.003, 0.029] | 0.095 |
| A4 layer + FDR sel. (joint BH) | 17.8 [0.0, 36.5] | 0.047 [0.000, 0.173] | 0.079 | 0.001 [0.000, 0.004] | 0.054 [0.000, 0.109] | 603.5 [527.6, 667.0] | 0.964 [0.917, 0.992] | 0.014 [0.003, 0.029] | 0.095 |
| A4s layer + FDR sel. (BH per stratum) | 74.1 [13.0, 341.2] | 0.123 [0.000, 0.351] | 0.246 | 0.014 [0.000, 0.090] | 0.186 [0.041, 0.727] | 547.2 [277.1, 640.4] | 0.964 [0.917, 0.992] | 0.014 [0.003, 0.029] | 0.095 |
| A5 sel., source only | 291.9 [99.2, 388.1] | 0.240 [0.055, 0.349] | 0.263 | 0.057 [0.004, 0.095] | 0.707 [0.303, 0.833] | 235.3 [164.6, 400.2] | 0.934 [0.888, 0.985] | 0.024 [0.006, 0.038] | 0.148 |
| D3 = A3, oracle routing (diag.) | 11.2 [6.0, 18.6] | 0.037 [0.000, 0.118] | 0.040 | 0.000 [0.000, 0.001] | 0.035 [0.019, 0.057] | 604.5 [526.7, 647.9] | 0.964 [0.917, 0.992] | 0.014 [0.003, 0.028] | 0.098 |
| D4 = A4, oracle routing (diag.) | 19.1 [0.0, 37.0] | 0.053 [0.000, 0.173] | 0.081 | 0.001 [0.000, 0.004] | 0.058 [0.000, 0.110] | 596.6 [520.7, 657.2] | 0.964 [0.917, 0.992] | 0.014 [0.003, 0.028] | 0.098 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 169.25 | 1.00 | 0.163 [0.105, 0.267] | 0.163 | 0.168 (568/3385) | 28.40 | n/a | 171.35 | 1.00 | 0.166 [0.104, 0.269] | 0.166 | n/a | 0.06 of 4.6 | n/a |
| A1 conformal | 9.00 | 0.90 | 0.079 [0.000, 0.333] | 0.088 | 0.067 (12/180) | 0.60 | 0.237 | 9.00 | 0.90 | 0.079 [0.000, 0.333] | 0.088 | 0.232 | 0.00 of 4.6 | 0.314 |
| A2 layer, no target | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.002 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 0.00 of 4.6 | 0.314 |
| A3 layer (m) | 1.30 | 0.45 | 0.133 [0.000, 0.762] | 0.296 | 0.231 (6/26) | 0.30 | 0.099 | 1.30 | 0.45 | 0.133 [0.000, 0.762] | 0.296 | 0.094 | 0.00 of 4.6 | 0.314 |
| A4 layer + FDR sel. (joint BH) | 0.05 | 0.05 | 0.000 [0.000, 0.000] | 0.000 | 0.000 (0/1) | 0.00 | 0.099 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.094 | 0.00 of 4.6 | 0.314 |
| A4s layer + FDR sel. (BH per stratum) | 41.50 | 0.25 | 0.048 [0.000, 0.352] | 0.190 | 0.333 (276/830) | 13.80 | 0.099 | 42.75 | 0.15 | 0.049 [0.000, 0.362] | 0.323 | 0.094 | 0.00 of 4.6 | 0.314 |
| A5 sel., source only | 223.70 | 1.00 | 0.229 [0.049, 0.354] | 0.229 | 0.252 (1126/4474) | 56.30 | 0.237 | 226.70 | 1.00 | 0.232 [0.046, 0.359] | 0.232 | 0.232 | 0.12 of 4.6 | 0.314 |
| D3 = A3, oracle routing (diag.) | 1.30 | 0.45 | 0.133 [0.000, 0.762] | 0.296 | 0.231 (6/26) | 0.30 | 0.098 | 1.30 | 0.45 | 0.133 [0.000, 0.762] | 0.296 | 0.104 | 0.00 of 4.6 | 0.080 |
| D4 = A4, oracle routing (diag.) | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.098 | 0.55 | 0.25 | 0.100 [0.000, 1.000] | 0.400 | 0.104 | 0.00 of 4.6 | 0.080 |

## Results, model HistGB, headline tau (recalibration-pool median)

### glass_Tg (m = 30; pool 1447 = 1318 source + 129 target candidates; tau = 776.80 [762.90, 798.67] K; source calibration n = 1319)

Acceptable share: source candidates 0.864, target candidates 0.517. Shift check: AUROC 0.882 [0.860, 0.905]; target flagged 0.491 [0.430, 0.563]; source flagged 0.046 [0.031, 0.064]. q_source/IQR_T 0.29, q_target/IQR_T 1.24. RMSE source/target candidates 28.3/99.3.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 1272.5 [1118.8, 1364.0] | 0.075 [0.059, 0.093] | 0.075 | 0.066 [0.055, 0.078] | 0.976 [0.957, 0.987] | 0.0 [0.0, 0.0] | 0.976 [0.957, 0.987] | 0.175 [0.109, 0.245] | n/a |
| A1 conformal | 984.8 [778.9, 1126.5] | 0.037 [0.030, 0.045] | 0.036 | 0.025 [0.019, 0.031] | 0.785 [0.707, 0.837] | 422.1 [304.8, 581.0] | 0.998 [0.997, 1.000] | 0.063 [0.000, 0.186] | 0.143 |
| A2 layer, no target | 908.0 [717.0, 1047.8] | 0.012 [0.007, 0.018] | 0.012 | 0.008 [0.004, 0.012] | 0.742 [0.668, 0.796] | 508.6 [387.3, 660.6] | 0.999 [0.998, 1.000] | 0.044 [0.000, 0.158] | 0.096 |
| A3 layer (m) | 921.4 [727.0, 1060.4] | 0.013 [0.008, 0.019] | 0.013 | 0.008 [0.006, 0.013] | 0.752 [0.676, 0.808] | 524.5 [404.3, 679.2] | 0.999 [0.998, 1.000] | 0.043 [0.000, 0.151] | 0.103 |
| A4 layer + FDR sel. (joint BH) | 1282.0 [1099.2, 1373.6] | 0.096 [0.070, 0.118] | 0.096 | 0.085 [0.059, 0.105] | 0.961 [0.939, 0.979] | 166.2 [94.7, 307.9] | 0.999 [0.998, 1.000] | 0.040 [0.000, 0.148] | 0.103 |
| A4s layer + FDR sel. (BH per stratum) | 1263.8 [1095.3, 1357.0] | 0.100 [0.071, 0.124] | 0.100 | 0.087 [0.062, 0.108] | 0.943 [0.921, 0.975] | 186.8 [118.9, 312.2] | 0.999 [0.998, 1.000] | 0.049 [0.000, 0.325] | 0.103 |
| A5 sel., source only | 1367.1 [1180.5, 1445.1] | 0.122 [0.091, 0.145] | 0.122 | 0.116 [0.084, 0.135] | 0.994 [0.977, 1.000] | 46.0 [0.0, 179.6] | 0.999 [0.997, 1.000] | 0.045 [0.000, 0.205] | 0.143 |
| D3 = A3, oracle routing (diag.) | 918.6 [732.6, 1044.9] | 0.009 [0.006, 0.015] | 0.009 | 0.006 [0.004, 0.010] | 0.753 [0.685, 0.803] | 526.4 [416.4, 668.5] | 0.999 [0.997, 1.000] | 0.046 [0.000, 0.179] | 0.098 |
| D4 = A4, oracle routing (diag.) | 1286.8 [1094.2, 1384.8] | 0.092 [0.063, 0.115] | 0.092 | 0.082 [0.052, 0.107] | 0.969 [0.942, 0.987] | 161.2 [84.2, 307.0] | 0.999 [0.997, 1.000] | 0.042 [0.000, 0.152] | 0.098 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 102.65 | 1.00 | 0.383 [0.302, 0.505] | 0.383 | 0.380 (780/2053) | 39.00 | n/a | 99.10 | 1.00 | 0.346 [0.279, 0.410] | 0.346 | n/a | 9.85 of 65.7 | n/a |
| A1 conformal | 84.05 | 1.00 | 0.339 [0.293, 0.393] | 0.339 | 0.336 (565/1681) | 28.25 | 0.583 | 76.80 | 1.00 | 0.325 [0.262, 0.399] | 0.325 | 0.545 | 4.80 of 65.7 | 0.343 |
| A2 layer, no target | 40.05 | 1.00 | 0.123 [0.054, 0.262] | 0.123 | 0.120 (96/801) | 4.80 | 0.174 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 4.80 of 65.7 | 0.343 |
| A3 layer (m) | 48.20 | 1.00 | 0.122 [0.054, 0.242] | 0.122 | 0.119 (115/964) | 5.75 | 0.240 | 13.35 | 1.00 | 0.076 [0.000, 0.204] | 0.076 | 0.077 | 4.80 of 65.7 | 0.343 |
| A4 layer + FDR sel. (joint BH) | 76.15 | 1.00 | 0.273 [0.158, 0.405] | 0.273 | 0.273 (416/1523) | 20.80 | 0.240 | 35.05 | 1.00 | 0.163 [0.044, 0.325] | 0.163 | 0.077 | 15.00 of 65.7 | 0.343 |
| A4s layer + FDR sel. (BH per stratum) | 62.15 | 1.00 | 0.278 [0.159, 0.430] | 0.278 | 0.275 (342/1243) | 17.10 | 0.240 | 6.95 | 0.15 | 0.025 [0.000, 0.206] | 0.169 | 0.077 | 15.90 of 65.7 | 0.343 |
| A5 sel., source only | 115.30 | 1.00 | 0.434 [0.350, 0.533] | 0.434 | 0.431 (994/2306) | 49.70 | 0.583 | 110.50 | 1.00 | 0.388 [0.317, 0.445] | 0.388 | 0.545 | 15.90 of 65.7 | 0.343 |
| D3 = A3, oracle routing (diag.) | 17.85 | 1.00 | 0.062 [0.000, 0.184] | 0.062 | 0.062 (22/357) | 1.10 | 0.079 | 40.95 | 1.00 | 0.055 [0.010, 0.114] | 0.055 | 0.186 | 0.15 of 65.7 | 0.024 |
| D4 = A4, oracle routing (diag.) | 43.20 | 1.00 | 0.158 [0.058, 0.272] | 0.158 | 0.170 (147/864) | 7.35 | 0.079 | 72.20 | 1.00 | 0.193 [0.095, 0.330] | 0.193 | 0.186 | 1.25 of 65.7 | 0.024 |

### glass_E (m = 22; pool 742 = 709 source + 33 target candidates; tau = 65.90 [61.40, 69.03] GPa; source calibration n = 709)

Acceptable share: source candidates 0.902, target candidates 0.494. Shift check: AUROC 0.889 [0.851, 0.917]; target flagged 0.521 [0.408, 0.606]; source flagged 0.049 [0.032, 0.068]. q_source/IQR_T 0.39, q_target/IQR_T 2.08. RMSE source/target candidates 5.5/20.7.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 675.4 [622.0, 723.0] | 0.052 [0.025, 0.081] | 0.051 | 0.047 [0.025, 0.068] | 0.976 [0.953, 0.995] | 0.0 [0.0, 0.0] | 0.976 [0.953, 0.995] | 0.236 [0.162, 0.354] | n/a |
| A1 conformal | 533.5 [432.5, 640.3] | 0.040 [0.019, 0.060] | 0.038 | 0.028 [0.016, 0.038] | 0.779 [0.678, 0.890] | 194.9 [93.3, 295.6] | 0.997 [0.994, 1.000] | 0.135 [0.000, 0.250] | 0.134 |
| A2 layer, no target | 492.2 [389.3, 596.2] | 0.014 [0.004, 0.025] | 0.013 | 0.009 [0.003, 0.016] | 0.738 [0.638, 0.838] | 236.7 [137.0, 338.6] | 0.997 [0.994, 1.000] | 0.130 [0.000, 0.241] | 0.094 |
| A3 layer (m) | 498.8 [396.1, 605.4] | 0.016 [0.004, 0.031] | 0.016 | 0.010 [0.003, 0.018] | 0.746 [0.645, 0.849] | 252.2 [149.8, 353.8] | 0.997 [0.994, 1.000] | 0.130 [0.000, 0.241] | 0.100 |
| A4 layer + FDR sel. (joint BH) | 679.5 [631.5, 709.7] | 0.077 [0.037, 0.122] | 0.077 | 0.070 [0.035, 0.108] | 0.956 [0.940, 0.971] | 76.5 [53.4, 115.5] | 0.998 [0.994, 1.000] | 0.116 [0.000, 0.245] | 0.100 |
| A4s layer + FDR sel. (BH per stratum) | 671.2 [622.8, 702.0] | 0.076 [0.037, 0.125] | 0.076 | 0.069 [0.034, 0.110] | 0.945 [0.924, 0.964] | 85.5 [62.0, 124.1] | 0.999 [0.994, 1.000] | 0.108 [0.000, 0.245] | 0.100 |
| A5 sel., source only | 722.5 [682.9, 742.0] | 0.096 [0.047, 0.148] | 0.095 | 0.093 [0.047, 0.142] | 0.996 [0.987, 1.000] | 11.9 [0.0, 43.2] | 0.999 [0.994, 1.000] | 0.110 [0.000, 0.250] | 0.134 |
| D3 = A3, oracle routing (diag.) | 505.1 [408.9, 611.1] | 0.013 [0.005, 0.026] | 0.012 | 0.008 [0.003, 0.016] | 0.758 [0.666, 0.859] | 245.2 [144.0, 341.0] | 0.997 [0.994, 1.000] | 0.135 [0.000, 0.250] | 0.104 |
| D4 = A4, oracle routing (diag.) | 692.5 [654.4, 716.5] | 0.078 [0.038, 0.125] | 0.077 | 0.072 [0.036, 0.115] | 0.974 [0.968, 0.980] | 63.7 [47.5, 93.5] | 0.999 [0.994, 1.000] | 0.107 [0.000, 0.250] | 0.104 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 32.95 | 1.00 | 0.507 [0.257, 0.725] | 0.507 | 0.507 (334/659) | 16.70 | n/a | 48.85 | 1.00 | 0.335 [0.193, 0.456] | 0.335 | n/a | 3.10 of 15.8 | n/a |
| A1 conformal | 30.85 | 1.00 | 0.510 [0.257, 0.772] | 0.510 | 0.504 (311/617) | 15.55 | 0.795 | 41.25 | 1.00 | 0.348 [0.169, 0.466] | 0.348 | 0.571 | 2.40 of 15.8 | 0.602 |
| A2 layer, no target | 14.25 | 1.00 | 0.184 [0.000, 0.432] | 0.184 | 0.168 (48/285) | 2.40 | 0.288 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 2.40 of 15.8 | 0.602 |
| A3 layer (m) | 16.30 | 1.00 | 0.242 [0.054, 0.500] | 0.242 | 0.227 (74/326) | 3.70 | 0.385 | 6.55 | 0.95 | 0.156 [0.000, 0.609] | 0.164 | 0.075 | 2.40 of 15.8 | 0.602 |
| A4 layer + FDR sel. (joint BH) | 19.15 | 1.00 | 0.290 [0.064, 0.490] | 0.290 | 0.287 (110/383) | 5.50 | 0.385 | 9.90 | 1.00 | 0.248 [0.000, 0.559] | 0.248 | 0.075 | 3.10 of 15.8 | 0.602 |
| A4s layer + FDR sel. (BH per stratum) | 15.80 | 1.00 | 0.201 [0.000, 0.393] | 0.201 | 0.196 (62/316) | 3.10 | 0.385 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.075 | 3.10 of 15.8 | 0.602 |
| A5 sel., source only | 33.00 | 1.00 | 0.506 [0.257, 0.713] | 0.506 | 0.506 (334/660) | 16.70 | 0.795 | 51.10 | 1.00 | 0.345 [0.204, 0.454] | 0.345 | 0.571 | 3.10 of 15.8 | 0.602 |
| D3 = A3, oracle routing (diag.) | 2.55 | 0.90 | 0.446 [0.000, 1.000] | 0.495 | 0.529 (27/51) | 1.35 | 0.109 | 26.70 | 1.00 | 0.087 [0.000, 0.269] | 0.087 | 0.304 | 0.05 of 15.8 | 0.025 |
| D4 = A4, oracle routing (diag.) | 4.20 | 1.00 | 0.657 [0.059, 1.000] | 0.657 | 0.619 (52/84) | 2.60 | 0.109 | 37.30 | 1.00 | 0.173 [0.076, 0.293] | 0.173 | 0.304 | 0.10 of 15.8 | 0.025 |

### glass_HV (m = 23; pool 330 = 295 source + 35 target candidates; tau = 4.21 [4.17, 4.26] GPa; source calibration n = 294)

Acceptable share: source candidates 0.945, target candidates 0.520. Shift check: AUROC 0.939 [0.915, 0.960]; target flagged 0.539 [0.299, 0.686]; source flagged 0.042 [0.019, 0.066]. q_source/IQR_T 0.71, q_target/IQR_T 2.35. RMSE source/target candidates 0.7/2.0.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 318.2 [311.5, 324.1] | 0.083 [0.061, 0.102] | 0.083 | 0.080 [0.059, 0.099] | 0.983 [0.962, 0.997] | 0.0 [0.0, 0.0] | 0.983 [0.962, 0.997] | 0.410 [0.095, 0.622] | n/a |
| A1 conformal | 247.8 [207.8, 277.1] | 0.061 [0.037, 0.096] | 0.062 | 0.046 [0.026, 0.070] | 0.783 [0.657, 0.861] | 81.0 [52.0, 121.3] | 0.997 [0.990, 1.000] | 0.795 [0.125, 1.000] | 0.170 |
| A2 layer, no target | 224.9 [189.4, 250.6] | 0.025 [0.011, 0.048] | 0.024 | 0.017 [0.007, 0.030] | 0.739 [0.623, 0.831] | 103.8 [78.8, 140.1] | 0.997 [0.990, 1.000] | 0.795 [0.125, 1.000] | 0.106 |
| A3 layer (m) | 226.8 [191.4, 253.1] | 0.026 [0.013, 0.049] | 0.026 | 0.018 [0.009, 0.032] | 0.744 [0.630, 0.836] | 125.0 [99.3, 161.1] | 0.997 [0.990, 1.000] | 0.795 [0.125, 1.000] | 0.109 |
| A4 layer + FDR sel. (joint BH) | 304.6 [293.5, 317.0] | 0.064 [0.042, 0.090] | 0.064 | 0.059 [0.039, 0.085] | 0.960 [0.924, 0.987] | 48.5 [36.0, 59.5] | 1.000 [1.000, 1.000] | n/a | 0.109 |
| A4s layer + FDR sel. (BH per stratum) | 298.7 [289.5, 311.6] | 0.059 [0.040, 0.083] | 0.059 | 0.054 [0.036, 0.076] | 0.946 [0.909, 0.978] | 54.3 [41.4, 63.5] | 1.000 [1.000, 1.000] | n/a | 0.109 |
| A5 sel., source only | 330.0 [330.0, 330.0] | 0.100 [0.080, 0.124] | 0.100 | 0.100 [0.080, 0.124] | 1.000 [1.000, 1.000] | 0.0 [0.0, 0.0] | 1.000 [1.000, 1.000] | n/a | 0.170 |
| D3 = A3, oracle routing (diag.) | 222.3 [185.4, 248.5] | 0.016 [0.005, 0.031] | 0.016 | 0.011 [0.003, 0.023] | 0.736 [0.622, 0.817] | 129.5 [103.5, 167.1] | 0.997 [0.990, 1.000] | 0.795 [0.125, 1.000] | 0.092 |
| D4 = A4, oracle routing (diag.) | 300.1 [296.0, 310.2] | 0.061 [0.042, 0.092] | 0.061 | 0.056 [0.038, 0.086] | 0.949 [0.929, 0.970] | 53.0 [42.8, 57.0] | 1.000 [1.000, 1.000] | n/a | 0.092 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 33.45 | 1.00 | 0.501 [0.400, 0.629] | 0.501 | 0.502 (336/669) | 16.80 | n/a | 31.00 | 1.00 | 0.503 [0.303, 0.691] | 0.503 | n/a | 2.80 of 16.1 | n/a |
| A1 conformal | 26.30 | 1.00 | 0.447 [0.282, 0.644] | 0.447 | 0.456 (240/526) | 12.00 | 0.799 | 22.80 | 1.00 | 0.430 [0.223, 0.716] | 0.430 | 0.676 | 2.70 of 16.1 | 0.549 |
| A2 layer, no target | 12.15 | 1.00 | 0.209 [0.091, 0.425] | 0.209 | 0.222 (54/243) | 2.70 | 0.260 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 2.70 of 16.1 | 0.549 |
| A3 layer (m) | 12.50 | 1.00 | 0.226 [0.091, 0.441] | 0.226 | 0.240 (60/250) | 3.00 | 0.286 | 1.85 | 0.65 | 0.067 [0.000, 0.486] | 0.103 | 0.043 | 2.70 of 16.1 | 0.549 |
| A4 layer + FDR sel. (joint BH) | 18.30 | 1.00 | 0.221 [0.138, 0.387] | 0.221 | 0.232 (85/366) | 4.25 | 0.286 | 5.85 | 1.00 | 0.315 [0.000, 0.587] | 0.315 | 0.043 | 2.80 of 16.1 | 0.549 |
| A4s layer + FDR sel. (BH per stratum) | 16.15 | 1.00 | 0.160 [0.067, 0.319] | 0.160 | 0.173 (56/323) | 2.80 | 0.286 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.043 | 2.80 of 16.1 | 0.549 |
| A5 sel., source only | 35.00 | 1.00 | 0.480 [0.371, 0.629] | 0.480 | 0.480 (336/700) | 16.80 | 0.799 | 31.30 | 1.00 | 0.508 [0.311, 0.695] | 0.508 | 0.676 | 2.80 of 16.1 | 0.549 |
| D3 = A3, oracle routing (diag.) | 0.85 | 0.35 | 0.180 [0.000, 1.000] | 0.514 | 0.471 (8/17) | 0.40 | 0.063 | 9.00 | 1.00 | 0.081 [0.000, 0.317] | 0.081 | 0.104 | 0.10 of 16.1 | 0.083 |
| D4 = A4, oracle routing (diag.) | 5.05 | 1.00 | 0.530 [0.068, 1.000] | 0.530 | 0.426 (43/101) | 2.15 | 0.063 | 14.60 | 1.00 | 0.205 [0.098, 0.364] | 0.205 | 0.104 | 0.70 of 16.1 | 0.083 |

### glass_Tliq (m = 30; pool 1617 = 1552 source + 65 target candidates; tau = 1364.15 [1324.21, 1389.59] K; source calibration n = 1552)

Acceptable share: source candidates 0.523, target candidates 0.476. Shift check: AUROC 0.816 [0.789, 0.838]; target flagged 0.208 [0.153, 0.254]; source flagged 0.050 [0.036, 0.063]. q_source/IQR_T 0.34, q_target/IQR_T 1.60. RMSE source/target candidates 47.9/204.1.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 852.7 [750.9, 1020.5] | 0.087 [0.072, 0.112] | 0.086 | 0.045 [0.038, 0.058] | 0.923 [0.900, 0.956] | 0.0 [0.0, 0.0] | 0.923 [0.900, 0.956] | 0.084 [0.059, 0.112] | n/a |
| A1 conformal | 530.5 [397.0, 735.1] | 0.026 [0.012, 0.053] | 0.025 | 0.008 [0.004, 0.014] | 0.608 [0.513, 0.714] | 607.0 [538.4, 660.6] | 0.988 [0.979, 0.994] | 0.021 [0.012, 0.034] | 0.124 |
| A2 layer, no target | 509.8 [380.8, 711.2] | 0.021 [0.008, 0.048] | 0.020 | 0.006 [0.003, 0.012] | 0.587 [0.491, 0.695] | 668.6 [592.2, 716.0] | 0.992 [0.985, 0.997] | 0.014 [0.006, 0.024] | 0.100 |
| A3 layer (m) | 511.6 [382.7, 714.1] | 0.021 [0.008, 0.049] | 0.021 | 0.006 [0.003, 0.012] | 0.588 [0.492, 0.698] | 695.7 [621.2, 745.0] | 0.992 [0.985, 0.997] | 0.014 [0.006, 0.024] | 0.103 |
| A4 layer + FDR sel. (joint BH) | 862.0 [734.7, 1062.6] | 0.101 [0.085, 0.126] | 0.101 | 0.054 [0.040, 0.074] | 0.918 [0.891, 0.956] | 345.4 [281.4, 387.1] | 0.992 [0.985, 0.997] | 0.014 [0.006, 0.024] | 0.103 |
| A4s layer + FDR sel. (BH per stratum) | 862.1 [738.2, 1057.0] | 0.106 [0.089, 0.127] | 0.106 | 0.057 [0.042, 0.076] | 0.913 [0.884, 0.945] | 345.2 [285.6, 394.0] | 0.992 [0.985, 0.997] | 0.014 [0.006, 0.024] | 0.103 |
| A5 sel., source only | 896.6 [762.9, 1103.2] | 0.112 [0.096, 0.134] | 0.112 | 0.062 [0.046, 0.085] | 0.942 [0.915, 0.974] | 240.8 [180.9, 294.2] | 0.988 [0.979, 0.994] | 0.021 [0.012, 0.034] | 0.124 |
| D3 = A3, oracle routing (diag.) | 498.8 [370.4, 692.9] | 0.012 [0.005, 0.024] | 0.011 | 0.003 [0.002, 0.006] | 0.579 [0.487, 0.682] | 677.8 [616.5, 728.7] | 0.989 [0.979, 0.994] | 0.020 [0.012, 0.034] | 0.100 |
| D4 = A4, oracle routing (diag.) | 847.1 [717.3, 1047.7] | 0.092 [0.070, 0.115] | 0.093 | 0.049 [0.032, 0.071] | 0.910 [0.883, 0.939] | 329.5 [271.0, 378.6] | 0.989 [0.979, 0.994] | 0.020 [0.012, 0.034] | 0.100 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 47.15 | 1.00 | 0.386 [0.267, 0.500] | 0.386 | 0.387 (365/943) | 18.25 | n/a | 34.35 | 1.00 | 0.261 [0.170, 0.353] | 0.261 | n/a | 13.70 of 51.5 | n/a |
| A1 conformal | 32.85 | 1.00 | 0.232 [0.091, 0.384] | 0.232 | 0.240 (158/657) | 7.90 | 0.687 | 20.65 | 1.00 | 0.137 [0.049, 0.245] | 0.137 | 0.422 | 6.40 of 51.5 | 0.636 |
| A2 layer, no target | 29.05 | 1.00 | 0.211 [0.077, 0.387] | 0.211 | 0.220 (128/581) | 6.40 | 0.503 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 6.40 of 51.5 | 0.636 |
| A3 layer (m) | 29.80 | 1.00 | 0.211 [0.075, 0.387] | 0.211 | 0.220 (131/596) | 6.55 | 0.560 | 1.80 | 0.65 | 0.056 [0.000, 0.333] | 0.086 | 0.053 | 6.40 of 51.5 | 0.636 |
| A4 layer + FDR sel. (joint BH) | 43.65 | 1.00 | 0.342 [0.239, 0.499] | 0.342 | 0.341 (298/873) | 14.90 | 0.560 | 8.55 | 1.00 | 0.087 [0.000, 0.269] | 0.087 | 0.053 | 14.15 of 51.5 | 0.636 |
| A4s layer + FDR sel. (BH per stratum) | 40.95 | 1.00 | 0.345 [0.230, 0.520] | 0.345 | 0.346 (283/819) | 14.15 | 0.560 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.053 | 14.15 of 51.5 | 0.636 |
| A5 sel., source only | 48.35 | 1.00 | 0.399 [0.308, 0.511] | 0.399 | 0.399 (386/967) | 19.30 | 0.687 | 37.50 | 1.00 | 0.289 [0.163, 0.394] | 0.289 | 0.422 | 14.15 of 51.5 | 0.636 |
| D3 = A3, oracle routing (diag.) | 1.20 | 0.35 | 0.121 [0.000, 0.762] | 0.345 | 0.292 (7/24) | 0.35 | 0.110 | 17.60 | 1.00 | 0.082 [0.000, 0.169] | 0.082 | 0.329 | 0.20 of 51.5 | 0.066 |
| D4 = A4, oracle routing (diag.) | 6.40 | 0.90 | 0.220 [0.000, 0.537] | 0.245 | 0.297 (38/128) | 1.90 | 0.110 | 32.50 | 1.00 | 0.196 [0.091, 0.296] | 0.196 | 0.329 | 1.15 of 51.5 | 0.066 |

### steel_yield (m = 13; pool 60 = 39 source + 21 target candidates; tau = 1647.09 [1472.75, 1750.66] MPa; source calibration n = 39)

Acceptable share: source candidates 0.118, target candidates 0.486. Shift check: AUROC 0.984 [0.969, 1.000]; target flagged 0.745 [0.424, 0.977]; source flagged 0.014 [0.000, 0.051]. q_source/IQR_T 0.27, q_target/IQR_T 1.49. RMSE source/target candidates 111.4/616.7.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 4.1 [1.0, 11.1] | 0.495 [0.000, 1.000] | 0.427 | 0.029 [0.000, 0.067] | 0.125 [0.000, 0.325] | 0.0 [0.0, 0.0] | 0.125 [0.000, 0.325] | 0.226 [0.076, 0.386] | n/a |
| A1 conformal | 0.4 [0.0, 2.5] | 0.000 [0.000, 0.000] | 0.000 | 0.000 [0.000, 0.000] | 0.018 [0.000, 0.100] | 15.9 [6.4, 36.0] | 0.390 [0.105, 0.653] | 0.208 [0.066, 0.455] | 0.374 |
| A2 layer, no target | 0.4 [0.0, 2.5] | 0.000 [0.000, 0.000] | 0.000 | 0.000 [0.000, 0.000] | 0.018 [0.000, 0.100] | 28.4 [17.4, 47.5] | 0.932 [0.689, 1.000] | 0.030 [0.000, 0.123] | 0.138 |
| A3 layer (m) | 0.4 [0.0, 2.5] | 0.000 [0.000, 0.000] | 0.000 | 0.000 [0.000, 0.000] | 0.018 [0.000, 0.100] | 41.4 [30.4, 60.5] | 0.932 [0.689, 1.000] | 0.030 [0.000, 0.123] | 0.160 |
| A4 layer + FDR sel. (joint BH) | 0.0 [0.0, 0.0] | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 41.8 [30.4, 62.6] | 0.932 [0.689, 1.000] | 0.030 [0.000, 0.123] | 0.160 |
| A4s layer + FDR sel. (BH per stratum) | 1.1 [0.0, 11.0] | 0.027 [0.000, 0.286] | 0.273 | 0.005 [0.000, 0.052] | 0.033 [0.000, 0.329] | 40.7 [29.9, 57.2] | 0.932 [0.689, 1.000] | 0.030 [0.000, 0.123] | 0.160 |
| A5 sel., source only | 0.8 [0.0, 8.9] | 0.024 [0.000, 0.247] | 0.471 | 0.007 [0.000, 0.070] | 0.028 [0.000, 0.295] | 15.6 [2.4, 38.0] | 0.390 [0.105, 0.653] | 0.208 [0.066, 0.455] | 0.374 |
| D3 = A3, oracle routing (diag.) | 0.4 [0.0, 2.5] | 0.000 [0.000, 0.000] | 0.000 | 0.000 [0.000, 0.000] | 0.018 [0.000, 0.100] | 44.1 [38.0, 60.0] | 0.974 [0.832, 1.000] | 0.013 [0.000, 0.073] | 0.095 |
| D4 = A4, oracle routing (diag.) | 0.0 [0.0, 0.0] | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 44.5 [38.0, 62.0] | 0.974 [0.832, 1.000] | 0.013 [0.000, 0.073] | 0.095 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 1.10 | 0.60 | 0.500 [0.000, 1.000] | 0.833 | 0.818 (18/22) | 0.90 | n/a | 0.40 | 0.25 | 0.067 [0.000, 0.683] | 0.267 | n/a | 0.79 of 5.6 | n/a |
| A1 conformal | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.860 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.874 | 0.00 of 5.6 | 0.795 |
| A2 layer, no target | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.198 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 0.00 of 5.6 | 0.795 |
| A3 layer (m) | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.260 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.087 | 0.00 of 5.6 | 0.795 |
| A4 layer + FDR sel. (joint BH) | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.260 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.087 | 0.00 of 5.6 | 0.795 |
| A4s layer + FDR sel. (BH per stratum) | 0.10 | 0.05 | 0.050 [0.000, 0.525] | 1.000 | 1.000 (2/2) | 0.10 | 0.260 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.087 | 0.11 of 5.6 | 0.795 |
| A5 sel., source only | 0.40 | 0.05 | 0.025 [0.000, 0.262] | 0.500 | 0.500 (4/8) | 0.20 | 0.860 | 0.30 | 0.05 | 0.017 [0.000, 0.175] | 0.333 | 0.874 | 0.11 of 5.6 | 0.795 |
| D3 = A3, oracle routing (diag.) | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.062 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.100 | 0.00 of 5.6 | 0.000 |
| D4 = A4, oracle routing (diag.) | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.062 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.100 | 0.00 of 5.6 | 0.000 |

### polymer_Tg (m = 30; pool 1346 = 866 source + 480 target candidates; tau = 511.73 [506.57, 518.77] K; source calibration n = 867)

Acceptable share: source candidates 0.075, target candidates 0.497. Shift check: AUROC 0.987 [0.981, 0.991]; target flagged 0.992 [0.960, 1.000]; source flagged 0.047 [0.036, 0.061]. q_source/IQR_T 0.65, q_target/IQR_T 0.96. RMSE source/target candidates 49.2/68.9.

All candidates:

| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |
|---|---|---|---|---|---|---|---|---|---|
| A0 point | 274.6 [221.8, 346.6] | 0.205 [0.121, 0.284] | 0.210 | 0.043 [0.023, 0.073] | 0.713 [0.625, 0.810] | 0.0 [0.0, 0.0] | 0.713 [0.625, 0.810] | 0.081 [0.058, 0.101] | n/a |
| A1 conformal | 30.9 [14.9, 62.9] | 0.056 [0.000, 0.163] | 0.055 | 0.001 [0.000, 0.004] | 0.096 [0.050, 0.189] | 515.9 [465.2, 562.4] | 0.956 [0.933, 0.985] | 0.016 [0.006, 0.025] | 0.142 |
| A2 layer, no target | 10.8 [7.5, 17.0] | 0.023 [0.000, 0.134] | 0.028 | 0.000 [0.000, 0.001] | 0.035 [0.023, 0.053] | 637.6 [606.2, 659.2] | 0.983 [0.974, 0.992] | 0.007 [0.003, 0.012] | 0.059 |
| A3 layer (m) | 16.1 [8.4, 41.7] | 0.046 [0.000, 0.163] | 0.053 | 0.001 [0.000, 0.002] | 0.050 [0.026, 0.129] | 609.6 [558.6, 657.2] | 0.976 [0.964, 0.991] | 0.010 [0.004, 0.014] | 0.097 |
| A4 layer + FDR sel. (joint BH) | 17.0 [0.0, 50.6] | 0.062 [0.000, 0.214] | 0.126 | 0.002 [0.000, 0.008] | 0.048 [0.000, 0.129] | 608.8 [565.9, 651.9] | 0.976 [0.964, 0.991] | 0.010 [0.004, 0.014] | 0.097 |
| A4s layer + FDR sel. (BH per stratum) | 102.0 [5.7, 340.1] | 0.135 [0.000, 0.330] | 0.231 | 0.018 [0.000, 0.083] | 0.261 [0.020, 0.789] | 523.8 [264.3, 629.5] | 0.976 [0.964, 0.991] | 0.010 [0.004, 0.014] | 0.097 |
| A5 sel., source only | 340.8 [222.3, 461.9] | 0.268 [0.128, 0.351] | 0.279 | 0.071 [0.021, 0.116] | 0.808 [0.623, 0.928] | 206.1 [129.8, 301.1] | 0.956 [0.933, 0.985] | 0.016 [0.006, 0.025] | 0.142 |
| D3 = A3, oracle routing (diag.) | 16.2 [8.4, 41.7] | 0.046 [0.000, 0.163] | 0.052 | 0.001 [0.000, 0.002] | 0.051 [0.026, 0.129] | 604.5 [556.7, 646.2] | 0.975 [0.963, 0.991] | 0.010 [0.004, 0.014] | 0.099 |
| D4 = A4, oracle routing (diag.) | 18.2 [0.0, 51.1] | 0.070 [0.000, 0.213] | 0.129 | 0.002 [0.000, 0.008] | 0.052 [0.000, 0.130] | 602.5 [555.8, 642.3] | 0.975 [0.963, 0.991] | 0.010 [0.004, 0.014] | 0.099 |

Design-region subgroups (true target candidates; flagged candidates; target candidates the shift check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:

| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 point | 219.10 | 1.00 | 0.200 [0.122, 0.291] | 0.200 | 0.206 (904/4382) | 45.20 | n/a | 221.45 | 1.00 | 0.203 [0.121, 0.295] | 0.203 | n/a | 0.06 of 4.6 | n/a |
| A1 conformal | 20.05 | 1.00 | 0.079 [0.000, 0.250] | 0.079 | 0.070 (28/401) | 1.40 | 0.222 | 20.10 | 1.00 | 0.079 [0.000, 0.250] | 0.079 | 0.216 | 0.00 of 4.6 | 0.202 |
| A2 layer, no target | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.001 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.000 | 0.00 of 4.6 | 0.202 |
| A3 layer (m) | 5.30 | 0.60 | 0.069 [0.000, 0.333] | 0.115 | 0.104 (11/106) | 0.55 | 0.102 | 5.30 | 0.60 | 0.069 [0.000, 0.333] | 0.115 | 0.097 | 0.00 of 4.6 | 0.202 |
| A4 layer + FDR sel. (joint BH) | 0.05 | 0.05 | 0.000 [0.000, 0.000] | 0.000 | 0.000 (0/1) | 0.00 | 0.102 | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | 0.097 | 0.00 of 4.6 | 0.202 |
| A4s layer + FDR sel. (BH per stratum) | 71.25 | 0.40 | 0.074 [0.000, 0.342] | 0.185 | 0.264 (376/1425) | 18.80 | 0.102 | 72.35 | 0.30 | 0.076 [0.000, 0.353] | 0.252 | 0.097 | 0.00 of 4.6 | 0.202 |
| A5 sel., source only | 271.05 | 1.00 | 0.263 [0.111, 0.349] | 0.263 | 0.275 (1490/5421) | 74.50 | 0.222 | 274.50 | 1.00 | 0.265 [0.112, 0.353] | 0.265 | 0.216 | 0.06 of 4.6 | 0.202 |
| D3 = A3, oracle routing (diag.) | 5.30 | 0.60 | 0.069 [0.000, 0.333] | 0.115 | 0.104 (11/106) | 0.55 | 0.101 | 5.35 | 0.60 | 0.068 [0.000, 0.333] | 0.114 | 0.104 | 0.00 of 4.6 | 0.090 |
| D4 = A4, oracle routing (diag.) | 0.00 | 0.00 | 0.000 [0.000, 0.000] | n/a | n/a (0/0) | 0.00 | 0.101 | 0.55 | 0.35 | 0.150 [0.000, 1.000] | 0.429 | 0.104 | 0.00 of 4.6 | 0.090 |

## Sensitivity: specification threshold (RF and HistGB)

FDR (all candidates) / FDR (true target candidates) / measurements, mean over splits. `rpool_median` = median of y over the split's recalibration pool (HEADLINE; protocol-compliant: no target-test label; the pool contains the m recalibration records); `rpool_q75` = 75th percentile of the recalibration pool (protocol-compliant sensitivity); `rpool_holdout_median` = median of the recalibration-pool records not used for recalibration (independent of both target-stratum calibration labels and candidates); `tgt_median` = median of the whole target pool (task's literal definition; uses target-test labels = protocol deviation; sensitivity only); `tgt_q75` = 75th percentile of the whole target pool (task's literal sensitivity; uses target-test labels).

| dataset | model | tau | A0 | A1 | A2 | A3 | A4 | A4s | A5 | D3 | D4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glass_Tg | RF | rpool_median | 0.09/0.42/0 | 0.04/0.37/440 | 0.01/0.15/528 | 0.01/0.14/550 | 0.10/0.28/182 | 0.10/0.30/196 | 0.13/0.46/51 | 0.01/0.06/554 | 0.09/0.15/180 |
| glass_Tg | RF | rpool_q75 | 0.11/0.54/0 | 0.06/0.49/602 | 0.02/0.15/672 | 0.02/0.16/700 | 0.10/0.29/440 | 0.11/0.26/439 | 0.15/0.56/310 | 0.02/0.11/707 | 0.09/0.32/450 |
| glass_Tg | RF | rpool_holdout_median | 0.09/0.42/0 | 0.04/0.37/458 | 0.01/0.15/546 | 0.01/0.15/568 | 0.10/0.27/191 | 0.10/0.29/207 | 0.13/0.45/62 | 0.01/0.07/572 | 0.09/0.15/188 |
| glass_Tg | RF | tgt_median | 0.09/0.42/0 | 0.04/0.37/451 | 0.02/0.16/538 | 0.02/0.15/562 | 0.10/0.29/182 | 0.10/0.32/198 | 0.13/0.46/55 | 0.01/0.03/566 | 0.09/0.13/180 |
| glass_Tg | RF | tgt_q75 | 0.11/0.55/0 | 0.06/0.51/588 | 0.02/0.19/658 | 0.02/0.20/686 | 0.11/0.29/440 | 0.11/0.25/441 | 0.15/0.57/312 | 0.02/0.33/692 | 0.10/0.30/448 |
| glass_Tg | HistGB | rpool_median | 0.08/0.38/0 | 0.04/0.34/422 | 0.01/0.12/509 | 0.01/0.12/525 | 0.10/0.27/166 | 0.10/0.28/187 | 0.12/0.43/46 | 0.01/0.06/526 | 0.09/0.16/161 |
| glass_Tg | HistGB | rpool_q75 | 0.11/0.50/0 | 0.05/0.40/557 | 0.02/0.13/636 | 0.02/0.14/661 | 0.10/0.25/422 | 0.10/0.22/421 | 0.13/0.52/292 | 0.01/0.10/668 | 0.09/0.26/428 |
| glass_Tg | HistGB | rpool_holdout_median | 0.08/0.39/0 | 0.04/0.34/437 | 0.01/0.13/523 | 0.01/0.13/539 | 0.10/0.27/175 | 0.10/0.28/197 | 0.12/0.43/57 | 0.01/0.07/541 | 0.09/0.17/172 |
| glass_Tg | HistGB | tgt_median | 0.08/0.39/0 | 0.04/0.34/432 | 0.01/0.13/518 | 0.01/0.13/535 | 0.10/0.28/162 | 0.10/0.28/179 | 0.12/0.43/40 | 0.01/0.04/537 | 0.09/0.14/158 |
| glass_Tg | HistGB | tgt_q75 | 0.10/0.49/0 | 0.05/0.41/542 | 0.02/0.15/619 | 0.02/0.16/645 | 0.11/0.25/412 | 0.11/0.23/411 | 0.14/0.52/282 | 0.01/0.17/651 | 0.10/0.26/417 |
| glass_E | RF | rpool_median | 0.06/0.51/0 | 0.04/0.51/191 | 0.02/0.18/232 | 0.02/0.19/249 | 0.08/0.29/76 | 0.08/0.20/87 | 0.10/0.51/14 | 0.01/0.15/244 | 0.08/0.50/65 |
| glass_E | RF | rpool_q75 | 0.10/0.74/0 | 0.05/0.79/311 | 0.01/0.36/342 | 0.02/0.38/362 | 0.10/0.51/200 | 0.10/0.48/206 | 0.13/0.73/139 | 0.01/0.09/351 | 0.09/0.45/194 |
| glass_E | RF | rpool_holdout_median | 0.06/0.53/0 | 0.05/0.53/211 | 0.02/0.18/251 | 0.02/0.19/267 | 0.08/0.32/102 | 0.08/0.24/113 | 0.10/0.53/41 | 0.01/0.07/262 | 0.08/0.40/93 |
| glass_E | RF | tgt_median | 0.05/0.49/0 | 0.04/0.50/197 | 0.01/0.15/238 | 0.01/0.16/254 | 0.09/0.27/65 | 0.09/0.16/77 | 0.11/0.49/3 | 0.01/0.17/249 | 0.09/0.51/54 |
| glass_E | RF | tgt_q75 | 0.11/0.74/0 | 0.06/0.80/309 | 0.02/0.40/340 | 0.02/0.42/360 | 0.11/0.54/213 | 0.11/0.50/219 | 0.14/0.74/154 | 0.01/0.09/350 | 0.10/0.44/208 |
| glass_E | HistGB | rpool_median | 0.05/0.51/0 | 0.04/0.51/195 | 0.01/0.18/237 | 0.02/0.24/252 | 0.08/0.29/77 | 0.08/0.20/86 | 0.10/0.51/12 | 0.01/0.45/245 | 0.08/0.66/64 |
| glass_E | HistGB | rpool_q75 | 0.10/0.75/0 | 0.04/0.77/307 | 0.01/0.39/338 | 0.01/0.44/357 | 0.10/0.57/201 | 0.10/0.52/203 | 0.13/0.75/135 | 0.01/0.16/344 | 0.09/0.54/189 |
| glass_E | HistGB | rpool_holdout_median | 0.06/0.54/0 | 0.04/0.55/212 | 0.01/0.20/253 | 0.01/0.24/268 | 0.08/0.32/105 | 0.08/0.24/112 | 0.10/0.53/40 | 0.01/0.18/260 | 0.08/0.53/92 |
| glass_E | HistGB | tgt_median | 0.05/0.49/0 | 0.04/0.50/199 | 0.01/0.15/241 | 0.02/0.20/257 | 0.09/0.27/66 | 0.09/0.17/74 | 0.11/0.49/2 | 0.01/0.45/250 | 0.09/0.70/53 |
| glass_E | HistGB | tgt_q75 | 0.12/0.75/0 | 0.04/0.75/308 | 0.02/0.43/337 | 0.02/0.48/356 | 0.11/0.55/212 | 0.11/0.51/213 | 0.15/0.74/147 | 0.01/0.14/343 | 0.10/0.56/201 |
| glass_HV | RF | rpool_median | 0.08/0.48/0 | 0.07/0.50/66 | 0.03/0.19/92 | 0.03/0.22/113 | 0.06/0.25/50 | 0.06/0.16/54 | 0.10/0.48/0 | 0.02/0.22/119 | 0.06/0.59/53 |
| glass_HV | RF | rpool_q75 | 0.18/0.76/0 | 0.12/0.84/198 | 0.06/0.52/211 | 0.06/0.53/233 | 0.12/0.49/103 | 0.12/0.45/99 | 0.18/0.76/48 | 0.03/0.15/236 | 0.10/0.73/111 |
| glass_HV | RF | rpool_holdout_median | 0.08/0.50/0 | 0.08/0.53/68 | 0.03/0.21/94 | 0.03/0.23/115 | 0.07/0.25/49 | 0.06/0.17/54 | 0.10/0.50/0 | 0.02/0.19/121 | 0.06/0.62/53 |
| glass_HV | RF | tgt_median | 0.08/0.48/0 | 0.07/0.50/68 | 0.03/0.18/93 | 0.03/0.21/115 | 0.07/0.24/50 | 0.06/0.15/54 | 0.10/0.48/0 | 0.02/0.24/121 | 0.06/0.60/54 |
| glass_HV | RF | tgt_q75 | 0.21/0.79/0 | 0.14/0.93/233 | 0.06/0.59/243 | 0.07/0.59/266 | 0.13/0.55/116 | 0.13/0.50/111 | 0.20/0.80/61 | 0.03/0.10/267 | 0.10/0.70/123 |
| glass_HV | HistGB | rpool_median | 0.08/0.50/0 | 0.06/0.45/81 | 0.02/0.21/104 | 0.03/0.23/125 | 0.06/0.22/48 | 0.06/0.16/54 | 0.10/0.48/0 | 0.02/0.18/130 | 0.06/0.53/53 |
| glass_HV | HistGB | rpool_q75 | 0.17/0.76/0 | 0.12/0.80/194 | 0.06/0.56/206 | 0.06/0.56/229 | 0.12/0.51/102 | 0.12/0.45/101 | 0.18/0.76/50 | 0.03/0.00/231 | 0.10/0.74/108 |
| glass_HV | HistGB | rpool_holdout_median | 0.08/0.52/0 | 0.06/0.47/83 | 0.03/0.23/106 | 0.03/0.23/127 | 0.07/0.23/48 | 0.06/0.17/54 | 0.10/0.50/0 | 0.02/0.17/131 | 0.06/0.53/52 |
| glass_HV | HistGB | tgt_median | 0.08/0.50/0 | 0.06/0.44/83 | 0.02/0.20/105 | 0.02/0.21/127 | 0.06/0.21/49 | 0.06/0.15/54 | 0.10/0.48/0 | 0.02/0.17/131 | 0.06/0.53/54 |
| glass_HV | HistGB | tgt_q75 | 0.19/0.79/0 | 0.12/0.88/226 | 0.07/0.75/235 | 0.07/0.75/258 | 0.12/0.56/120 | 0.13/0.52/117 | 0.18/0.79/73 | 0.04/0.00/259 | 0.10/0.68/128 |
| glass_Tliq | RF | rpool_median | 0.08/0.35/0 | 0.02/0.20/660 | 0.01/0.18/719 | 0.02/0.18/745 | 0.10/0.31/365 | 0.10/0.31/368 | 0.11/0.37/266 | 0.01/0.13/723 | 0.09/0.17/348 |
| glass_Tliq | RF | rpool_q75 | 0.10/0.45/0 | 0.03/0.24/704 | 0.02/0.23/769 | 0.02/0.23/792 | 0.10/0.46/420 | 0.11/0.47/424 | 0.11/0.46/325 | 0.01/0.02/768 | 0.09/0.17/398 |
| glass_Tliq | RF | rpool_holdout_median | 0.08/0.34/0 | 0.02/0.21/659 | 0.01/0.19/718 | 0.01/0.19/745 | 0.10/0.31/361 | 0.10/0.31/367 | 0.11/0.36/265 | 0.01/0.14/724 | 0.09/0.16/340 |
| glass_Tliq | RF | tgt_median | 0.08/0.33/0 | 0.02/0.21/655 | 0.01/0.18/715 | 0.01/0.19/742 | 0.09/0.28/372 | 0.10/0.28/375 | 0.10/0.35/274 | 0.01/0.14/718 | 0.09/0.16/353 |
| glass_Tliq | RF | tgt_q75 | 0.09/0.44/0 | 0.03/0.23/711 | 0.02/0.22/774 | 0.02/0.22/798 | 0.10/0.43/422 | 0.10/0.45/426 | 0.11/0.44/327 | 0.01/0.04/773 | 0.09/0.17/398 |
| glass_Tliq | HistGB | rpool_median | 0.09/0.39/0 | 0.03/0.23/607 | 0.02/0.21/669 | 0.02/0.21/696 | 0.10/0.34/345 | 0.11/0.35/345 | 0.11/0.40/241 | 0.01/0.12/678 | 0.09/0.22/330 |
| glass_Tliq | HistGB | rpool_q75 | 0.11/0.57/0 | 0.05/0.36/651 | 0.04/0.36/718 | 0.04/0.36/744 | 0.11/0.57/403 | 0.11/0.59/403 | 0.12/0.57/303 | 0.02/0.06/716 | 0.09/0.23/382 |
| glass_Tliq | HistGB | rpool_holdout_median | 0.09/0.39/0 | 0.03/0.25/602 | 0.02/0.23/663 | 0.02/0.24/690 | 0.10/0.36/339 | 0.11/0.36/343 | 0.11/0.40/241 | 0.01/0.08/674 | 0.09/0.19/322 |
| glass_Tliq | HistGB | tgt_median | 0.08/0.36/0 | 0.02/0.21/597 | 0.02/0.20/658 | 0.02/0.20/685 | 0.09/0.31/351 | 0.10/0.31/350 | 0.10/0.37/247 | 0.01/0.13/669 | 0.08/0.20/336 |
| glass_Tliq | HistGB | tgt_q75 | 0.11/0.55/0 | 0.04/0.33/657 | 0.04/0.34/723 | 0.04/0.34/749 | 0.11/0.54/398 | 0.11/0.57/400 | 0.11/0.56/299 | 0.02/0.06/723 | 0.08/0.23/382 |
| steel_yield | RF | rpool_median | 0.28/0.23/0 | 0.00/0.00/14 | 0.00/0.00/27 | 0.00/0.00/40 | 0.00/0.00/41 | 0.00/0.00/41 | 0.00/0.00/15 | 0.00/0.00/44 | 0.00/0.00/44 |
| steel_yield | RF | rpool_q75 | 0.10/0.00/0 | 0.00/0.00/1 | 0.00/0.00/17 | 0.00/0.00/29 | 0.00/0.00/29 | 0.00/0.00/29 | 0.00/0.00/1 | 0.00/0.00/34 | 0.00/0.00/34 |
| steel_yield | RF | rpool_holdout_median | 0.34/0.30/0 | 0.03/0.10/22 | 0.03/0.10/32 | 0.03/0.10/45 | 0.00/0.00/47 | 0.01/0.01/46 | 0.00/0.00/23 | 0.01/0.00/48 | 0.00/0.00/49 |
| steel_yield | RF | tgt_median | 0.26/0.05/0 | 0.00/0.00/11 | 0.00/0.00/25 | 0.00/0.00/38 | 0.00/0.00/38 | 0.00/0.00/38 | 0.00/0.00/11 | 0.00/0.00/41 | 0.00/0.00/41 |
| steel_yield | RF | tgt_q75 | 0.00/0.00/0 | 0.00/0.00/0 | 0.00/0.00/16 | 0.00/0.00/29 | 0.00/0.00/29 | 0.00/0.00/29 | 0.00/0.00/0 | 0.00/0.00/34 | 0.00/0.00/34 |
| steel_yield | HistGB | rpool_median | 0.50/0.50/0 | 0.00/0.00/16 | 0.00/0.00/28 | 0.00/0.00/41 | 0.00/0.00/42 | 0.03/0.05/41 | 0.02/0.03/16 | 0.00/0.00/44 | 0.00/0.00/44 |
| steel_yield | HistGB | rpool_q75 | 0.10/0.00/0 | 0.00/0.00/1 | 0.00/0.00/17 | 0.00/0.00/27 | 0.00/0.00/27 | 0.03/0.00/27 | 0.00/0.00/1 | 0.00/0.00/31 | 0.00/0.00/31 |
| steel_yield | HistGB | rpool_holdout_median | 0.37/0.37/0 | 0.11/0.26/21 | 0.11/0.30/32 | 0.11/0.30/44 | 0.00/0.00/46 | 0.05/0.11/43 | 0.06/0.07/22 | 0.03/0.00/47 | 0.00/0.00/49 |
| steel_yield | HistGB | tgt_median | 0.37/0.48/0 | 0.00/0.00/14 | 0.00/0.00/27 | 0.00/0.00/40 | 0.00/0.00/40 | 0.04/0.05/39 | 0.02/0.03/13 | 0.00/0.00/42 | 0.00/0.00/42 |
| steel_yield | HistGB | tgt_q75 | 0.00/0.00/0 | 0.00/0.00/0 | 0.00/0.00/16 | 0.00/0.00/29 | 0.00/0.00/29 | 0.00/0.00/29 | 0.00/0.00/0 | 0.00/0.00/34 | 0.00/0.00/34 |
| polymer_Tg | RF | rpool_median | 0.17/0.16/0 | 0.04/0.08/508 | 0.01/0.00/636 | 0.04/0.13/610 | 0.05/0.00/604 | 0.12/0.05/547 | 0.24/0.23/235 | 0.04/0.13/604 | 0.05/0.00/597 |
| polymer_Tg | RF | rpool_q75 | 0.19/0.16/0 | 0.11/0.10/298 | 0.01/0.00/582 | 0.01/0.00/455 | 0.03/0.00/454 | 0.08/0.00/449 | 0.12/0.10/265 | 0.01/0.00/451 | 0.03/0.00/450 |
| polymer_Tg | RF | rpool_holdout_median | 0.17/0.17/0 | 0.04/0.08/511 | 0.01/0.00/637 | 0.04/0.13/612 | 0.05/0.00/605 | 0.12/0.05/549 | 0.24/0.23/235 | 0.04/0.13/606 | 0.05/0.00/598 |
| polymer_Tg | RF | tgt_median | 0.18/0.17/0 | 0.03/0.06/510 | 0.01/0.00/636 | 0.03/0.13/612 | 0.05/0.00/606 | 0.11/0.05/550 | 0.23/0.22/245 | 0.03/0.13/606 | 0.05/0.00/599 |
| polymer_Tg | RF | tgt_q75 | 0.19/0.17/0 | 0.15/0.15/300 | 0.00/0.00/583 | 0.00/0.00/457 | 0.05/0.00/453 | 0.09/0.00/450 | 0.13/0.12/265 | 0.00/0.00/453 | 0.05/0.00/449 |
| polymer_Tg | HistGB | rpool_median | 0.20/0.20/0 | 0.06/0.08/516 | 0.02/0.00/638 | 0.05/0.07/610 | 0.06/0.00/609 | 0.14/0.07/524 | 0.27/0.26/206 | 0.05/0.07/604 | 0.07/0.00/602 |
| polymer_Tg | HistGB | rpool_q75 | 0.24/0.23/0 | 0.03/0.00/342 | 0.03/0.00/583 | 0.03/0.00/462 | 0.00/0.00/461 | 0.09/0.02/447 | 0.20/0.19/268 | 0.03/0.00/459 | 0.00/0.00/458 |
| polymer_Tg | HistGB | rpool_holdout_median | 0.21/0.20/0 | 0.05/0.08/517 | 0.02/0.00/639 | 0.05/0.07/611 | 0.06/0.00/610 | 0.12/0.06/538 | 0.27/0.26/206 | 0.05/0.07/606 | 0.07/0.00/604 |
| polymer_Tg | HistGB | tgt_median | 0.21/0.20/0 | 0.05/0.07/517 | 0.01/0.00/638 | 0.04/0.07/611 | 0.05/0.00/614 | 0.14/0.08/515 | 0.27/0.27/204 | 0.04/0.07/606 | 0.06/0.00/608 |
| polymer_Tg | HistGB | tgt_q75 | 0.23/0.22/0 | 0.03/0.00/343 | 0.03/0.00/583 | 0.03/0.00/464 | 0.01/0.00/463 | 0.10/0.02/449 | 0.20/0.21/273 | 0.03/0.00/461 | 0.01/0.00/459 |

## Self-test of trustlayer.py (synthetic, assumptions true by construction)

```
{'selection_pvalue_closed_vs_generic_maxabsdiff': 0.0, 'exchangeable_coverage_mean': 0.9016333333333334, 'novelty_flag_rate_in_domain_mean': 0.04904166666666667, 'novelty_flag_rate_shifted_mean': 1.0, 'gap_demo': {'setup': 'x~U(0,1)^2, y=10x0+N(0,0.2^2), minus 5 with prob 0.3 if x0>=0.9; oracle mu=10x0; n_cal=400, 400 candidates, spec y>=8.5, alpha=0.1, q=0.1, reps=300', 'interval_accept': {'P(accept and violate)': 0.029425000000000003, 'FDR_among_accepted': 0.26159709897550726, 'mean_n_accepted': 45.096666666666664}, 'conformal_selection': {'P(accept and violate)': 0.001641666666666667, 'FDR_among_accepted': 0.028869297644891382, 'mean_n_accepted': 2.22}}, 'homoscedastic_demo': {'setup': 'y=10x0+N(0,1), oracle mu, n_cal=300, 300 candidates, spec y>=5, q=0.1, reps=300', 'selection_FDR': 0.09416137132608551, 'selection_mean_n': 154.82, 'interval_accept_FDR': 0.006023316843646267}, 'mondrian_routing': {'rare(<n_min)': 'source:all', 'common': 'source:common', 'unseen': 'source:all', 'flagged_no_target': 'none', 'flagged_no_target_q': inf, 'n_min': 9, 'n_min_novelty': 20}}
```
