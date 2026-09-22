# Systematic literature survey for Figure 2

Response to Reviewer 2, comment 13 ("The literature survey needs to be more systematic; otherwise, Figure 2 may give the impression of 'selective sampling'").

Protocol: `revision/audit/A6_survey_protocol.md`, frozen before the run. Source: Europe PMC REST title/abstract search, 2020-2025, English, original research articles, open access. Frozen query version per area: A1 v1, A2 v1, A3 v2, A4 v2, A5 v1. Frames were permuted with `numpy.random.default_rng(20260919 + area_index)` and screened in that order until 40 eligible papers were reached (A5: the whole frame).

**Sample.** 871 open-access records in the five frames; 518 screened in the frozen random order; 169 eligible and coded, covering 153 unique papers (16 papers fell in two areas). Half of the coded records (85 area-records, 81 unique papers) were coded a second time, independently and blind, by a different coder, and every disagreement was adjudicated against the full text by a third reader.

## 1. Headline proportions (final, adjudicated codes)

Proportion of coded papers that report each practice, with 95% Wilson confidence intervals.

| Area | n | U>=1 (any UQ) | U>=2 (per-prediction) | U3 (calibrated) | E1 (external) | EP (prospective) | S1 (shift test) | U>=2 & E1 & S1 |
|---|---|---|---|---|---|---|---|---|
| A1 imaging-based characterisation | 40 | 15/40 = 37.5% (24.2-53.0) | 2/40 = 5.0% (1.4-16.5) | 0/40 = 0.0% (0.0-8.8) | 2/40 = 5.0% (1.4-16.5) | 6/40 = 15.0% (7.1-29.1) | 16/40 = 40.0% (26.3-55.4) | 0/40 = 0.0% (0.0-8.8) |
| A2 property prediction / inverse design | 40 | 9/40 = 22.5% (12.3-37.5) | 0/40 = 0.0% (0.0-8.8) | 0/40 = 0.0% (0.0-8.8) | 1/40 = 2.5% (0.4-12.9) | 10/40 = 25.0% (14.2-40.2) | 5/40 = 12.5% (5.5-26.1) | 0/40 = 0.0% (0.0-8.8) |
| A3 biosensors and bioelectronics | 40 | 4/40 = 10.0% (4.0-23.1) | 0/40 = 0.0% (0.0-8.8) | 0/40 = 0.0% (0.0-8.8) | 1/40 = 2.5% (0.4-12.9) | 1/40 = 2.5% (0.4-12.9) | 5/40 = 12.5% (5.5-26.1) | 0/40 = 0.0% (0.0-8.8) |
| A4 tissue engineering, scaffolds, hydrogels | 40 | 10/40 = 25.0% (14.2-40.2) | 2/40 = 5.0% (1.4-16.5) | 1/40 = 2.5% (0.4-12.9) | 2/40 = 5.0% (1.4-16.5) | 6/40 = 15.0% (7.1-29.1) | 9/40 = 22.5% (12.3-37.5) | 0/40 = 0.0% (0.0-8.8) |
| A5 autonomous and sequential discovery | 9 | 7/9 = 77.8% (45.3-93.7) | 7/9 = 77.8% (45.3-93.7) | 0/9 = 0.0% (0.0-29.9) | 1/9 = 11.1% (2.0-43.5) | 8/9 = 88.9% (56.5-98.0) | 1/9 = 11.1% (2.0-43.5) | 1/9 = 11.1% (2.0-43.5) |
| **All areas (unique papers)** | **153** | **42/153 = 27.5% (21.0-35.0)** | **11/153 = 7.2% (4.1-12.4)** | **1/153 = 0.7% (0.1-3.6)** | **7/153 = 4.6% (2.2-9.1)** | **29/153 = 19.0% (13.5-25.9)** | **30/153 = 19.6% (14.1-26.6)** | **1/153 = 0.7% (0.1-3.6)** |
| All areas (area-records) | 169 | 45/169 = 26.6% (20.5-33.8) | 11/169 = 6.5% (3.7-11.3) | 1/169 = 0.6% (0.1-3.3) | 7/169 = 4.1% (2.0-8.3) | 31/169 = 18.3% (13.2-24.9) | 36/169 = 21.3% (15.8-28.1) | 1/169 = 0.6% (0.1-3.3) |

**U3 sensitivity.** The single U3 in the survey (PMC10265106) was coded by one coder with low confidence and flagged borderline: the main text reports that all evaluated scores fell inside the GPR's 95% interval, but the plot is in Supporting Fig. S5 and the held-out status of those scores is not stated. It is kept as U3. If it is demoted to U2, U3 becomes 0/153 = 0.0% (0.0-2.4) overall.

**Codes.** U0 none; U1 spread of model *performance* (SD/SE/CI over folds, seeds or repeats); U2 per-prediction uncertainty reported or used operationally; U3 U2 plus an empirical calibration or coverage assessment. E1 evaluation on data from a source independent of the training data's generating process (another lab, instrument, cohort, public dataset, or simulation to experiment). EP prospective validation: new samples produced or measured after the model was fixed and compared with its predictions. S1 an explicit distribution-shift evaluation with performance reported.

### 1.1 S1 sub-codes

Coder A recorded the S1 sub-code for A1 and A2 only; for A3-A5 the field is absent, so the synthetic share cannot be reported for those areas.

| Area | S1 papers | sub-codes |
|---|---|---|
| A1 | 16 | extrapolation 1, natural 12, natural+synthetic 1, synthetic 2 |
| A2 | 5 | extrapolation 1, natural 3, synthetic 1 |
| A3 | 5 | not_recorded 5 |
| A4 | 9 | not_recorded 9 |
| A5 | 1 | not_recorded 1 |

### 1.2 Secondary analysis by year band

| Band | n | U>=1 | U>=2 | U3 | E1 | EP | S1 |
|---|---|---|---|---|---|---|---|
| 2020-2022 | 31 | 7/31 = 22.6% (11.4-39.8) | 1/31 = 3.2% (0.6-16.2) | 0/31 = 0.0% (0.0-11.0) | 0/31 = 0.0% (0.0-11.0) | 6/31 = 19.4% (9.2-36.3) | 5/31 = 16.1% (7.1-32.6) |
| 2023-2025 | 122 | 35/122 = 28.7% (21.4-37.3) | 10/122 = 8.2% (4.5-14.4) | 1/122 = 0.8% (0.1-4.5) | 7/122 = 5.7% (2.8-11.4) | 23/122 = 18.9% (12.9-26.7) | 25/122 = 20.5% (14.3-28.5) |

### 1.3 Mapping onto Figure 2

Protocol section 7 fixes the wording rule in advance: "reported" if >= 50%, "limited" if 10-49%, "rarely" if < 10%, with the CI shown. Applying it to the adjudicated codes:

| Area | Uncertainty (U>=2) | External validation (E1) | Shift testing (S1) |
|---|---|---|---|
| A1 imaging-based characterisation | rarely (5%) | rarely (5%) | limited (40%) |
| A2 property prediction / inverse design | rarely (0%) | rarely (2%) | limited (12%) |
| A3 biosensors and bioelectronics | rarely (0%) | rarely (2%) | limited (12%) |
| A4 tissue engineering, scaffolds, hydrogels | rarely (5%) | rarely (5%) | limited (22%) |
| A5 autonomous and sequential discovery | reported (78%) | limited (11%) | limited (11%) |

The aggregate claim in the submitted Figure 2 legend - that calibrated uncertainty and distribution-shift testing are "reported rarely or only in limited form" - survives the systematic sample. Three nuances that the qualitative wording hides should be carried into the revised figure and text. (i) Per-prediction uncertainty is not uniformly rare: it is the norm in A5 (7/9 = 77.8% (45.3-93.7)), because Gaussian-process surrogates are intrinsic to Bayesian optimisation and active learning, while it is absent in A2 and A3. (ii) Having per-prediction uncertainty and *assessing* its calibration are different things: U3 is 1/153 = 0.7% (0.1-3.6) across the whole sample, and the one case is borderline. (iii) Shift testing is not uniformly rare either - it reaches 16/40 = 40.0% (26.3-55.4) in imaging, where sim-to-real, acquisition-condition and grouped-split evaluations are common. External validation is the practice that is genuinely rare everywhere (7/153 = 4.6% (2.2-9.1)), and the three trust practices almost never co-occur: 1/153 = 0.7% (0.1-3.6) of papers report U>=2, E1 and S1 together.

## 2. PRISMA-style flow

| Stage | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| Identified, Europe PMC topic query, all types | 657 | 761 | 765 | 639 | 375 |
| Removed: tagged reviews | 185 | 203 | 172 | 281 | 27 |
| Removed: preprints (SRC:PPR) | 68 | 61 | 49 | 51 | 64 |
| Research-only records | 397 | 495 | 542 | 302 | 282 |
| Open-access sampling frame | 199 | 198 | 189 | 140 | 145 |
| Screened in random order | 116 | 62 | 79 | 116 | 145 |
| Full texts retrieved | 48 | 46 | 46 | 51 | 28 |
| Excluded | 76 | 22 | 39 | 76 | 136 |
| Eligible | 40 | 40 | 40 | 40 | 9 |
| Coded (coder A) | 40 | 40 | 40 | 40 | 9 |
| Dual coded (coder B) | 20 | 20 | 20 | 20 | 5 |

Exclusion reasons (`NOT_RESEARCH` review, perspective, resource without model evaluation; `NO_ML` no machine-learning model; `NOT_BIOMED` no biomedical material or device; `WRONG_AREA` eligible but not this area):

| Exclusion reason | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| NOT_BIOMED | 50 | 8 | 23 | 23 | 78 |
| NO_ML | 9 | 7 | 13 | 28 | 46 |
| WRONG_AREA | 12 | 4 | 3 | 15 | 9 |
| NOT_RESEARCH | 5 | 3 | 0 | 10 | 3 |

A5's frame was exhausted: all 145 open-access records were screened and only 9 were eligible, so A5 is a census of its frame rather than a 40-paper sample. A4 reached the quota after 116 of 140 records.

## 3. Inter-coder reliability (before adjudication)

Coder A coded all 169 eligible area-records. Coder B independently coded a blind 50% subset: 85 area-records covering 81 unique papers. Cohen's kappa; 95% CIs from 2000 bootstrap draws resampling papers (seed 20260919). Because several items are rare, raw agreement, PABAK and positive/negative specific agreement are reported as well (the kappa paradox).

| Item | n | % agreement (95% CI) | Cohen's kappa (95% CI) | PABAK | pos. spec. agr. | neg. spec. agr. | A+ / B+ |
|---|---|---|---|---|---|---|---|
| U, 4 levels (unweighted) | 85 | 98.8% (96.4-100.0) | 0.973 (0.91 to 1.00) | - | - | - | - / - |
| U, 4 levels (linearly weighted) | 85 | 98.8% (96.4-100.0) | 0.978 (0.92 to 1.00) | - | - | - | - / - |
| U >= 1 | 85 | 98.8% (96.4-100.0) | 0.971 (0.90 to 1.00) | 0.976 | 0.980 | 0.992 | 24 / 25 |
| U >= 2 | 85 | 100.0% (100.0-100.0) | 1.000 (1.00 to 1.00) | 1.000 | 1.000 | 1.000 | 6 / 6 |
| U >= 3 (U3) | 85 | 100.0% (100.0-100.0) | - (-) | 1.000 | - | 1.000 | 0 / 0 |
| E1 | 85 | 100.0% (100.0-100.0) | 1.000 (1.00 to 1.00) | 1.000 | 1.000 | 1.000 | 4 / 4 |
| EP | 85 | 95.3% (90.4-98.9) | 0.859 (0.69 to 0.97) | 0.906 | 0.889 | 0.970 | 18 / 18 |
| S1 | 85 | 98.8% (96.4-100.0) | 0.961 (0.86 to 1.00) | 0.976 | 0.968 | 0.993 | 16 / 15 |

Per area (percent agreement, Cohen's kappa):

| Item | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| U, 4 levels (unweighted) | 95% / 0.91 | 100% / 1.00 | 100% / 1.00 | 100% / 1.00 | 100% / 1.00 |
| U, 4 levels (linearly weighted) | 95% / 0.92 | 100% / 1.00 | 100% / 1.00 | 100% / 1.00 | 100% / 1.00 |
| U >= 1 | 95% / 0.90 | 100% / 1.00 | 100% / 1.00 | 100% / 1.00 | 100% / 1.00 |
| U >= 2 | 100% / 1.00 | 100% / - | 100% / - | 100% / 1.00 | 100% / 1.00 |
| U >= 3 (U3) | 100% / - | 100% / - | 100% / - | 100% / - | 100% / - |
| E1 | 100% / 1.00 | 100% / 1.00 | 100% / 1.00 | 100% / 1.00 | 100% / - |
| EP | 95% / 0.86 | 95% / 0.86 | 100% / 1.00 | 90% / 0.69 | 100% / 1.00 |
| S1 | 95% / 0.90 | 100% / 1.00 | 100% / 1.00 | 100% / 1.00 | 100% / - |

`-` for kappa means chance agreement is 1 and kappa is undefined: both coders used a single category (this is the case for U3 everywhere, and for several items within single areas). Percent agreement is the informative statistic there.

Per-area bootstrap confidence intervals for both statistics are in `survey_results.json` under `reliability.by_area`; they are omitted from the table above only for readability.

**Disagreements.** 6 item-level disagreements over 85 dual-coded area-records x 4 items = 340 decisions, i.e. 1.8% of decisions. Every one is adjudicated in section 4. The protocol's threshold (section 9: revise the item definition and recode if kappa < 0.6) is met by every item for which kappa is defined; EP is the weakest at kappa = 0.86 (95% CI 0.69 to 0.97), and four of the six disagreements, plus all three cross-area inconsistencies, are EP or E calls. EP is the hardest item in this codebook: it turns on whether the samples post-date the fixing of the model, which papers often do not state. The protocol's human-versus-LLM blind check (section 9) has not been run, so these figures bound LLM-to-LLM agreement only, not accuracy.

## 4. Adjudication

The adjudicator re-read the cached Europe PMC JATS full text of every disputed paper. Final codes = coder A, except where adjudicated below; the adjudicated code applies to the paper in every area in which it was sampled.

**PMC10161767** (A1) - item **U**: coder A 0, coder B 1 -> **U1**  
*Fast extraction of three-dimensional nanofiber orientation from WAXD patterns using machine learning.* (IUCrJ, 2023)  
Coder B is right. The Results text of the paper states 'Table 1 summarizes the performance metrics of various algorithms under two phases together with the corresponding standard deviations in parentheses', and the Methods state the models are 'trained five times' over random weight initialisations and dataset splits. That is the SD of a performance metric over repeated runs/splits, which section 7 lists under U1. Coder A applied the 'trained 5 times and averaged with no spread' exclusion but missed the Table 1 spread, which sits in <floats-group> and was not reached by the regex screen.

**PMC12796348** (A1) - item **EP**: coder A 0, coder B 1 -> **EP0**  
*MIDAS: rapid, multiplexed molecular profiling for integrated host-pathogen analysis.* (Nature communications, 2025)  
Coder A is right. The paper itself calls the swine work a 'retrospective validation study'; the specimens were collected and frozen before the model existed, and the Discussion lists 'larger-scale prospective swine studies' as future work. The reported concordance (30/32 samples; Pearson r = 0.87) is between the MIDAS assay readout and reference assays, not between a model prediction and a newly produced measurement. Section 7 requires samples produced or measured AFTER the model was fixed.

**PMC11538249** (A1) - item **S**: coder A 1, coder B 0 -> **S1 (natural)**  
*Deep learning based highly accurate transplanted bioengineered corneal equivalent thickness measurement using optical coherence tomography.* (NPJ digital medicine, 2024)  
Coder A is right on the frozen codebook. Section 7 lists 'group-, cluster- or scaffold-based split' under the S1 natural sub-code without any additional requirement that the paper frame the grouping factor as the shift under study. The Methods state 'the data obtained from the same object was used exclusively for either training or testing, not both ... the objects used for training were different from those used for validation and testing', i.e. an animal-level grouped split, and segmentation performance on the held-out animals is reported (Tables 1-2). Coder B applied a stricter rule that the frozen protocol does not contain.

**PMC9513834** (A2) - item **EP**: coder A 1, coder B 0 -> **EP1**  
*High-Content Screening and Analysis of Stem Cell-Derived Neural Interfaces Using a Combinatorial Nanotechnology and Machine Learning Approach.* (Research (Washington, D.C.), 2022)  
Coder A is right. Section 7 gives 'synthesising the predicted optimum' as the leading EP example. The Results state 'Using this design as a guide, we synthesized aligned nanofibers using a standard rotating drum electrospinning method, producing nanofibers with four distinct diameters (including the diameters that were projected to be ideal by our radar charts)', and the measured axonal alignment at ~200 nm 'fits the estimation in the radar charts'. New material was made on a different platform after the GPR-derived maps were fixed and compared with them. Coder B's stated reason addresses U and E (hold-out CV, one lab), not the EP definition. The comparison is qualitative, which section 7 does not forbid; confidence is medium.

**PMC11170757** (A1, A4) - item **EP**: coder A 1, coder B 0 -> **EP0**  
*Quantitative Evaluation of the Pore and Window Sizes of Tissue Engineering Scaffolds on Scanning Electron Microscope Images Using Deep Learning.* (ACS omega, 2024)  
Coder B is right. The Introduction gives the chronology explicitly: (i) seven PolyHIPE groups were fabricated, (ii) four users blind-quantified their pores and windows, (iii) only afterwards was the Pore D2 YOLOv5 detector developed. The test-group scaffolds and their manual ground truth therefore pre-date the model, so section 7's requirement of samples 'produced or measured AFTER the model was fixed' is not met. The comparison is automated versus manual measurement on the same SEM images, and the Conclusions state 'it is not applicable to assume one way as a gold standard'. The single-verifier pilot (protocol section 10.2) also recorded EP=0 for this paper.

**PMC7076403** (A2, A4) - item **EP**: coder A 1 in A2, 0 in A4, coder B 1 -> **EP1**  
*A 3D Bioprinted Pseudo-Bone Drug Delivery Scaffold for Bone Tissue Engineering.* (Pharmaceutics, 2020)  
Coder B and coder A's A2 coding are right; coder A's A4 coding was inconsistent with its own A2 coding of the same paper. The Results state 'The optimized 3D bioprinted scaffold formulation, resulting in the highest factor of response ... was thus synthesized incorporating 14% w/v PPF and 16% w/v PF127. The optimized 3D bioprinted scaffold displayed a controlled release of simvastatin over a 20-day duration ... with significant correlation to the predicted release kinetics ascertained using ANN', and Figure 10's legend is 'Correlation of in vitro simvastatin release analysis of the optimized 3D bioprinted scaffold with predicted release kinetics using ANN modeling'. That is the predicted optimum synthesised and its measured behaviour compared with the model prediction. Agreement is reported qualitatively, so confidence is medium.

**PMC11921027** (A1, A4) - item **E**: coder A 1 in A1, 0 in A4, coder B 0 -> **E0**  
*Collagen Hybridizing Peptides Promote Collagen Fibril Growth In Vitro.* (ACS applied bio materials, 2025)  
Coder A's A4 coding and coder B are right. What the paper calls its 'external data sets' are AFM images the authors acquired themselves on their own JPK Nanowizard4; no other laboratory, instrument, cohort, site, or public/literature dataset is involved. Section 7's E1 exclusions route 'a separately collected batch from the same lab and set-up' away from E1. The different collagen source (rat-tail tendon, reformed fibrils) is a distribution shift and is already captured as S1 natural, which both coders gave.

**PMC8575943** (A1, A4) - item **EP**: coder A 1 in A1, 0 in A4, coder B 1 in A1, 0 in A4 -> **EP0**  
*A deep learning approach to identify and segment alpha-smooth muscle actin stress fiber positive cells.* (Scientific reports, 2021)  
Both coders coded this paper EP=1 in its A1 record and EP=0 in its A4 record, so the paper needed a single adjudicated code. EP=0. The substrate-stiffening data set is part of the study's own Fig. 2 experiments; the Results say it 'was collected at a later time, under different microscope settings' relative to the training images, but it was collected before the model was applied and the model is run on it retrospectively and compared with a manual analysis. Moreover the model was not fixed for that set: 'the experimental images were normalized by a different set of values than the training set ... optimizing these normalization parameters until the model best matched the manual analysis'. Section 7 requires samples produced or measured after the model was fixed. The changed microscope settings are a shift and stay coded S1 natural in both areas.

Eight items over seven papers were adjudicated: six coder-A-versus-coder-B disagreements, plus two items (PMC11921027 E, PMC8575943 EP) where the coders agreed within each area but the same paper had been given different codes in its two area records. PMC7076403 was both. Sixteen papers were sampled in two areas, and three of them carried such a cross-area inconsistency (for PMC8575943 both coders made the same one, EP=1 in its A1 record and EP=0 in its A4 record). The protocol requires a paper to be coded once, so these were adjudicated too. Their existence is itself a finding: the area context in which a paper is read can move a borderline call, and all three sit on the EP/E boundary.

**PMC10265106** (A4) - item **U**, single-coder borderline, code upheld at U3.  
Only U3 in the whole survey; coder A flagged it borderline with low confidence and it is outside coder B's 50% subset, so it is not an adjudicable disagreement. On re-reading, the main text does report a coverage-style assessment of the GPR's own predictive intervals: 'the rationality of the GPR model in this research was confirmed by comparing the confidence interval and the evaluated scores. As demonstrated in Figure S5 (Supporting Information), all evaluated scores were within the 95% confidence interval'. That is empirical coverage against a nominal level, which section 7 lists under U3, and is different from the pilot's U3 false positive (an untested 'well-calibrated' claim). Two weaknesses are recorded and the code is left at U3: the supporting plot is in the SI (SI_dependency = true) and the main text does not state whether the compared scores are the held-out 2.5 w/v% test set or all 40 points. A stricter adjudicator would return U2, which would make U3 = 0/153 overall; both figures are reported.

## 5. Limitations

- **coverage.** Europe PMC coverage. With the same boolean logic, only 23-28% of the article DOIs that OpenAlex returns are present in the Europe PMC research-only set (A1 26%, 363/1380; A2 26%, 455/1755; A3 23%, 1105/4790; A4 28%, 771/2778); in the other direction 92-96% of Europe PMC DOIs are in OpenAlex. Materials and engineering journals that Europe PMC does not index (Chemical Engineering Journal, Nano Energy, Advanced Functional Materials, IEEE Sensors Journal, Sensors and Actuators A/B, Int. J. Bioprinting and others) are therefore absent, and part of the gap is OpenAlex noise that the pilot could not separate. The proportions reported here therefore describe the Europe PMC open-access literature, not the whole field.
- **oa_only.** Open-access only. The sampling frame is the open-access subset, because full text is needed for coding. It over-represents open-access biomedical and multidisciplinary venues (Scientific Reports, Advanced Science, MDPI and Frontiers titles). The non-PMC sensitivity sample of protocol section 6.4 was not drawn.
- **keyword_queries.** Keyword queries. Identification is a title/abstract boolean search, frozen per area after one refinement round (A1 v1, A2 v1, A3 v2, A4 v2, A5 v1). Papers that describe their ML or their material in other words are missed, and title/abstract precision was low in some areas, so the screen-until-quota sample is a random sample of what the query retrieves, not of the field. Areas are non-exclusive frames: 16 papers were sampled in two areas.
- **llm_coding.** LLM-assisted coding. Both coders are large language models working from a regex candidate screen plus an evaluation digest over the parsed JATS full text, with a verbatim quote required for every positive code. One coder covered all eligible papers, a second, independent coder a blind 50% subset, and a third reader adjudicated every disagreement against the full text. The human-versus-LLM blind check of protocol section 9 (20 papers coded by a domain-expert author) has not been done, so errors shared by both coders would not be detected here.
- **main_text_only.** Main article only. The construct is 'reported in the main article'; supplementary files were not screened. Evaluation detail that lives only in the SI is therefore coded absent, which biases the proportions downwards to an unknown degree.
- **a5_census.** A5 is a census, not a 40-paper sample. The whole A5 open-access frame (145 records) was screened and yielded only 9 eligible papers, so the A5 confidence intervals are wide and the pre-specified frame extension of protocol section 6.3 was not run.
- **pooling.** Pooling. Per-area proportions are unweighted. The pooled figure is computed over the 153 unique papers (each paper once); the area-record version (169) is given for comparison. Neither is weighted by frame size, so the pooled value is descriptive.
- **power.** Precision. With about 40 papers per area, a 0/40 observation has a 95% Wilson upper bound of 8.8%, which supports or refutes 'rarely reported' (<10%) per area but is not enough for fine between-area contrasts unless the differences are large.

## 6. Files

- `survey_results.json` - every number in this note, machine readable.
- `survey_coded_papers.csv` - one row per coded area-record (169 rows, 153 unique papers) with area, PMCID, DOI, year, journal, title, final U/E/EP/S codes and the adjudication flag. This is the Supporting Information table that replaces the hand-picked Table 6.
- `sample_A*.json` - the screening decision and reason for every record examined.
- `codes_A_A*.json`, `codes_B_A*.json` - both coders' codes with verbatim evidence quotes.
