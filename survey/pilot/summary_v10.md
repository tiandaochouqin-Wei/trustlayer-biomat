## Title/abstract precision (first 12 records of each random frame)
| Area | v1 incl./12 | v2 incl./12 | reasons for exclusion (v1+v2, first 12) |
|---|---|---|---|
| A1_imaging | 4/12 | 1/12 | {'NOT_BIOMED': 11, 'WRONG_AREA': 4, 'NO_ML': 4} |
| A2_property | 10/12 | n/a | {'NO_ML': 2} |
| A3_sensors | 2/12 | 8/12 | {'NOT_BIOMED': 12, 'WRONG_AREA': 2} |
| A4_tissue | 1/12 | 4/12 | {'NOT_BIOMED': 8, 'NOT_RESEARCH': 3, 'WRONG_AREA': 4, 'NO_ML': 4} |
| A5_autonomous | 4/12 | 1/12 | {'NOT_BIOMED': 9, 'NO_ML': 10} |
all screened records: 127 | exclusion reasons: {'NOT_BIOMED': 48, 'WRONG_AREA': 13, 'NO_ML': 23, 'NOT_RESEARCH': 3}

## Regex screen: 25 papers, 6818 sentences, 136 candidate hits
| Pattern | hits | papers hit | TP hits | hit precision | papers with >=1 TP |
|---|---|---|---|---|---|
| E_external | 3 | 2 | 0 | 0% | 0/2 |
| E_prospective | 13 | 6 | 9 | 69% | 3/6 |
| S_group | 2 | 2 | 0 | 0% | 0/2 |
| S_ood | 4 | 2 | 2 | 50% | 1/2 |
| S_robust | 31 | 10 | 3 | 10% | 2/10 |
| U1_spread | 14 | 5 | 10 | 71% | 1/5 |
| U2_acquisition | 16 | 4 | 13 | 81% | 3/4 |
| U2_gp | 33 | 3 | 32 | 97% | 2/3 |
| U2_probabilistic | 2 | 2 | 2 | 100% | 2/2 |
| U2_uq | 7 | 1 | 7 | 100% | 1/1 |
| U3_calibration | 11 | 5 | 0 | 0% | 0/5 |
FP categories: [('HOMONYM', 16), ('TRANSFER_LEARNING_ADAPTATION', 16), ('ANALYTICAL_CALIBRATION_CURVE', 9), ('NOT_AN_EVALUATION', 3), ('MEASUREMENT_REPLICATE_SPREAD', 3), ('BACKGROUND_GENERAL', 3), ('LIMITATION_OR_FUTURE_WORK', 2), ('SAMPLE_WISE_LOOCV', 1), ('OTHER_WORK_CITED', 1), ('GENERIC_DISCUSSION', 1), ('NOT_ML', 1), ('CLAIM_WITHOUT_ASSESSMENT', 1), ('HYPERPARAMETER_BO', 1)]
hits by section x verdict: {('RESULTS_DISCUSSION', 'TP'): 24, ('METHODS', 'FP'): 14, ('RESULTS', 'FP'): 13, ('DISCUSSION', 'FP'): 5, ('FIG', 'FP'): 9, ('RESULTS', 'TP'): 10, ('ABSTRACT', 'FP'): 2, ('INTRO', 'FP'): 8, ('RESULTS_DISCUSSION', 'FP'): 5, ('TABLE', 'FP'): 2, ('METHODS', 'TP'): 22, ('FIG', 'TP'): 5, ('ABSTRACT', 'TP'): 4, ('TABLE', 'TP'): 7, ('CONCLUSION', 'TP'): 4, ('DISCUSSION', 'TP'): 1, ('INTRO', 'TP'): 1}

## Paper-level: regex-only code vs verified code
| Area | PMCID | U regex→verified | E1 | EP | S1 |
|---|---|---|---|---|---|
| A1_imaging | PMC10161767 | U0→U0 | 0→1 | 0→0 | 1→1 |
| A1_imaging | PMC12538699 | U3→U0 | 0→0 | 1→0 | 1→0 |
| A1_imaging | PMC8665348 | U0→U1 | 1→0 | 0→1 | 1→1 |
| A1_imaging | PMC8471570 | U0→U0 | 0→0 | 0→0 | 0→0 |
| A1_imaging | PMC12796348 | U3→U0 | 0→0 | 0→0 | 1→1 |
| A2_property | PMC12296543 | U0→U0 | 0→0 | 0→0 | 0→0 |
| A2_property | PMC12666576 | U0→U0 | 0→0 | 0→0 | 0→0 |
| A2_property | PMC11106676 | U0→U1 | 0→0 | 1→0 | 1→0 |
| A2_property | PMC9546622 | U0→U0 | 0→0 | 1→1 | 1→0 |
| A2_property | PMC11165469 | U0→U0 | 0→0 | 0→0 | 1→0 |
| A3_sensors | PMC11377509 | U0→U0 | 0→0 | 0→0 | 1→0 |
| A3_sensors | PMC10039194 | U0→U0 | 0→0 | 0→0 | 0→0 |
| A3_sensors | PMC10407620 | U0→U0 | 0→0 | 0→0 | 0→0 |
| A3_sensors | PMC12499495 | U3→U0 | 0→0 | 0→0 | 0→0 |
| A3_sensors | PMC11348057 | U0→U0 | 0→0 | 0→0 | 1→0 |
| A4_tissue | PMC11468966 | U3→U0 | 0→0 | 0→1 | 0→0 |
| A4_tissue | PMC11085928 | U0→U0 | 0→0 | 0→0 | 0→0 |
| A4_tissue | PMC10070414 | U0→U0 | 0→0 | 0→0 | 1→1 |
| A4_tissue | PMC11170757 | U0→U0 | 0→0 | 0→0 | 0→0 |
| A4_tissue | PMC8778756 | U0→U0 | 0→0 | 0→0 | 0→0 |
| A5_autonomous | PMC12511971 | U1→U0 | 0→0 | 1→1 | 0→0 |
| A5_autonomous | PMC12587402 | U2→U2 | 0→0 | 0→1 | 1→0 |
| A5_autonomous | PMC11422183 | U2→U2 | 0→0 | 1→1 | 0→0 |
| A5_autonomous | PMC12157168 | U3→U2 | 0→0 | 0→1 | 0→0 |
| A5_autonomous | PMC12590431 | U2→U2 | 0→0 | 0→1 | 0→0 |
regex-only agreement with verified code: {'U': '17/25', 'E1': '23/25', 'EP': '18/25', 'S1': '18/25'}

## Paper-level confusion (regex flag vs verified), TP/FP/FN/TN
U>=1: TP=4 FP=5 FN=2 TN=14
U>=2: TP=4 FP=4 FN=0 TN=17
U3: TP=0 FP=5 FN=0 TN=20
E1: TP=0 FP=1 FN=1 TN=23
EP: TP=3 FP=2 FN=5 TN=15
S1: TP=4 FP=7 FN=0 TN=14

## Verified prevalence in pilot (n per area = 5; descriptive only)
A1_imaging {'U>=1': '1/5 (95% CI 0.04-0.62)', 'U>=2': '0/5 (95% CI 0.00-0.43)', 'U3': '0/5 (95% CI 0.00-0.43)', 'E1': '1/5 (95% CI 0.04-0.62)', 'EP': '1/5 (95% CI 0.04-0.62)', 'S1': '3/5 (95% CI 0.23-0.88)'}
A2_property {'U>=1': '1/5 (95% CI 0.04-0.62)', 'U>=2': '0/5 (95% CI 0.00-0.43)', 'U3': '0/5 (95% CI 0.00-0.43)', 'E1': '0/5 (95% CI 0.00-0.43)', 'EP': '1/5 (95% CI 0.04-0.62)', 'S1': '0/5 (95% CI 0.00-0.43)'}
A3_sensors {'U>=1': '0/5 (95% CI 0.00-0.43)', 'U>=2': '0/5 (95% CI 0.00-0.43)', 'U3': '0/5 (95% CI 0.00-0.43)', 'E1': '0/5 (95% CI 0.00-0.43)', 'EP': '0/5 (95% CI 0.00-0.43)', 'S1': '0/5 (95% CI 0.00-0.43)'}
A4_tissue {'U>=1': '0/5 (95% CI 0.00-0.43)', 'U>=2': '0/5 (95% CI 0.00-0.43)', 'U3': '0/5 (95% CI 0.00-0.43)', 'E1': '0/5 (95% CI 0.00-0.43)', 'EP': '1/5 (95% CI 0.04-0.62)', 'S1': '1/5 (95% CI 0.04-0.62)'}
A5_autonomous {'U>=1': '4/5 (95% CI 0.38-0.96)', 'U>=2': '4/5 (95% CI 0.38-0.96)', 'U3': '0/5 (95% CI 0.00-0.43)', 'E1': '0/5 (95% CI 0.00-0.43)', 'EP': '5/5 (95% CI 0.57-1.00)', 'S1': '0/5 (95% CI 0.00-0.43)'}
ALL {'U>=1': '6/25 (95% CI 0.11-0.43)', 'U>=2': '4/25 (95% CI 0.06-0.35)', 'U3': '0/25 (95% CI 0.00-0.13)', 'E1': '1/25 (95% CI 0.01-0.20)', 'EP': '8/25 (95% CI 0.17-0.52)', 'S1': '4/25 (95% CI 0.06-0.35)'}

## Text available to the screen
median sentences/paper: 247 | papers with >=1 supplementary file: 20 / 25
