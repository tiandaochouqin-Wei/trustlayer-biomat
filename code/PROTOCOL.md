# Revision analysis protocol (binding for every script in revision/code)

All new or re-run analyses import `common.py` and follow these rules. They answer
Reviewer 1 comment 1 and Reviewer 2 comments 1, 10, 11 and 12.

## Data and shift definitions
- Loaders are in `common.py` (`glass_dataset`, `steel_dataset`, `polymer_dataset`,
  `design_shift_datasets`). Do not redefine thresholds elsewhere.
- Replicate measurements of an identical composition share a group id
  (`Dataset.groups`); splits are grouped, so one composition never sits on both
  sides of any split.

## Split protocol (per seed s in 0..19)
- Source pool: train 60 % / source-calibration 20 % / source-test 20 %.
- Target pool: recalibration pool = first half, target-test = second half.
  Recalibration sets of size k are the first k records of the pool; every k is
  evaluated on the same target-test half. Headline m = clip(floor(|T|/3), 10, 30).
- Intermediate records (neither source nor target) are excluded and counted.
- `check_disjoint` must pass; persist indices with `save_splits` when a script
  introduces a new split family.
- Anything selected using target labels (recalibration, budget choice, thresholds,
  hyperparameters) must be selected from the recalibration pool only. The
  target-test half is touched only for final evaluation.

## Conformal defaults
- alpha = 0.10; score |y - yhat| unless stated; finite-sample quantile
  `conformal_q` (returns +inf when k < ceil((k+1)(1-alpha)), i.e. k < 9 at alpha = 0.10;
  report such cases as "infinite interval", never silently drop them).
- Classification: LAC score 1 - p_y (sets = {c : p_c >= 1 - q}).

## Reporting
- Every quantity per split; report mean and 2.5–97.5 percentile interval across
  the 20 splits (`summarize`). Say explicitly that splits resample one dataset.
- Metric set (`interval_metrics` etc.): coverage, mean width, width normalised by
  the IQR of the evaluated domain's property values, interval score
  (Gneiting & Raftery 2007), weighted interval score (Bracher et al. 2021),
  interval calibration error over levels {0.5,...,0.95}, width-stratified coverage,
  worst-cluster (k-means, k = 8) coverage, risk-coverage curve / relative AURC,
  RMSE reduction on the 50 % most confident, abstention rate where a decision rule exists.
- Results go to `revision/results/<script-tag>.json` via `dump`; a short
  human-readable summary to `revision/results/<script-tag>.md`.
- No figures in experiment scripts (figures are produced later from the JSON
  under one style); keep scripts deterministic (seeded) and re-runnable.
- Model hyperparameters fixed a priori in `make_model`; no tuning on target data.
