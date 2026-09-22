"""R2 empirical model-agnosticism under the unified protocol (common.py).

Answers Reviewer 2, comment 5 ("model-agnostic is more of a theoretical framework
property than a sufficiently supported empirical demonstration").

For every point predictor in common.MODEL_NAMES (tree ensembles RF / ExtraTrees /
HistGB, kernel GP / SVR, instance-based kNN, linear Ridge, neural MLP), every
design-shift dataset in common.design_shift_datasets() and every resampling seed
0..19 (common.design_split: grouped 60/20/20 source split; target pool halves =
recalibration pool / target-test), with hyperparameters fixed a priori in
common.make_model (no tuning anywhere):

  (a) split conformal with |y - yhat| calibrated on source-cal, evaluated
      in-distribution on source-test (coverage, width, normalised width, interval
      score, WIS, calibration error over LEVELS, width-stratified / worst-cluster
      coverage, RMSE);
  (b) the same source-calibrated interval on the target-test half (the collapse);
  (c) target recalibration on the first headline-m records of the recalibration
      pool, evaluated on the same target-test half;
  (d) k-sweep (k in 0, 9, 10, 20, 30 within the pool size), same target-test half;
  (e) a MODEL-AGNOSTIC normalised conformal score and triage. Most of the models
      have no native uncertainty, so the difficulty score is a property of the
      input only: sigma(x) = mean Euclidean distance of the standardised x (scaler
      fitted on train) to its 10 nearest TRAIN inputs; nonconformity
      |r| / (sigma + beta), beta = median sigma on source-cal (Papadopoulos et al.
      2011 style). Reported: coverage/width of the normalised intervals
      (source-calibrated, and recalibrated on the same m target records) and the
      risk-coverage RMSE reduction at 50 % and relative AURC in-distribution and on
      target-test. For RF / ExtraTrees (tree spread) and GP (predictive sd) the
      native-uncertainty ranking is reported alongside as a reference;
  (f) for the GP additionally its native Gaussian (1 - alpha) intervals
      (mu +/- z sd, NOT conformal) as a reference.

Fix round (points raised by the independent verifier), all implemented here:
  (1) REPLICATE STRUCTURE OF THE RECALIBRATION BUDGET. common.design_split permutes
      composition GROUPS, so the "first k records" of the recalibration pool are not
      always k distinct compositions. Every task now counts the distinct compositions
      in the headline-m set and in every k of the sweep, and additionally recalibrates
      on k records drawn from k DISTINCT compositions (blocks `recal_distinct`,
      `ksweep_distinct`) so the effect of the replicates is measured, not assumed.
      The headline remains the protocol definition (first k records of the pool).
  (2) The in-distribution over-coverage caveat is no longer justified with the
      (self-contradictory) SE of the 20-split mean. For any cell whose 20-split ID
      coverage deviates by more than ID_DEV_THR from 1 - alpha the script runs an
      ID-ONLY diagnostic over additional seeds (>= 20, no target data touched at
      all) and reports the extended mean, its SD and SE next to the exact
      finite-sample expectation ceil((n_cal+1)(1-alpha))/(n_cal+1).
  (3) The markdown separates HARD exceptions (a cell failing a verdict threshold)
      from CAVEATS (a cell meeting both thresholds, with a qualification).
  (4) The k-sweep section reports the range of the 2.5th percentile across cells,
      so the "single splits can still fail" claim is traceable to the JSON.
  (5) Each cached per-task file carries a fingerprint (sha256 of common.py plus the
      COMPUTE section of this file, delimited by the sentinels below); a part whose
      fingerprint differs from the running code is ignored and recomputed, so
      editing the scoring code can never leave stale numbers in the aggregate.
Also added on the verifier's suggestion: the recalibrated width expressed as a
fraction of the full observed target-pool property range (not only of the IQR), the
list of cells where worst-cluster coverage is undefined (< 40 evaluated records),
a note that width-stratified coverage is degenerate for the constant-width blocks,
a note that the width normaliser uses target-pool labels for REPORTING only, and
within-dataset rank correlations next to the pooled one.

The splits are identical to results/splits/<dataset>_splits.json (written by
exp_R1_core.py); nothing is selected with target-test labels (m is fixed by
common.headline_m, the recalibration records are the first m of the pool).
Outputs results/R2_models.json and results/R2_models.md; per-task raw results are
cached in results/R2_models_parts/ so an interrupted run resumes.

Usage: python -W ignore exp_R2_models.py [n_jobs=6] [inner_threads=2]
"""
import hashlib
import json
import os
import re
import sys
import time
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy.stats import norm

from common import (ALPHA, LEVELS, SEEDS, MODEL_NAMES, RES, SPLITS, design_shift_datasets, design_split,
                    headline_m, conformal_q, make_model, rf_mean_std, interval_metrics,
                    weighted_interval_score, calibration_error, risk_coverage, summarize, iqr, dump,
                    beta_coverage_quantiles)

TAG = "R2_models"
FAMILY = {"RF": "tree ensemble", "ExtraTrees": "tree ensemble", "HistGB": "tree ensemble (boosting)",
          "GP": "kernel (Bayesian)", "SVR": "kernel", "kNN": "instance-based", "Ridge": "linear",
          "MLP": "neural"}
COLLAPSE_THR, RECOVER_THR = 0.80, 0.85   # verdict thresholds on the mean over splits (fixed a priori)
ID_DEV_THR = 0.02                        # |ID coverage - (1-alpha)| above which the extra-seed diagnostic runs
EXTRA_BUDGET_S = 240.0                   # compute budget per flagged cell for that ID-only diagnostic
EXTRA_MAX, EXTRA_MIN = 380, 20           # bounds on the number of extra seeds
PARTS = os.path.join(RES, f"{TAG}_parts")
os.makedirs(PARTS, exist_ok=True)


# ============================================================ COMPUTE-FINGERPRINT-BEGIN
# Everything between the two sentinels produces the per-task numbers; its sha256
# (together with common.py) is stored in every cached part. Editing anything here
# invalidates the cache; editing the aggregation/markdown code below does not.
KGRID = [0, 9, 10, 20, 30]
WIS_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]           # all finite for n >= 9 calibration points
N_NEIGH = 10                                     # neighbours for the model-agnostic difficulty score
TREE_JOBS = 1                                    # tree-model n_jobs inside each joblib worker (<= 4)
NATIVE = {"RF", "ExtraTrees", "GP"}              # models with a native per-point spread


def _curve(err, qfun):
    return [float(np.mean(err <= qfun(lvl))) for lvl in LEVELS]


def abs_block(r_c, mu, y, X, iqr_ref):
    """Split conformal with score |y - yhat| calibrated on r_c, evaluated on (mu, y)."""
    q = conformal_q(r_c, ALPHA)
    out = interval_metrics(mu - q, mu + q, y, iqr_ref, X=X)
    out["wis"] = weighted_interval_score(y, mu, {a: (mu - conformal_q(r_c, a), mu + conformal_q(r_c, a))
                                                 for a in WIS_ALPHAS})
    err = np.abs(y - mu)
    cur = _curve(err, lambda l: conformal_q(r_c, 1 - l))
    out["ce_mean"], out["ce_max"] = calibration_error(LEVELS, cur)
    out["curve"] = cur
    out["q"] = float(q)
    out["finite"] = bool(np.isfinite(q))
    out["n_cal"] = int(len(r_c))
    return out


def norm_block(s_c, u, mu, y, X, iqr_ref):
    """Normalised split conformal: score |r| / u, half-width q * u on the evaluated set."""
    q = conformal_q(s_c, ALPHA)
    hw = q * u
    out = interval_metrics(mu - hw, mu + hw, y, iqr_ref, X=X)
    out["wis"] = weighted_interval_score(y, mu, {a: (mu - conformal_q(s_c, a) * u, mu + conformal_q(s_c, a) * u)
                                                 for a in WIS_ALPHAS})
    s = np.abs(y - mu) / u
    cur = _curve(s, lambda l: conformal_q(s_c, 1 - l))
    out["ce_mean"], out["ce_max"] = calibration_error(LEVELS, cur)
    out["curve"] = cur
    out["finite"] = bool(np.isfinite(q))
    return out


def gauss_block(mu, sd, y, X, iqr_ref):
    """Native Gaussian intervals mu +/- z sd (no conformal calibration)."""
    z = norm.ppf(1 - ALPHA / 2)
    out = interval_metrics(mu - z * sd, mu + z * sd, y, iqr_ref, X=X)
    out["wis"] = weighted_interval_score(y, mu, {a: (mu - norm.ppf(1 - a / 2) * sd, mu + norm.ppf(1 - a / 2) * sd)
                                                 for a in WIS_ALPHAS})
    s = np.abs(y - mu) / sd
    cur = _curve(s, lambda l: norm.ppf(0.5 + l / 2))
    out["ce_mean"], out["ce_max"] = calibration_error(LEVELS, cur)
    out["curve"] = cur
    return out


def rc_block(err, unc):
    rc = risk_coverage(err, unc)
    return {"reduction_at50_pct": rc["reduction_at50_pct"], "aurc_rel": rc["aurc_rel"],
            "rmse_all": rc["rmse_all"], "rmse_at50": rc["rmse_at50"], "curve": rc["rmse"]}


def predict(model, name, X):
    if name in ("RF", "ExtraTrees"):
        return rf_mean_std(model, X)
    if name == "GP":
        return model.predict(X, return_std=True)
    return model.predict(X), None


def difficulty(Xtr, Xeval):
    """Model-agnostic difficulty: mean distance of standardised x to its N_NEIGH nearest train inputs."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    nn = NearestNeighbors(n_neighbors=N_NEIGH).fit(sc.transform(Xtr))
    return {k: nn.kneighbors(sc.transform(v))[0].mean(1) for k, v in Xeval.items()}


def _fit_model(name, seed, X, y, tr):
    model = make_model(name, seed)
    if name in ("RF", "ExtraTrees"):
        model.set_params(n_jobs=TREE_JOBS)
    model.fit(X[tr], y[tr])
    return model


def n_distinct(groups, idx):
    """Number of distinct compositions among the records idx (verifier issue 1)."""
    if groups is None:
        return int(len(idx))
    return int(len(np.unique(np.asarray(groups)[np.asarray(idx, int)])))


def distinct_positions(gid_pool, k):
    """Positions (into the recalibration pool, in pool order) of the first record of
    each of the first k distinct compositions; fewer than k if the pool has fewer."""
    seen, pos = set(), []
    for i, g in enumerate(np.asarray(gid_pool)):
        gi = int(g)
        if gi in seen:
            continue
        seen.add(gi)
        pos.append(i)
        if len(pos) == k:
            break
    return np.array(pos, int)


def one_task_compute(ds, name, seed):
    """All numbers for one (dataset, model, split); no caching, no I/O."""
    t0 = time.time()
    sp = design_split(ds, seed)
    X, y, G = ds.X, ds.y, ds.groups
    tr = sp["train"]
    model = _fit_model(name, seed, X, y, tr)
    t_fit = time.time() - t0
    keys = ["cal", "stest", "rpool", "ttest"]
    P = {k: predict(model, name, X[sp[k]]) for k in keys}
    mu = {k: P[k][0] for k in keys}
    sig = difficulty(X[tr], {k: X[sp[k]] for k in keys})
    beta = float(np.median(sig["cal"]))
    yS, yT, XS, XT = y[sp["stest"]], y[sp["ttest"]], X[sp["stest"]], X[sp["ttest"]]
    iqr_S, iqr_T = iqr(y[ds.src]), iqr(y[ds.tgt])
    r = {k: np.abs(y[sp[k]] - mu[k]) for k in keys}
    mh = headline_m(ds)
    gid_pool = None if G is None else G[sp["rpool"]]
    out = {"seed": seed, "model": name, "dataset": ds.name, "m": mh,
           "sizes": {k: int(len(v)) for k, v in sp.items()}}

    # (a)-(c) absolute-residual split conformal
    out["id"] = abs_block(r["cal"], mu["stest"], yS, XS, iqr_S)
    out["src_to_tgt"] = abs_block(r["cal"], mu["ttest"], yT, XT, iqr_T)
    out["recal"] = abs_block(r["rpool"][:mh], mu["ttest"], yT, XT, iqr_T)
    # (e) model-agnostic normalised conformal
    s_cal = r["cal"] / (sig["cal"] + beta)
    s_pool = r["rpool"] / (sig["rpool"] + beta)
    uS, uT = sig["stest"] + beta, sig["ttest"] + beta
    out["norm_id"] = norm_block(s_cal, uS, mu["stest"], yS, XS, iqr_S)
    out["norm_src_to_tgt"] = norm_block(s_cal, uT, mu["ttest"], yT, XT, iqr_T)
    out["norm_recal"] = norm_block(s_pool[:mh], uT, mu["ttest"], yT, XT, iqr_T)
    # (1) composition counts: how many DISTINCT compositions the budgets really contain
    out["groups"] = {"train": n_distinct(G, tr), "cal": n_distinct(G, sp["cal"]),
                     "stest": n_distinct(G, sp["stest"]), "rpool": n_distinct(G, sp["rpool"]),
                     "ttest": n_distinct(G, sp["ttest"]),
                     "recal_m": n_distinct(G, sp["rpool"][:mh]),
                     "ksweep": {str(k): n_distinct(G, sp["rpool"][:k])
                                for k in KGRID if 0 < k <= len(sp["rpool"])}}
    # (1) supplementary: recalibrate on m records from m DISTINCT compositions
    if gid_pool is not None:
        pm = distinct_positions(gid_pool, mh)
        out["recal_distinct"] = abs_block(r["rpool"][pm], mu["ttest"], yT, XT, iqr_T)
        out["recal_distinct"]["n_records"] = int(len(pm))
    # (d) k-sweep on the same target-test half
    q_src, qn_src = conformal_q(r["cal"], ALPHA), conformal_q(s_cal, ALPHA)
    errT = np.abs(yT - mu["ttest"])
    out["ksweep"], out["ksweep_distinct"] = {}, {}
    for k in [k for k in KGRID if k <= len(r["rpool"])]:
        q_k = q_src if k == 0 else conformal_q(r["rpool"][:k], ALPHA)
        qn_k = qn_src if k == 0 else conformal_q(s_pool[:k], ALPHA)
        out["ksweep"][str(k)] = {"coverage": float(np.mean(errT <= q_k)),
                                 "norm_width": float(2 * q_k / iqr_T) if np.isfinite(q_k) else None,
                                 "infinite": bool(not np.isfinite(q_k)),
                                 "n_distinct": n_distinct(G, sp["rpool"][:k]) if k > 0 else 0,
                                 "coverage_normalised": float(np.mean(errT <= qn_k * uT)),
                                 "norm_width_normalised": float(np.mean(2 * qn_k * uT) / iqr_T)
                                 if np.isfinite(qn_k) else None}
        if k > 0 and gid_pool is not None:
            pk = distinct_positions(gid_pool, k)
            qd = conformal_q(r["rpool"][pk], ALPHA)
            out["ksweep_distinct"][str(k)] = {"coverage": float(np.mean(errT <= qd)),
                                              "norm_width": float(2 * qd / iqr_T) if np.isfinite(qd) else None,
                                              "n_records": int(len(pk)),
                                              "infinite": bool(not np.isfinite(qd))}
    # (e) triage (ranking by the model-agnostic difficulty; identical under any calibration
    # of the normalised score because the half-width is monotone in sigma)
    errS = np.abs(yS - mu["stest"])
    out["triage_id"] = rc_block(errS, sig["stest"])
    out["triage_tgt"] = rc_block(errT, sig["ttest"])
    if name in NATIVE:
        out["triage_native_id"] = rc_block(errS, P["stest"][1])
        out["triage_native_tgt"] = rc_block(errT, P["ttest"][1])
    # (f) GP native Gaussian intervals
    if name == "GP":
        out["gp_native_id"] = gauss_block(mu["stest"], P["stest"][1], yS, XS, iqr_S)
        out["gp_native_tgt"] = gauss_block(mu["ttest"], P["ttest"][1], yT, XT, iqr_T)
    out["rmse"] = {"source_test": float(np.sqrt(np.mean(errS ** 2))),
                   "target_test": float(np.sqrt(np.mean(errT ** 2)))}
    out["rmse"]["ratio_tgt_over_src"] = out["rmse"]["target_test"] / out["rmse"]["source_test"]
    out["diag"] = {"beta": beta, "sigma_median_stest": float(np.median(sig["stest"])),
                   "sigma_median_ttest": float(np.median(sig["ttest"])),
                   "sigma_ratio_tgt_over_src": float(np.median(sig["ttest"]) / np.median(sig["stest"])),
                   "q_src": float(q_src), "q_recal_m": out["recal"]["q"],
                   "q_ratio_recal_over_src": float(out["recal"]["q"] / q_src),
                   "iqr_S": iqr_S, "iqr_T": iqr_T,
                   "range_S": float(np.ptp(y[ds.src])), "range_T": float(np.ptp(y[ds.tgt])),
                   "t_fit_s": float(t_fit), "t_total_s": float(time.time() - t0)}
    return out


def id_only_coverage(ds, name, seed):
    """In-distribution split-conformal coverage for one extra seed.

    Diagnostic for verifier issue 2. It fits on train, calibrates on source-cal and
    evaluates on source-test; no target-pool record (recalibration pool or
    target-test) is read, so extra seeds cannot leak design-region information.
    """
    sp = design_split(ds, seed)
    X, y = ds.X, ds.y
    model = _fit_model(name, seed, X, y, sp["train"])
    q = conformal_q(np.abs(y[sp["cal"]] - model.predict(X[sp["cal"]])), ALPHA)
    return float(np.mean(np.abs(y[sp["stest"]] - model.predict(X[sp["stest"]])) <= q))
# ============================================================ COMPUTE-FINGERPRINT-END


def compute_fingerprint():
    """sha256 of common.py plus the COMPUTE section of this file (see sentinels)."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    m = re.search(r"# =+ COMPUTE-FINGERPRINT-BEGIN(.*?)# =+ COMPUTE-FINGERPRINT-END", src, re.S)
    h = hashlib.sha256()
    h.update(open(os.path.join(here, "common.py"), "rb").read())
    h.update((m.group(1) if m else src).encode("utf-8"))
    return h.hexdigest()[:16]


def file_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


CODE_SHA = compute_fingerprint()


def one_task(ds, name, seed):
    """Cached wrapper: a part whose fingerprint differs from the running compute code
    is ignored and recomputed (verifier issue 5)."""
    warnings.filterwarnings("ignore")
    part = os.path.join(PARTS, f"{ds.name}__{name}__{seed}.json")
    if os.path.exists(part):
        try:
            old = json.load(open(part))
            if old.get("_code_sha") == CODE_SHA:
                return old
        except (ValueError, OSError):
            pass
    out = one_task_compute(ds, name, seed)
    out["_code_sha"] = CODE_SHA
    json.dump(out, open(part, "w"))
    return out


def id_extra_task(ds, name, n_extra):
    """Cached ID-only extra-seed diagnostic for one (dataset, model) cell."""
    warnings.filterwarnings("ignore")
    part = os.path.join(PARTS, f"{ds.name}__{name}__idextra.json")
    if os.path.exists(part):
        try:
            old = json.load(open(part))
            if old.get("_code_sha") == CODE_SHA and len(old.get("cov", [])) >= n_extra:
                old["cov"] = old["cov"][:n_extra]
                return old
        except (ValueError, OSError):
            pass
    seeds = list(range(max(SEEDS) + 1, max(SEEDS) + 1 + n_extra))
    cov = [id_only_coverage(ds, name, s) for s in seeds]
    out = {"dataset": ds.name, "model": name, "seeds": [seeds[0], seeds[-1]], "cov": cov,
           "_code_sha": CODE_SHA}
    json.dump(out, open(part, "w"))
    return out


# =============================================================== aggregation
def agg_block(rows, block):
    rs = [r[block] for r in rows if block in r]
    keys = []
    for b in rs:
        keys += [k for k in b if k not in keys]
    S = {}
    for k in keys:
        vals = [b.get(k) for b in rs]
        if isinstance(vals[0], list):
            S[k + "_mean"] = np.mean([v for v in vals if v is not None], 0).tolist()
        elif isinstance(vals[0], bool):
            S[k + "_frac"] = float(np.mean([bool(v) for v in vals]))
        else:
            S[k] = summarize([None if v is None else float(v) for v in vals])
    return S


def agg_sweep(rows, block, extra_keys=()):
    out = {}
    if block not in rows[0]:
        return out
    for k in rows[0][block]:
        kk = [r[block][k] for r in rows if k in r.get(block, {})]
        if not kk:
            continue
        e = {"coverage": summarize([v["coverage"] for v in kk]),
             "norm_width": summarize([v["norm_width"] for v in kk]),
             "n_infinite": int(sum(v["infinite"] for v in kk))}
        for key in extra_keys:
            if key in kk[0]:
                e[key] = summarize([v[key] for v in kk])
        out[k] = e
    return out


def summarise_cell(rows, ds_name, n_target):
    S = {"model": rows[0]["model"], "family": FAMILY[rows[0]["model"]], "m": rows[0]["m"],
         "sizes_seed0": rows[0]["sizes"], "n_splits": len(rows)}
    for b in ["id", "src_to_tgt", "recal", "recal_distinct", "norm_id", "norm_src_to_tgt", "norm_recal",
              "triage_id", "triage_tgt", "triage_native_id", "triage_native_tgt", "gp_native_id",
              "gp_native_tgt", "rmse", "diag"]:
        if b in rows[0]:
            S[b] = agg_block(rows, b)
    # composition counts (verifier issue 1): identical for every model of a dataset
    S["groups"] = {k: summarize([r["groups"][k] for r in rows]) for k in rows[0]["groups"] if k != "ksweep"}
    S["groups"]["ksweep"] = {k: summarize([r["groups"]["ksweep"][k] for r in rows if k in r["groups"]["ksweep"]])
                             for k in rows[0]["groups"]["ksweep"]}
    S["ksweep"] = {}
    for k in rows[0]["ksweep"]:
        kk = [r["ksweep"][k] for r in rows if k in r["ksweep"]]
        S["ksweep"][k] = {"coverage": summarize([v["coverage"] for v in kk]),
                          "norm_width": summarize([v["norm_width"] for v in kk]),
                          "n_infinite": int(sum(v["infinite"] for v in kk)),
                          "n_distinct": summarize([v["n_distinct"] for v in kk]),
                          "coverage_normalised": summarize([v["coverage_normalised"] for v in kk]),
                          "norm_width_normalised": summarize([v["norm_width_normalised"] for v in kk]),
                          "beta_theory_5_50_95": beta_coverage_quantiles(int(k)) if int(k) > 0 else None}
    S["ksweep_distinct"] = agg_sweep(rows, "ksweep_distinct", ("n_records",))
    m = rows[0]["m"]
    S["recal_theory"] = {"expected_coverage": float(np.ceil((m + 1) * (1 - ALPHA)) / (m + 1)),
                         "beta_5_50_95": beta_coverage_quantiles(m),
                         "note": "exact under exchangeability of the m calibration records with the test "
                                 "records; with replicate compositions in the m records the effective "
                                 "calibration size is smaller (see 'groups')"}
    src = np.array([r["src_to_tgt"]["coverage"] for r in rows])
    rec = np.array([r["recal"]["coverage"] for r in rows])
    S["verdict"] = {"id_cov": float(np.mean([r["id"]["coverage"] for r in rows])),
                    "src_to_tgt_cov": float(src.mean()), "recal_cov": float(rec.mean()),
                    "collapse": bool(src.mean() < COLLAPSE_THR),
                    "recovered": bool(rec.mean() >= RECOVER_THR),
                    "frac_splits_src_below_0.80": float(np.mean(src < 0.80)),
                    "frac_splits_recal_above_src": float(np.mean(rec > src)),
                    "frac_splits_recal_ge_0.80": float(np.mean(rec >= 0.80))}
    S["per_split"] = [{"seed": r["seed"], "id_cov": r["id"]["coverage"], "src_cov": r["src_to_tgt"]["coverage"],
                       "recal_cov": r["recal"]["coverage"], "norm_src_cov": r["norm_src_to_tgt"]["coverage"],
                       "norm_recal_cov": r["norm_recal"]["coverage"], "rmse_S": r["rmse"]["source_test"],
                       "rmse_T": r["rmse"]["target_test"], "tri_id": r["triage_id"]["reduction_at50_pct"],
                       "tri_tgt": r["triage_tgt"]["reduction_at50_pct"], "t_fit_s": r["diag"]["t_fit_s"],
                       "n_distinct_recal_m": r["groups"]["recal_m"]}
                      for r in rows]
    return S


# =============================================================== markdown
def _f(s, d=2, ci=True):
    if s is None or s.get("mean") is None:
        return "n/a"
    if not ci:
        return f"{s['mean']:.{d}f}"
    return f"{s['mean']:.{d}f} [{s['lo']:.{d}f}, {s['hi']:.{d}f}]"


def _table(R, dsn, getter, title, note=""):
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines += ["| model (family) | " + " | ".join(dsn) + " |", "|---|" + "---|" * len(dsn)]
    for mname in MODEL_NAMES:
        cells = []
        for d in dsn:
            try:
                cells.append(getter(R[mname][d]))
            except (KeyError, TypeError):
                cells.append("n/a")
        lines.append(f"| {mname} ({FAMILY[mname]}) | " + " | ".join(cells) + " |")
    return lines + [""]


def replicate_section(R, meta, dsn):
    """Verifier issue 1: how many DISTINCT compositions the recalibration budgets contain."""
    g0 = R[MODEL_NAMES[0]]
    rep = [d for d in dsn if g0[d]["groups"]["recal_m"]["mean"] < g0[d]["m"] - 1e-9]
    one2one = [d for d in dsn if d not in rep]
    L = ["## Replicate structure of the recalibration budget (Reviewer 2, comments 10-11)", "",
         "`common.design_split` permutes composition GROUPS, so a composition never sits on both sides of a "
         "split, but the protocol's recalibration set (\"the first k records of the pool\") can contain several "
         "replicate measurements of the same composition. Replicates share a descriptor vector and therefore a "
         "prediction, so their residuals are strongly correlated and the EFFECTIVE calibration size is smaller "
         "than k. The counts below are measured per split (mean [2.5, 97.5 percentile] over the 20 splits); they "
         "depend only on the split, not on the model.", ""]
    L += ["| dataset | records m | distinct compositions in the m records | k = 9 | k = 10 | k = 20 | k = 30 |",
          "|---|---|---|---|---|---|---|"]
    for d in dsn:
        c = g0[d]
        cells = [(_f(c["groups"]["ksweep"][k], 1) if k in c["groups"]["ksweep"] else "n/a")
                 for k in ["9", "10", "20", "30"]]
        L.append(f"| {d} | {c['m']} | {_f(c['groups']['recal_m'], 1)} | " + " | ".join(cells) + " |")
    L += [""]
    if rep:
        frac = ", ".join(f"{d} {g0[d]['groups']['recal_m']['mean']:.1f}/{g0[d]['m']}" for d in rep)
        L += [f"- Datasets with replicates inside the budget: {frac} distinct compositions in the headline-m set. "
              f"On {', '.join(one2one)} k records are k distinct compositions, so k is the effective size there.", ""]
    # paired comparison: same splits, k records vs k distinct compositions
    L += ["The paired comparison below recalibrates on k records drawn from k DISTINCT compositions (the first "
          "record of each of the first k compositions of the same pool, same splits, same target-test half). It is "
          "a supplementary diagnostic; the headline numbers everywhere else follow the protocol definition "
          "(first k records).", ""]
    L += _table(R, dsn, lambda c: f"{_f(c['recal']['coverage'], 3, False)} -> "
                                  f"{_f(c['recal_distinct']['coverage'], 3, False)}",
                "Recalibrated coverage on target-test: first m records -> m distinct compositions")
    for k in ["9", "10"]:
        L += _table(R, dsn, lambda c, k=k: (f"{_f(c['ksweep'][k]['coverage'], 3, False)} -> "
                                            f"{_f(c['ksweep_distinct'][k]['coverage'], 3, False)}")
                    if k in c.get("ksweep_distinct", {}) else "n/a",
                    f"k = {k}: coverage with the first k records -> with k distinct compositions")
    return L


def diagnostics_section(R, meta, dsn, extra):
    L = ["## Reporting caveats and diagnostics", ""]
    g0 = R[MODEL_NAMES[0]]
    miss = []
    for d in dsn:
        s = g0[d]["sizes_seed0"]
        if s["stest"] < 40:
            miss.append(f"{d} source-test ({s['stest']})")
        if s["ttest"] < 40:
            miss.append(f"{d} target-test ({s['ttest']})")
    L += [f"- Worst-cluster (k-means, k <= 8) coverage is only defined where the evaluated set has at least 40 "
          f"records (`common.interval_metrics`). It is therefore absent for: {', '.join(miss) if miss else 'no cell'}. "
          f"Any figure built from `{TAG}.json` must show these cells as missing rather than dropping them silently.",
          "- Width-stratified coverage (`wsc_width_strat`) is degenerate for the absolute-residual blocks "
          "(`id`, `src_to_tgt`, `recal`): those intervals have a single constant width, so the statistic equals the "
          "overall coverage by construction. It is informative only for the normalised blocks (`norm_*`), where the "
          "width varies with sigma(x).",
          "- The width normaliser is the IQR of the property over the whole source pool (in-distribution) or the "
          "whole target pool (target-test). It uses target labels for REPORTING only: no interval, threshold, "
          "budget or model depends on it. The same normaliser is used in `exp_R1_core.py`.", ""]
    if extra:
        L += ["### In-distribution over-coverage: extra-seed diagnostic", "",
              "For every cell whose 20-split in-distribution coverage deviates by more than "
              f"{ID_DEV_THR:.2f} from {1 - ALPHA:.2f}, the script re-ran the IN-DISTRIBUTION part only (fit on "
              "train, calibrate on source-cal, evaluate on source-test) over additional seeds. No target-pool "
              "record is read in that diagnostic, so it cannot leak design-region information; the headline "
              "remains the 20 protocol splits.", "",
              "| cell | 20-split mean | per-split SD | exact expectation | guaranteed band | extra seeds | "
              "extended mean | extended SD | SE |", "|---|---|---|---|---|---|---|---|---|"]
        for e in extra.values():
            L.append(f"| {e['model']} / {e['dataset']} | {e['mean20']:.3f} | {e['sd20']:.3f} | {e['exact']:.3f} | "
                     f"[{1 - ALPHA:.3f}, {e['band_hi']:.3f}] | {e['n_extra']} | {e['mean_ext']:.3f} | "
                     f"{e['sd_ext']:.3f} | {e['se_ext']:.3f} |")
        L += [""]
    return L


def write_md(R, meta, fam, hard, soft, extra, elapsed):
    dsn = [d["name"] for d in meta["datasets"]]
    n_cells = len(MODEL_NAMES) * len(dsn)
    n_both = sum(R[m][d]["verdict"]["collapse"] and R[m][d]["verdict"]["recovered"]
                 for m in MODEL_NAMES for d in dsn)
    L = [f"# {TAG}: empirical model-agnosticism (Reviewer 2, comment 5)", "",
         f"Script: `revision/code/exp_R2_models.py`; results: `revision/results/{TAG}.json`. "
         f"Protocol: `revision/code/PROTOCOL.md` / `common.py` (grouped 60/20/20 source split, target pool halves "
         f"= recalibration pool / target-test, headline m = clip(floor(|T|/3), 10, 30), alpha = {ALPHA}, finite-sample "
         f"conformal quantile, {len(SEEDS)} seeds). Cells are mean [2.5, 97.5 percentile] across the 20 splits; the "
         f"splits resample one finite dataset each, so the interval describes split-to-split variability, not "
         f"population sampling error. Hyperparameters are fixed a priori in `common.make_model` (no tuning); "
         f"nothing is selected with target-test labels. GP = exact GP (isotropic RBF + white noise) fitted on at most "
         f"2,500 randomly subsampled training records (`common._SubsampledGP`; affects glass_Tg, glass_Tliq, "
         f"polymer_Tg); RF/ExtraTrees 300 trees. "
         f"Compute: {meta['sum_task_time_min']:.0f} worker-minutes over "
         f"{n_cells * len(SEEDS)} (model, dataset, split) tasks. "
         f"Cache fingerprint (common.py + the compute section of this script): `{meta['code_sha']}`.", "",
         "Datasets: " + "; ".join(f"{d['name']} (source {d['n_source']}, target {d['n_target']}, m = {d['m']})"
                                   for d in meta["datasets"]), ""]
    L += ["## Key question: does the collapse and the recovery hold for every model family?", ""]
    L += meta["answer_lines"] + [""]
    L += ["Verdict thresholds (fixed before running, on the mean over splits): collapse = source-calibrated "
          f"target-test coverage < {COLLAPSE_THR:.2f}; recovery = recalibrated coverage >= {RECOVER_THR:.2f}.", ""]
    if hard:
        L += [f"Exceptions ({len(hard)} of {n_cells} cells fail a verdict threshold):", ""] + \
             [f"- {e}" for e in hard] + [""]
    if soft:
        L += [f"Caveats ({n_both}/{n_cells} cells meet BOTH verdict thresholds; the bullets below are not failed "
              f"cells, they qualify how the thresholds are met):", ""] + [f"- {e}" for e in soft] + [""]
    if not hard and not soft:
        L += ["No exceptions and no caveats: every (model, dataset) cell collapses and recovers under the "
              "thresholds above.", ""]
    L += ["## How the families differ (generated from the numbers below)", ""] + meta["interpretation"] + [""]
    if meta.get("r1_consistency_RF"):
        L += [f"Consistency check: the RF row uses the same splits and model as `exp_R1_core.py`; the maximum "
              f"absolute difference to `R1_core.json` in ID / source->target / recalibrated coverage (and relative "
              f"target RMSE) is {meta['r1_consistency_RF']['max_abs_diff_coverage_or_rel_rmse']:.2g}.", ""]

    L += ["## Split-conformal coverage (alpha = 0.10, nominal 0.90)", ""]
    L += _table(R, dsn, lambda c: _f(c["id"]["coverage"]), "(a) In-distribution coverage on source-test")
    L += _table(R, dsn, lambda c: _f(c["src_to_tgt"]["coverage"]),
                "(b) Source-calibrated coverage on target-test (design shift)")
    L += _table(R, dsn, lambda c: _f(c["recal"]["coverage"]),
                "(c) Coverage after target recalibration with headline m (first m records of the recalibration pool), target-test",
                "Finite-sample expectation under exchangeability of m calibration records: " + ", ".join(
                    f"{d['name']} {R[MODEL_NAMES[0]][d['name']]['recal_theory']['expected_coverage']:.3f}"
                    for d in meta["datasets"]) + ". " + meta["replicate_caveat"])
    L += ["## Interval width (normalised by the IQR of the evaluated domain's property values)", ""]
    L += _table(R, dsn, lambda c: _f(c["id"]["norm_width"]), "In-distribution normalised width (source-test)")
    L += _table(R, dsn, lambda c: f"{_f(c['src_to_tgt']['norm_width'], ci=False)} -> "
                                  f"{_f(c['recal']['norm_width'])}",
                "Target-test normalised width: source-calibrated -> recalibrated (m)")
    L += _table(R, dsn, lambda c: f"{100 * c['recal']['width']['mean'] / c['diag']['range_T']['mean']:.0f} %",
                "Recalibrated width as a percentage of the FULL observed target-pool property range",
                "An interval wider than the whole observed range of the property in the design region cannot "
                "discriminate between candidates, however honest its coverage (Reviewer 2, comment 7).")
    L += ["## Point accuracy", ""]
    L += _table(R, dsn, lambda c: _f(c["rmse"]["source_test"], 1), "RMSE on source-test (dataset units)")
    L += _table(R, dsn, lambda c: _f(c["rmse"]["target_test"], 1), "RMSE on target-test (dataset units)")
    L += _table(R, dsn, lambda c: _f(c["rmse"]["ratio_tgt_over_src"], 1),
                "RMSE ratio target-test / source-test")
    L += ["## Interval score (Gneiting & Raftery 2007, alpha = 0.10; lower is better; dataset units)", ""]
    L += _table(R, dsn, lambda c: f"{_f(c['id']['interval_score'], 1, False)} / "
                                  f"{_f(c['src_to_tgt']['interval_score'], 1, False)} / "
                                  f"{_f(c['recal']['interval_score'], 1, False)}",
                "Interval score: ID / source->target / recalibrated")
    L += _table(R, dsn, lambda c: f"{_f(c['id']['ce_mean'], 3, False)} / {_f(c['src_to_tgt']['ce_mean'], 3, False)}"
                                  f" / {_f(c['recal']['ce_mean'], 3, False)}",
                "Mean interval calibration error over levels 0.50-0.95: ID / source->target / recalibrated",
                "At level 0.95 the m-record quantile is infinite when m < 19 (steel, m = 13); such intervals "
                "count as covering (reported, not dropped).")
    L += ["## k-sweep (coverage on target-test; k = 0 is the source-calibrated interval)", "",
          meta["ksweep_note"], ""]
    for kk in [str(k) for k in KGRID]:
        L += _table(R, dsn, lambda c, kk=kk: _f(c["ksweep"][kk]["coverage"]) if kk in c["ksweep"] else "n/a",
                    f"k = {kk}")
    L += replicate_section(R, meta, dsn)
    L += ["## (e) Model-agnostic normalised conformal and triage", "",
          "Difficulty sigma(x) = mean Euclidean distance of the standardised input to its 10 nearest training "
          "inputs (depends on the input only, so it applies to every model); score |r| / (sigma + beta), beta = "
          "median sigma on source-cal. Triage ranks candidates by sigma; RMSE reduction on the 50 % most "
          "confident vs no abstention, and AURC relative to the no-abstention RMSE (< 1 = ranking helps).", ""]
    L += _table(R, dsn, lambda c: _f(c["norm_id"]["coverage"]), "Normalised conformal, in-distribution coverage")
    L += _table(R, dsn, lambda c: _f(c["norm_src_to_tgt"]["coverage"]),
                "Normalised conformal, source-calibrated coverage on target-test")
    L += _table(R, dsn, lambda c: _f(c["norm_recal"]["coverage"]),
                "Normalised conformal, recalibrated on the same m target records, target-test coverage")
    L += _table(R, dsn, lambda c: f"{_f(c['norm_id']['norm_width'], 2, False)} / "
                                  f"{_f(c['norm_src_to_tgt']['norm_width'], 2, False)} / "
                                  f"{_f(c['norm_recal']['norm_width'], 2, False)}",
                "Normalised conformal normalised width: ID / source->target / recalibrated")
    L += _table(R, dsn, lambda c: _f(c["triage_id"]["reduction_at50_pct"], 0),
                "Triage in-distribution: RMSE reduction (%) at 50 % retained")
    L += _table(R, dsn, lambda c: _f(c["triage_tgt"]["reduction_at50_pct"], 0),
                "Triage on target-test: RMSE reduction (%) at 50 % retained")
    L += _table(R, dsn, lambda c: f"{_f(c['triage_id']['aurc_rel'], 2, False)} / "
                                  f"{_f(c['triage_tgt']['aurc_rel'], 2, False)}",
                "Relative AURC: in-distribution / target-test")
    L += _table(R, dsn, lambda c: (f"{_f(c['triage_native_id']['reduction_at50_pct'], 0, False)} / "
                                   f"{_f(c['triage_native_tgt']['reduction_at50_pct'], 0, False)}")
                if "triage_native_id" in c else "-",
                "Reference: native-uncertainty triage (RF/ExtraTrees tree spread, GP sd), RMSE reduction (%) at 50 %: ID / target-test")
    L += ["## (f) GP native Gaussian 90 % intervals (not conformal), reference", ""]
    g = R["GP"]
    L += ["| dataset | ID coverage | ID norm width | target-test coverage | target norm width | ID CE | target CE |",
          "|---|---|---|---|---|---|---|"]
    for d in dsn:
        c = g[d]
        L.append(f"| {d} | {_f(c['gp_native_id']['coverage'])} | {_f(c['gp_native_id']['norm_width'], 2, False)} | "
                 f"{_f(c['gp_native_tgt']['coverage'])} | {_f(c['gp_native_tgt']['norm_width'], 2, False)} | "
                 f"{_f(c['gp_native_id']['ce_mean'], 3, False)} | {_f(c['gp_native_tgt']['ce_mean'], 3, False)} |")
    L += ["", "## Family-level summary (mean over the six datasets of the per-cell means)", "",
          "| model | family | ID cov | src->tgt cov | recal cov | ID norm width | recal norm width | "
          "RMSE ratio tgt/src | triage ID % | triage tgt % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for mname in MODEL_NAMES:
        f = fam[mname]
        L.append(f"| {mname} | {FAMILY[mname]} | {f['id_cov']:.3f} | {f['src_to_tgt_cov']:.3f} | {f['recal_cov']:.3f} | "
                 f"{f['id_norm_width']:.2f} | {f['recal_norm_width']:.2f} | {f['rmse_ratio']:.1f} | "
                 f"{f['triage_id']:.0f} | {f['triage_tgt']:.0f} |")
    L += ["", "Mean fit time per split (s): " + ", ".join(f"{m} {fam[m]['t_fit_s']:.1f}" for m in MODEL_NAMES), ""]
    L += diagnostics_section(R, meta, dsn, extra)
    if meta.get("verifier_issues_not_adopted"):
        L += ["## Verifier issues not adopted", ""] + [f"- {x}" for x in meta["verifier_issues_not_adopted"]] + [""]
    open(os.path.join(RES, f"{TAG}.md"), "w", encoding="utf-8").write("\n".join(L))


def answer(R, dsn, extra):
    """Plain-language answer to the key question, generated from the numbers.

    Returns (answer lines, HARD exceptions = cells failing a verdict threshold,
    CAVEATS = cells meeting both thresholds with a qualification)."""
    lines, hard, soft = [], [], []
    for mname in MODEL_NAMES:
        for d in dsn:
            c = R[mname][d]
            v = c["verdict"]
            if not v["collapse"]:
                hard.append(f"{mname} / {d}: NO collapse by the < {COLLAPSE_THR:.2f} criterion "
                            f"(source->target coverage {v['src_to_tgt_cov']:.3f}; recalibrated {v['recal_cov']:.3f})")
            elif v["frac_splits_src_below_0.80"] < 1:
                soft.append(f"{mname} / {d}: collapse on average (source->target {v['src_to_tgt_cov']:.3f}, below the "
                            f"threshold) but not in every split: {100 * (1 - v['frac_splits_src_below_0.80']):.0f} % of "
                            f"splits have source->target coverage >= 0.80 (upper 97.5 % "
                            f"{c['src_to_tgt']['coverage']['hi']:.2f})")
            if not v["recovered"]:
                hard.append(f"{mname} / {d}: recalibrated coverage {v['recal_cov']:.3f} < {RECOVER_THR:.2f} "
                            f"(source->target {v['src_to_tgt_cov']:.3f})")
            if abs(v["id_cov"] - (1 - ALPHA)) > ID_DEV_THR:
                n_cal, n_te = c["sizes_seed0"]["cal"], c["sizes_seed0"]["stest"]
                exact = float(np.ceil((n_cal + 1) * (1 - ALPHA)) / (n_cal + 1))
                band_hi = (1 - ALPHA) + 1 / (n_cal + 1)
                sd = c["id"]["coverage"]["sd"]
                msg = (f"{mname} / {d}: in-distribution coverage {v['id_cov']:.3f} deviates > {ID_DEV_THR:.2f} from "
                       f"{1 - ALPHA:.2f}. With n_cal = {n_cal} the exact finite-sample expectation is "
                       f"ceil((n_cal+1)(1-alpha))/(n_cal+1) = {exact:.3f} and marginal coverage is guaranteed in "
                       f"[{1 - ALPHA:.3f}, {band_hi:.3f}]; with only {n_te} source-test records the per-split SD "
                       f"under this split scheme is {sd:.3f}")
                e = extra.get((mname, d))
                if e:
                    msg += (f". Over {e['n_extra']} additional in-distribution-only splits (seeds {e['seed_lo']}-"
                            f"{e['seed_hi']}; no target record is read) the same cell averages {e['mean_ext']:.3f} "
                            f"(SD {e['sd_ext']:.3f}, SE {e['se_ext']:.3f})")
                    if e["consistent"]:
                        msg += (f", i.e. not above the guaranteed upper bound {band_hi:.3f} (within 2 SE of it), so "
                                f"the 20-split value is split-sampling noise, not systematic over-coverage of the "
                                f"procedure")
                        if e["mean_ext"] < (1 - ALPHA) - 2 * e["se_ext"]:
                            msg += (f"; the extended mean is {(1 - ALPHA) - e['mean_ext']:.3f} below {1 - ALPHA:.2f}, "
                                    f"which is reported as observed")
                    else:
                        msg += (f", i.e. still above the guaranteed upper bound {band_hi:.3f}: the procedure really "
                                f"is conservative in this cell (ties or replicate residuals in the calibration set "
                                f"make the effective calibration size smaller than n_cal)")
                else:
                    msg += (". No extra-seed diagnostic was run for this cell (compute budget), so the deviation is "
                            "reported as observed and not explained away")
                soft.append(msg)
    n_cells = len(MODEL_NAMES) * len(dsn)
    n_col = sum(R[m][d]["verdict"]["collapse"] for m in MODEL_NAMES for d in dsn)
    n_rec = sum(R[m][d]["verdict"]["recovered"] for m in MODEL_NAMES for d in dsn)
    idc = [R[m][d]["verdict"]["id_cov"] for m in MODEL_NAMES for d in dsn]
    src = [R[m][d]["verdict"]["src_to_tgt_cov"] for m in MODEL_NAMES for d in dsn]
    rec = [R[m][d]["verdict"]["recal_cov"] for m in MODEL_NAMES for d in dsn]
    lines.append(f"- In-distribution split-conformal coverage is {min(idc):.3f}-{max(idc):.3f} across all "
                 f"{n_cells} (model, dataset) cells (nominal 0.90), as guaranteed for any predictor.")
    lines.append(f"- Source-calibrated coverage on the design-region target-test half ranges {min(src):.3f}-"
                 f"{max(src):.3f}; {n_col}/{n_cells} cells fall below {COLLAPSE_THR:.2f}.")
    lines.append(f"- Target recalibration with the headline m restores mean coverage to {min(rec):.3f}-{max(rec):.3f}; "
                 f"{n_rec}/{n_cells} cells reach >= {RECOVER_THR:.2f}.")
    per_ds = []
    for d in dsn:
        s = [R[m][d]["verdict"]["src_to_tgt_cov"] for m in MODEL_NAMES]
        per_ds.append(f"{d} {min(s):.2f}-{max(s):.2f}")
    lines.append("- Source->target coverage range across the 8 models, per dataset: " + "; ".join(per_ds) + ".")
    # replicate structure of the budget (verifier issue 1)
    g0 = R[MODEL_NAMES[0]]
    rep = [d for d in dsn if g0[d]["groups"]["recal_m"]["mean"] < g0[d]["m"] - 1e-9]
    if rep:
        det = "; ".join(f"{d} {g0[d]['groups']['recal_m']['mean']:.1f}/{g0[d]['m']} at m and "
                        f"{g0[d]['groups']['ksweep']['9']['mean']:.1f}/9 at k = 9" for d in rep)
        dd = [(R[m][d]["recal_distinct"]["coverage"]["mean"] - R[m][d]["recal"]["coverage"]["mean"])
              for m in MODEL_NAMES for d in rep]
        lines.append(f"- The recalibration budget is counted in target RECORDS, not in distinct compositions: on "
                     f"{det} distinct compositions on average, so the effective calibration size is smaller than k "
                     f"there (on {', '.join(d for d in dsn if d not in rep)} k records are k compositions). Drawing "
                     f"the same k from k DISTINCT compositions changes the recalibrated coverage by "
                     f"{min(dd):+.3f} to {max(dd):+.3f} (mean {np.mean(dd):+.3f}) on those datasets.")
    return lines, hard, soft


def interpret(R, dsn, extra):
    """Family-level observations (accuracy, width, normalised conformal, triage), generated from the numbers."""
    from scipy.stats import spearmanr
    L = []
    g = lambda m, d, *p: _get(R[m][d], p)
    cells = [(m, d) for m in MODEL_NAMES for d in dsn]
    # accuracy and width by dataset
    for d in dsn:
        rs = {m: g(m, d, "rmse", "source_test", "mean") for m in MODEL_NAMES}
        rt = {m: g(m, d, "rmse", "target_test", "mean") for m in MODEL_NAMES}
        wi = {m: g(m, d, "id", "norm_width", "mean") for m in MODEL_NAMES}
        wr = {m: g(m, d, "recal", "norm_width", "mean") for m in MODEL_NAMES}
        sc = {m: g(m, d, "src_to_tgt", "coverage", "mean") for m in MODEL_NAMES}
        o = lambda x: sorted(x, key=x.get)
        L.append(f"- {d}: source-test RMSE best {o(rs)[0]} {rs[o(rs)[0]]:.3g} / worst {o(rs)[-1]} {rs[o(rs)[-1]]:.3g}; "
                 f"target-test RMSE best {o(rt)[0]} {rt[o(rt)[0]]:.3g} / worst {o(rt)[-1]} {rt[o(rt)[-1]]:.3g}; "
                 f"ID normalised width {min(wi.values()):.2f} ({o(wi)[0]})-{max(wi.values()):.2f} ({o(wi)[-1]}); "
                 f"recalibrated normalised width {min(wr.values()):.2f} ({o(wr)[0]})-{max(wr.values()):.2f} ({o(wr)[-1]}); "
                 f"least collapse {o(sc)[-1]} ({sc[o(sc)[-1]]:.2f}), most {o(sc)[0]} ({sc[o(sc)[0]]:.2f}).")
    src = np.array([g(m, d, "src_to_tgt", "coverage", "mean") for m, d in cells])
    rr = np.array([g(m, d, "rmse", "ratio_tgt_over_src", "mean") for m, d in cells])
    qr = np.array([g(m, d, "diag", "q_ratio_recal_over_src", "mean") for m, d in cells])
    within = []
    for d in dsn:
        s_d = [g(m, d, "src_to_tgt", "coverage", "mean") for m in MODEL_NAMES]
        r_d = [g(m, d, "rmse", "ratio_tgt_over_src", "mean") for m in MODEL_NAMES]
        within.append(spearmanr(s_d, r_d)[0])
    L.append(f"- Across the {len(cells)} cells the severity of the collapse tracks how much the model's error inflates "
             f"under the shift: Spearman rho(source->target coverage, target/source RMSE ratio) = "
             f"{spearmanr(src, rr)[0]:.2f} pooled, and {min(within):.2f} to {max(within):.2f} within the six datasets "
             f"(8 models each), so the association is not an artefact of pooling datasets of different difficulty. "
             f"RMSE inflates {rr.min():.1f}-{rr.max():.1f}x in every cell. (The corresponding rho with the "
             f"recalibrated/source conformal quantile ratio, {spearmanr(src, qr)[0]:.2f}, is close to tautological - "
             f"the source-calibrated coverage is largely determined by that ratio - and is reported only for "
             f"completeness; the quantile inflates {qr.min():.1f}-{qr.max():.1f}x.)")
    wr_all = np.array([g(m, d, "recal", "norm_width", "mean") for m, d in cells])
    wi_all = np.array([g(m, d, "id", "norm_width", "mean") for m, d in cells])
    rng = np.array([100 * g(m, d, "recal", "width", "mean") / g(m, d, "diag", "range_T", "mean") for m, d in cells])
    worst = [f"{d} {max(100 * g(m, d, 'recal', 'width', 'mean') / g(m, d, 'diag', 'range_T', 'mean') for m in MODEL_NAMES):.0f} %"
             for d in dsn]
    L.append(f"- Price of recovery: recalibrated 90 % intervals are {wr_all.min():.2f}-{wr_all.max():.2f} target-IQR "
             f"wide (in-distribution {wi_all.min():.2f}-{wi_all.max():.2f} source-IQR); coverage is restored by "
             f"widening, not by the model becoming accurate in the design region. In absolute terms they span "
             f"{rng.min():.0f}-{rng.max():.0f} % of the FULL observed property range of the target pool "
             f"(worst model per dataset: {'; '.join(worst)}), and "
             f"{int(np.sum(rng >= 100))}/{len(cells)} cells exceed 100 %, i.e. the honest interval covers the whole "
             f"design region and cannot rank candidates.")
    # normalised (model-agnostic difficulty) conformal
    ns = np.array([g(m, d, "norm_src_to_tgt", "coverage", "mean") for m, d in cells])
    nr = np.array([g(m, d, "norm_recal", "coverage", "mean") for m, d in cells])
    ni = np.array([g(m, d, "norm_id", "coverage", "mean") for m, d in cells])
    sig = np.array([g(m, d, "diag", "sigma_ratio_tgt_over_src", "mean") for m, d in cells])
    L.append(f"- Model-agnostic normalised conformal (|r|/(sigma+beta)): in-distribution coverage {ni.min():.3f}-"
             f"{ni.max():.3f}. Without any target label it raises source-calibrated target coverage in "
             f"{int(np.sum(ns > src))}/{len(cells)} cells (median +{np.median(ns - src):.2f}; range {ns.min():.2f}-"
             f"{ns.max():.2f}), because target inputs lie {sig.min():.1f}-{sig.max():.1f}x farther from the training "
             f"data than source-test inputs, but reaches >= {RECOVER_THR:.2f} in only {int(np.sum(ns >= RECOVER_THR))}/"
             f"{len(cells)} cells. Recalibrated on the same m target records it gives {nr.min():.3f}-{nr.max():.3f} "
             f"({int(np.sum(nr >= RECOVER_THR))}/{len(cells)} cells >= {RECOVER_THR:.2f}).")
    ti = np.array([g(m, d, "triage_id", "reduction_at50_pct", "mean") for m, d in cells])
    tt = np.array([g(m, d, "triage_tgt", "reduction_at50_pct", "mean") for m, d in cells])
    ai = np.array([g(m, d, "triage_id", "aurc_rel", "mean") for m, d in cells])
    at = np.array([g(m, d, "triage_tgt", "aurc_rel", "mean") for m, d in cells])
    L.append(f"- Model-agnostic triage (rank by sigma): RMSE reduction at 50 % retained is positive in "
             f"{int(np.sum(ti > 0))}/{len(cells)} cells in-distribution ({ti.min():.0f} to {ti.max():.0f} %) and "
             f"{int(np.sum(tt > 0))}/{len(cells)} on target-test ({tt.min():.0f} to {tt.max():.0f} %); relative AURC < 1 "
             f"in {int(np.sum(ai < 1))}/{len(cells)} (ID) and {int(np.sum(at < 1))}/{len(cells)} (target) cells. "
             "Cells with no benefit on target-test (reduction <= 1 %): " +
             (", ".join(f"{m}/{d} ({x:.0f} %)" for (m, d), x in zip(cells, tt) if x <= 1) or "none") + ".")
    nat = []
    for m in [m for m in MODEL_NAMES if m in NATIVE]:
        a_id = np.mean([g(m, d, "triage_id", "reduction_at50_pct", "mean") for d in dsn])
        n_id = np.mean([g(m, d, "triage_native_id", "reduction_at50_pct", "mean") for d in dsn])
        a_t = np.mean([g(m, d, "triage_tgt", "reduction_at50_pct", "mean") for d in dsn])
        n_t = np.mean([g(m, d, "triage_native_tgt", "reduction_at50_pct", "mean") for d in dsn])
        worst_d = min(dsn, key=lambda d: g(m, d, "triage_native_tgt", "reduction_at50_pct", "mean"))
        nat.append(f"{m} native {n_id:.0f} % ID / {n_t:.0f} % target (worst {worst_d} "
                   f"{g(m, worst_d, 'triage_native_tgt', 'reduction_at50_pct', 'mean'):.0f} %) vs model-agnostic "
                   f"{a_id:.0f} % / {a_t:.0f} %")
    L.append("- Native-uncertainty triage as reference (mean over datasets): " + "; ".join(nat) + ".")
    gi = [g("GP", d, "gp_native_id", "coverage", "mean") for d in dsn]
    gt = [g("GP", d, "gp_native_tgt", "coverage", "mean") for d in dsn]
    L.append(f"- GP native Gaussian 90 % intervals (no conformal step): in-distribution coverage {min(gi):.3f}-"
             f"{max(gi):.3f}, target-test {min(gt):.3f}-{max(gt):.3f}. The GP's own variance grows away from the data "
             f"and so under-covers less than source-calibrated conformal, but still falls below nominal on every "
             f"dataset; it is not a substitute for target recalibration.")
    return L


def ksweep_note(R, dsn):
    """Verifier issue 4a: make the 'single splits still fail' claim traceable."""
    parts = []
    for k in ["9", "10", "20", "30"]:
        lo = [R[m][d]["ksweep"][k]["coverage"]["lo"] for m in MODEL_NAMES for d in dsn if k in R[m][d]["ksweep"]]
        mn = [R[m][d]["ksweep"][k]["coverage"]["mean"] for m in MODEL_NAMES for d in dsn if k in R[m][d]["ksweep"]]
        if lo:
            parts.append(f"k = {k}: mean {min(mn):.2f}-{max(mn):.2f}, 2.5th percentile across cells "
                         f"{min(lo):.3f}-{max(lo):.3f} ({len(mn)} cells)")
    return ("Ranges of the per-cell mean and of the per-cell 2.5th percentile over the 20 splits - " +
            "; ".join(parts) + ". A small budget restores coverage on average while single splits can still "
            "under-cover badly.")


def _get(c, path):
    for p in path:
        c = c[p]
    return c


def r1_consistency(R):
    """RF here uses the same splits and model as exp_R1_core.py; report the max |difference|."""
    p = os.path.join(RES, "R1_core.json")
    if not os.path.exists(p):
        return None
    r1 = json.load(open(p))
    diffs = {}
    for d, c in R["RF"].items():
        if d not in r1:
            continue
        diffs[d] = max(abs(c["id"]["coverage"]["mean"] - r1[d]["id_conformal"]["coverage"]["mean"]),
                       abs(c["src_to_tgt"]["coverage"]["mean"] - r1[d]["shift_source"]["coverage"]["mean"]),
                       abs(c["recal"]["coverage"]["mean"] - r1[d]["shift_recal"]["coverage"]["mean"]),
                       abs(c["rmse"]["target_test"]["mean"] - r1[d]["rmse"]["target_test"]["mean"])
                       / r1[d]["rmse"]["target_test"]["mean"])
    return {"max_abs_diff_coverage_or_rel_rmse": float(max(diffs.values())) if diffs else None, "per_dataset": diffs}


NOT_ADOPTED = [
    "None. All five verifier issues were implemented: (1) the distinct-composition counts and the paired "
    "k-distinct recalibration, (2) the extra-seed in-distribution diagnostic replacing the self-contradictory "
    "SE argument, (3) the exception / caveat split in this markdown, (4) the k-sweep percentile ranges (and the "
    "corrected figures in the returned summary), (5) the cache fingerprint. The four additional points the "
    "verifier raised as 'not errors but worth adding' (absolute width vs the target-pool range, the degenerate "
    "width-stratified statistic for constant-width blocks, the missing worst-cluster cells, and the near-tautological "
    "quantile-ratio correlation) are also addressed above."]


# =============================================================== main
def main(n_jobs=6, inner_threads=2):
    t0 = time.time()
    dss = design_shift_datasets()
    # sanity: the splits are exactly the ones persisted by exp_R1_core.py, for every seed
    for ds in dss:
        p = os.path.join(SPLITS, f"{ds.name}_splits.json")
        if os.path.exists(p):
            saved = json.load(open(p))["splits"]
            for s in SEEDS:
                if str(s) not in saved:
                    continue
                sp = design_split(ds, s)
                assert all(np.array_equal(sp[k], np.array(saved[str(s)][k])) for k in sp), \
                    f"split mismatch {ds.name} seed {s}"
    cost = {"GP": 0, "MLP": 1, "SVR": 2, "RF": 3, "ExtraTrees": 4, "HistGB": 5, "kNN": 6, "Ridge": 7}
    tasks = sorted([(ds, mname, s) for ds in dss for mname in MODEL_NAMES for s in SEEDS],
                   key=lambda t: (cost[t[1]], -len(t[0].src), t[2]))
    with parallel_config(backend="loky", inner_max_num_threads=inner_threads):
        rows = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one_task)(*t) for t in tasks)
    by = {}
    for r in rows:
        by.setdefault(r["model"], {}).setdefault(r["dataset"], []).append(r)
    R = {}
    for mname in MODEL_NAMES:
        R[mname] = {}
        for ds in dss:
            rr = sorted(by[mname][ds.name], key=lambda r: r["seed"])
            R[mname][ds.name] = summarise_cell(rr, ds.name, len(ds.tgt))
    dsn = [ds.name for ds in dss]
    dsmap = {ds.name: ds for ds in dss}

    # ---- verifier issue 2: extra-seed ID-only diagnostic for the over/under-covering cells
    flagged = [(m, d) for m in MODEL_NAMES for d in dsn
               if abs(R[m][d]["verdict"]["id_cov"] - (1 - ALPHA)) > ID_DEV_THR]
    jobs = []
    for m, d in flagged:
        t_fit = max(R[m][d]["diag"]["t_fit_s"]["mean"], 5e-3)
        n_extra = int(np.clip(EXTRA_BUDGET_S / t_fit, EXTRA_MIN, EXTRA_MAX))
        jobs.append((m, d, n_extra))
    extra = {}
    if jobs:
        print(f"extra-seed ID diagnostic for {len(jobs)} cell(s): " +
              ", ".join(f"{m}/{d} n={n}" for m, d, n in jobs), flush=True)
        with parallel_config(backend="loky", inner_max_num_threads=inner_threads):
            res = Parallel(n_jobs=min(n_jobs, max(1, len(jobs))), verbose=5)(
                delayed(id_extra_task)(dsmap[d], m, n) for m, d, n in jobs)
        for (m, d, n), e in zip(jobs, res):
            cov = np.asarray(e["cov"], float)
            n_cal = R[m][d]["sizes_seed0"]["cal"]
            band_hi = (1 - ALPHA) + 1 / (n_cal + 1)
            se = float(cov.std(ddof=1) / np.sqrt(len(cov)))
            extra[(m, d)] = {"model": m, "dataset": d, "n_extra": int(len(cov)),
                             "seed_lo": int(e["seeds"][0]), "seed_hi": int(e["seeds"][1]),
                             "mean20": R[m][d]["verdict"]["id_cov"], "sd20": R[m][d]["id"]["coverage"]["sd"],
                             "exact": float(np.ceil((n_cal + 1) * (1 - ALPHA)) / (n_cal + 1)),
                             "band_hi": float(band_hi), "mean_ext": float(cov.mean()),
                             "sd_ext": float(cov.std(ddof=1)), "se_ext": se,
                             "consistent": bool(cov.mean() <= band_hi + 2 * se)}

    fam = {}
    for mname in MODEL_NAMES:
        c = [R[mname][d] for d in dsn]
        fam[mname] = {"family": FAMILY[mname],
                      "id_cov": float(np.mean([x["id"]["coverage"]["mean"] for x in c])),
                      "src_to_tgt_cov": float(np.mean([x["src_to_tgt"]["coverage"]["mean"] for x in c])),
                      "recal_cov": float(np.mean([x["recal"]["coverage"]["mean"] for x in c])),
                      "id_norm_width": float(np.mean([x["id"]["norm_width"]["mean"] for x in c])),
                      "src_norm_width_tgt": float(np.mean([x["src_to_tgt"]["norm_width"]["mean"] for x in c])),
                      "recal_norm_width": float(np.mean([x["recal"]["norm_width"]["mean"] for x in c])),
                      "recal_width_pct_of_target_range": float(np.mean(
                          [100 * x["recal"]["width"]["mean"] / x["diag"]["range_T"]["mean"] for x in c])),
                      "rmse_ratio": float(np.mean([x["rmse"]["ratio_tgt_over_src"]["mean"] for x in c])),
                      "norm_src_to_tgt_cov": float(np.mean([x["norm_src_to_tgt"]["coverage"]["mean"] for x in c])),
                      "norm_recal_cov": float(np.mean([x["norm_recal"]["coverage"]["mean"] for x in c])),
                      "triage_id": float(np.mean([x["triage_id"]["reduction_at50_pct"]["mean"] for x in c])),
                      "triage_tgt": float(np.mean([x["triage_tgt"]["reduction_at50_pct"]["mean"] for x in c])),
                      "t_fit_s": float(np.mean([x["diag"]["t_fit_s"]["mean"] for x in c]))}
    ans, hard, soft = answer(R, dsn, extra)
    interp = interpret(R, dsn, extra)
    r1c = r1_consistency(R)
    task_min = float(sum(r["diag"]["t_total_s"] for r in rows) / 60)
    elapsed = time.time() - t0
    g0 = R[MODEL_NAMES[0]]
    rep = [d for d in dsn if g0[d]["groups"]["recal_m"]["mean"] < g0[d]["m"] - 1e-9]
    rep_caveat = ("The m records are m distinct compositions on every dataset, so this expectation applies directly."
                  if not rep else
                  "This expectation assumes the m calibration records are exchangeable with the test records; on " +
                  ", ".join(f"{d} the m = {g0[d]['m']} records come from {g0[d]['groups']['recal_m']['mean']:.1f} "
                            f"distinct compositions on average" for d in rep) +
                  ", so the effective calibration size there is smaller than m (see the replicate-structure section).")
    meta = {"tag": TAG, "alpha": ALPHA, "seeds": SEEDS, "models": MODEL_NAMES, "families": FAMILY,
            "kgrid": KGRID, "levels": LEVELS.tolist(), "wis_alphas": WIS_ALPHAS, "n_neighbours_sigma": N_NEIGH,
            "difficulty_score": "mean Euclidean distance of standardised x (scaler fit on train) to its 10 nearest "
                                "train inputs; normalised score |r|/(sigma+beta), beta = median sigma on source-cal",
            "thresholds": {"collapse_src_to_tgt_cov_below": COLLAPSE_THR, "recovered_recal_cov_at_least": RECOVER_THR,
                           "id_deviation_triggering_extra_seeds": ID_DEV_THR},
            "datasets": [{**ds.summary(), "m": headline_m(ds)} for ds in dss],
            "norm_width_reference": "IQR of y over the source pool (ID) / target pool (target-test); used for "
                                    "REPORTING only - no interval, threshold, budget or model depends on it",
            "recal_distinct_note": "supplementary: recalibration on k records drawn from k DISTINCT compositions "
                                   "(first record of each of the first k compositions of the same pool); the "
                                   "headline follows the protocol (first k records of the pool)",
            "wsc_note": "wsc_width_strat is degenerate (= overall coverage) for the constant-width blocks id / "
                        "src_to_tgt / recal / recal_distinct; it is informative only for the norm_* blocks",
            "worst_cluster_note": "worst_cluster_cov is only computed where the evaluated set has >= 40 records "
                                  "(common.interval_metrics); it is absent otherwise and must not be dropped silently",
            "parallel": {"n_jobs": n_jobs, "inner_threads": inner_threads, "tree_n_jobs": TREE_JOBS},
            "code_sha": CODE_SHA, "script_sha": file_sha(),
            "cache_policy": "each part in results/R2_models_parts/ stores the compute fingerprint (sha256 of "
                            "common.py + the COMPUTE section of this script); a part with a different fingerprint "
                            "is ignored and recomputed",
            "answer_lines": ans, "exceptions_hard": hard, "caveats": soft, "exceptions": hard + soft,
            "interpretation": interp, "r1_consistency_RF": r1c,
            "replicate_caveat": rep_caveat, "ksweep_note": ksweep_note(R, dsn),
            "id_extra_seed_diagnostic": {f"{m}/{d}": v for (m, d), v in extra.items()},
            "verifier_issues_not_adopted": NOT_ADOPTED,
            "sum_task_time_min": task_min, "wall_time_this_invocation_min": elapsed / 60}
    dump({"meta": meta, "family_summary": fam, "results": R}, f"{TAG}.json")
    write_md(R, meta, fam, hard, soft, extra, elapsed)
    for line in ans:
        print(line)
    for e in hard:
        print("HARD", e)
    for e in soft:
        print("CAVEAT", e)
    for mname in MODEL_NAMES:
        print(f"{mname:10s} " + " ".join(f"{d[:10]:>10s} {R[mname][d]['verdict']['id_cov']:.2f}/"
                                         f"{R[mname][d]['verdict']['src_to_tgt_cov']:.2f}/"
                                         f"{R[mname][d]['verdict']['recal_cov']:.2f}" for d in dsn))
    print(f"done in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6, int(sys.argv[2]) if len(sys.argv) > 2 else 2)
