"""R1 core re-analysis under the unified protocol (common.py), reference model = RF.

For each of the six design-shift datasets and each of 20 resampling splits:
  * in-distribution calibration curve on source-test (split conformal vs the
    random-forest tree-spread heuristic turned into Gaussian intervals);
  * source-calibrated coverage on the target-test half (the collapse);
  * target recalibration with the headline m and a k-sweep drawn from the
    disjoint recalibration pool, always evaluated on the same target-test half;
  * the full metric set requested by Reviewer 2 (coverage, width, normalised
    width, interval score, WIS, calibration error, width-stratified and
    worst-cluster coverage, risk-coverage/AURC, abstention);
  * conformal-ranked triage (normalised conformal score) in-distribution and,
    after recalibration, on the shifted target half.
Outputs results/R1_core.json and results/splits/<dataset>_splits.json.
"""
import sys
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from common import (ALPHA, LEVELS, SEEDS, design_shift_datasets, design_split, headline_m,
                    conformal_q, make_model, rf_mean_std, interval_metrics, weighted_interval_score,
                    calibration_error, risk_coverage, summarize, iqr, save_splits, dump,
                    beta_coverage_quantiles, abstention_rate)

KGRID = [0, 5, 9, 10, 15, 20, 30, 40, 60, 80, 120, 200, 300]
WIS_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]   # all finite for n >= 9 calibration points


def one_split(ds, seed):
    sp = design_split(ds, seed)
    X, y = ds.X, ds.y
    m = make_model("RF", seed).fit(X[sp["train"]], y[sp["train"]])
    mu = {k: rf_mean_std(m, X[v]) for k, v in sp.items() if k != "train"}
    r_cal = np.abs(y[sp["cal"]] - mu["cal"][0])
    yS, yT = y[sp["stest"]], y[sp["ttest"]]
    iqr_S, iqr_T = iqr(y[ds.src]), iqr(y[ds.tgt])
    out = {"seed": seed, "sizes": {k: int(len(v)) for k, v in sp.items()}}

    # ---- in-distribution calibration curves (conformal vs RF heuristic) ----
    cov_c, cov_h = [], []
    for lvl in LEVELS:
        q = conformal_q(r_cal, 1 - lvl); z = norm.ppf(0.5 + lvl / 2)
        cov_c.append(float(np.mean(np.abs(yS - mu["stest"][0]) <= q)))
        cov_h.append(float(np.mean(np.abs(yS - mu["stest"][0]) <= z * mu["stest"][1])))
    out["id_curve_conformal"], out["id_curve_heuristic"] = cov_c, cov_h
    out["id_ce_conformal"], out["id_ce_heuristic"] = calibration_error(LEVELS, cov_c), calibration_error(LEVELS, cov_h)
    q_src = conformal_q(r_cal, ALPHA)
    out["id_conformal"] = interval_metrics(mu["stest"][0] - q_src, mu["stest"][0] + q_src, yS, iqr_S,
                                           X=X[sp["stest"]])
    zq = norm.ppf(1 - ALPHA / 2)
    out["id_heuristic"] = interval_metrics(mu["stest"][0] - zq * mu["stest"][1], mu["stest"][0] + zq * mu["stest"][1],
                                           yS, iqr_S, X=X[sp["stest"]])

    # ---- shift: source-calibrated vs recalibrated, on the same target-test half ----
    muT = mu["ttest"][0]
    out["shift_source"] = interval_metrics(muT - q_src, muT + q_src, yT, iqr_T, X=X[sp["ttest"]])
    ints = {a: (muT - conformal_q(r_cal, a), muT + conformal_q(r_cal, a)) for a in WIS_ALPHAS}
    out["shift_source"]["wis"] = weighted_interval_score(yT, muT, ints)
    out["shift_source"]["curve"] = [float(np.mean(np.abs(yT - muT) <= conformal_q(r_cal, 1 - l))) for l in LEVELS]
    r_pool = np.abs(y[sp["rpool"]] - mu["rpool"][0])
    mh = headline_m(ds)
    q_m = conformal_q(r_pool[:mh], ALPHA)
    out["shift_recal"] = interval_metrics(muT - q_m, muT + q_m, yT, iqr_T, X=X[sp["ttest"]])
    ints = {a: (muT - conformal_q(r_pool[:mh], a), muT + conformal_q(r_pool[:mh], a)) for a in WIS_ALPHAS}
    out["shift_recal"]["wis"] = weighted_interval_score(yT, muT, ints)
    out["shift_recal"]["curve"] = [float(np.mean(np.abs(yT - muT) <= conformal_q(r_pool[:mh], 1 - l))) for l in LEVELS]
    out["shift_recal"]["m"] = mh
    # abstention: share of target candidates whose recalibrated interval is wider than the
    # in-distribution conformal width (i.e. cannot be decided at the familiar precision)
    out["shift_recal"]["abstain_vs_id_width"] = abstention_rate(np.full(len(yT), 2 * q_m), 2 * q_src)

    ks = [k for k in KGRID if k <= len(r_pool)]
    out["ksweep"] = {}
    for k in ks:
        q_k = q_src if k == 0 else conformal_q(r_pool[:k], ALPHA)
        out["ksweep"][k] = {"coverage": float(np.mean(np.abs(yT - muT) <= q_k)),
                            "norm_width": float(2 * q_k / iqr_T) if np.isfinite(q_k) else None}

    # ---- conformal-ranked triage (normalised score = |r| / tree spread) ----
    sd_cal = mu["cal"][1] + 1e-6
    qn = conformal_q(r_cal / sd_cal, ALPHA)
    uS = mu["stest"][1] * qn
    out["triage_id"] = risk_coverage(np.abs(yS - mu["stest"][0]), uS)
    # under shift: recalibrate the normalised score on the m target records
    sd_pool = mu["rpool"][1][:mh] + 1e-6
    qn_t = conformal_q(r_pool[:mh] / sd_pool, ALPHA)
    uT = mu["ttest"][1] * qn_t
    out["triage_shift"] = risk_coverage(np.abs(yT - muT), uT)
    out["shift_recal_normalised"] = interval_metrics(muT - uT, muT + uT, yT, iqr_T)
    out["rmse"] = {"source_test": float(np.sqrt(np.mean((yS - mu["stest"][0]) ** 2))),
                   "target_test": float(np.sqrt(np.mean((yT - muT) ** 2)))}
    return out


def agg(rows, path):
    vals = []
    for r in rows:
        v = r
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                break
        vals.append(v)
    return summarize(vals)


def main(n_jobs=10):
    results = {}
    for ds in design_shift_datasets():
        save_splits(ds)
        rows = Parallel(n_jobs=n_jobs)(delayed(one_split)(ds, s) for s in SEEDS)
        S = {"dataset": ds.summary(), "m": headline_m(ds), "sizes_seed0": rows[0]["sizes"]}
        for block in ["id_conformal", "id_heuristic", "shift_source", "shift_recal", "shift_recal_normalised"]:
            S[block] = {k: agg(rows, [block, k]) for k in rows[0][block] if k not in ("curve",)}
        S["id_curve_conformal"] = np.mean([r["id_curve_conformal"] for r in rows], 0).tolist()
        S["id_curve_heuristic"] = np.mean([r["id_curve_heuristic"] for r in rows], 0).tolist()
        S["shift_curve_source"] = np.mean([r["shift_source"]["curve"] for r in rows], 0).tolist()
        S["shift_curve_recal"] = np.mean([r["shift_recal"]["curve"] for r in rows], 0).tolist()
        S["id_ce"] = {"conformal_mean": summarize([r["id_ce_conformal"][0] for r in rows]),
                      "heuristic_mean": summarize([r["id_ce_heuristic"][0] for r in rows])}
        S["ksweep"] = {k: {"coverage": agg(rows, ["ksweep", k, "coverage"]),
                           "norm_width": agg(rows, ["ksweep", k, "norm_width"]),
                           "beta_theory_5_50_95": beta_coverage_quantiles(k) if k > 0 else None}
                       for k in rows[0]["ksweep"]}
        for t in ["triage_id", "triage_shift"]:
            S[t] = {"reduction_at50_pct": agg(rows, [t, "reduction_at50_pct"]),
                    "aurc_rel": agg(rows, [t, "aurc_rel"]),
                    "curve_mean": np.mean([r[t]["rmse"] for r in rows], 0).tolist(),
                    "rmse_all": agg(rows, [t, "rmse_all"]), "grid": rows[0][t]["grid"]}
        S["rmse"] = {k: agg(rows, ["rmse", k]) for k in ["source_test", "target_test"]}
        S["per_split"] = [{"seed": r["seed"], "src_cov": r["shift_source"]["coverage"],
                           "recal_cov": r["shift_recal"]["coverage"], "id_cov": r["id_conformal"]["coverage"],
                           "tri_id": r["triage_id"]["reduction_at50_pct"]} for r in rows]
        results[ds.name] = S
        print(f"{ds.name:12s} ID {S['id_conformal']['coverage']['mean']:.3f} | heur {S['id_heuristic']['coverage']['mean']:.3f}"
              f" | src->tgt {S['shift_source']['coverage']['mean']:.3f} [{S['shift_source']['coverage']['lo']:.2f},{S['shift_source']['coverage']['hi']:.2f}]"
              f" | recal(m={S['m']}) {S['shift_recal']['coverage']['mean']:.3f} [{S['shift_recal']['coverage']['lo']:.2f},{S['shift_recal']['coverage']['hi']:.2f}]"
              f" | nw {S['shift_source']['norm_width']['mean']:.2f}->{S['shift_recal']['norm_width']['mean']:.2f}"
              f" | triID {S['triage_id']['reduction_at50_pct']['mean']:.0f}% triT {S['triage_shift']['reduction_at50_pct']['mean']:.0f}%", flush=True)
    dump(results, "R1_core.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
