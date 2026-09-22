"""R9 (legacy Figure 11a): weighted (likelihood-ratio) conformal under the design-region
shift, re-run under the unified protocol (common.py / PROTOCOL.md).

What is computed, per dataset (six design-shift datasets) and per split seed 0..19:
  * RF point predictor (make_model("RF")) trained on the source-train split only;
    absolute-residual scores on source-cal.
  * Source-calibrated split-conformal coverage on the target-test half (vanilla) and
    few-shot target recalibration with the headline m drawn from the recalibration
    pool (identical to R1_core; used as the reference arms).
  * Covariate-shift weights w(x) = e(x) / (1 - e(x)) from a probabilistic domain
    classifier e(x) = P(target | x) trained ONLY on source-train inputs (label 0) versus
    target recalibration-pool inputs (label 1). Target-test inputs and all labels are
    never used to build the weights. The class-prior factor n_S / n_T is a constant that
    cancels in the weighted quantile and in ESS/n, so it is omitted. Weights are formed
    from the classifier's log-odds, w = exp(logit e(x)), rescaled by a common constant
    per test point (no probability clipping, so saturated probabilities do not flatten
    the weights); the share of calibration points with e(x) < 1e-6 is reported.
      - "rff_D"  : logistic regression (C = 1, fixed a priori) on a random-Fourier-feature
                   map of the train-standardised inputs, D in {5, 20, 50, 100, 250, 500};
                   ONE map (W, b) per (seed, D) applied to every domain; RBF bandwidth =
                   median distance between source-train and recal-pool inputs.
                   RFF is a disclosed proxy for a high-capacity embedding, not a
                   foundation model.
      - "raw_LR" : logistic regression on the train-standardised raw features
                   (the standard low-dimensional weighted-conformal set-up).
      - "raw_HGB": histogram gradient-boosting classifier on the raw features
                   (flexible low-dimensional classifier).
      - "*_clip0.99": the practitioner's usual fix for extreme likelihood ratios, added
                   for rff_50, rff_500, raw_LR and raw_HGB: every log-weight (calibration
                   and test) is capped at the 99th percentile of the calibration
                   log-weights. This buys back finite intervals; whether they are valid
                   is exactly the question, and the answer is reported.
  * Weighted split conformal (Tibshirani et al. 2019; common.weighted_conformal_q), one
    weighted quantile per target-test point including the test point's own weight
    (mass at +inf); infinite intervals are counted and reported, never dropped.
  * Diagnostics: normalised ESS/n of the calibration weights, share of total weight on
    the 10 largest calibration weights, domain-classifier AUC (source-cal vs
    target-test inputs; evaluation-only diagnostic, not used for any choice).
  * Support check: share of target records whose shift-axis value exceeds the maximum
    in the source pool (1.0 = disjoint support, likelihood ratio undefined).

Positive control (overlapping covariate shift with a KNOWN likelihood ratio): the whole
dataset is split (grouped) into a source half and a target half; the source half is
thinned with acceptance sigma(-lam (s - 0.5)) and the target half with
sigma(+lam (s - 0.5)), s = rank percentile of the shift axis (a feature). The true
ratio is then w*(x) ~ exp(lam s(x)) with full overlap. We compare unweighted, oracle
weighted, raw_LR weighted and rff_500 weighted conformal. This checks the
implementation and shows when weighted conformal does work.

Legacy reproduction: the submitted analysis (demo/exp5_singular_shift_diagnostic.py) is
re-run in its original form (one 70/30 split, classifier on source-cal vs ALL target
inputs, plug-in weighted quantile without the test weight, 8 RFF draws) and once more
with only its RFF bug fixed (the original draws a DIFFERENT random feature map for the
calibration and the target inputs because rff_embed is called twice on one generator).

Output: results/R9_legacy_wcp.json (merged into results/R9_legacy.json).
"""
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import (ALPHA, DEMO, RES, SEEDS, _cut, _perm_grouped, check_disjoint, conformal_q,
                    design_shift_datasets, design_split, dump, headline_m, iqr, make_model,
                    rf_mean_std, save_splits, summarize, weighted_conformal_q)

D_GRID = [5, 20, 50, 100, 250, 500]
LAMS = [3.0, 6.0, 10.0]          # positive-control tilt strengths (fixed a priori)
CLIP_Q = 0.99                    # weight-clipping arm: cap at q99 of the calibration weights
CLIP_D = [50, 500]
TAG = "R9_legacy_wcp"


# ------------------------------------------------------------------ helpers
def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def zfit(X):
    mu, sd = X.mean(0), X.std(0)
    return mu, np.where(sd < 1e-9, 1.0, sd)


def rff_map(D, sigma, p, rng):
    W = rng.normal(0.0, 1.0 / sigma, size=(p, D)); b = rng.uniform(0, 2 * np.pi, size=D)
    return lambda Z: np.sqrt(2.0 / D) * np.cos(Z @ W + b)


def median_dist(A, B, rng, nmax=1000):
    a = A[rng.choice(len(A), min(nmax, len(A)), replace=False)]
    b = B[rng.choice(len(B), min(nmax, len(B)), replace=False)]
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float(np.median(d)) + 1e-6


def ess_frac(w):
    return float(w.sum() ** 2 / (len(w) * np.sum(w ** 2)))


def top10_share(w):
    return float(np.sort(w)[::-1][:10].sum() / w.sum())


def wcp_eval(r_cal, w_cal, r_test, w_test, iqr_T, log=False):
    """Tibshirani weighted conformal, one quantile per test point (test weight included).
    With log=True the weights are log-weights; they are rescaled per test point by the
    common constant exp(-max) (the weighted quantile is invariant to a common scale)."""
    q = np.empty(len(w_test))
    for i, wt in enumerate(w_test):
        if log:
            M = max(w_cal.max(), wt)
            q[i] = weighted_conformal_q(r_cal, np.exp(w_cal - M), np.exp(wt - M), ALPHA)
        else:
            q[i] = weighted_conformal_q(r_cal, w_cal, wt, ALPHA)
    fin = np.isfinite(q)
    cov = float(np.mean(r_test <= q))
    return {"coverage": cov, "frac_infinite": float(1 - fin.mean()),
            "n_test": int(len(q)), "n_finite": int(fin.sum()),
            "coverage_finite_only": float(np.mean(r_test[fin] <= q[fin])) if fin.any() else None,
            "median_norm_width_finite": float(np.median(2 * q[fin]) / iqr_T) if fin.any() else None,
            "mean_norm_width_finite": float(np.mean(2 * q[fin]) / iqr_T) if fin.any() else None}


def weight_block(l_cal, l_test, r_cal, r_test, iqr_T, auc, clip_q=None):
    """l_* are classifier log-odds, log w(x) up to an additive constant.
    clip_q caps every log-weight (calibration and test) at the clip_q quantile of the
    CALIBRATION log-weights: the practitioner's usual fix for extreme likelihood ratios.
    It uses no target-test labels and no target-test inputs beyond the weights themselves."""
    cap_info = {}
    if clip_q is not None:
        cap = float(np.quantile(l_cal, clip_q))
        cap_info = {"clip_quantile_of_cal_log_weights": clip_q, "log_weight_cap": cap,
                    "frac_test_weights_clipped": float(np.mean(l_test > cap)),
                    "frac_cal_weights_clipped": float(np.mean(l_cal > cap))}
        l_cal = np.minimum(l_cal, cap); l_test = np.minimum(l_test, cap)
    out = wcp_eval(r_cal, l_cal, r_test, l_test, iqr_T, log=True)
    w = np.exp(l_cal - l_cal.max())
    out.update({"ess_frac": ess_frac(w), "top10_weight_share": top10_share(w),
                "frac_cal_prob_below_1e-6": float(np.mean(l_cal < np.log(1e-6))), "domain_auc": auc, **cap_info})
    return out


def fit_rf(X, y, seed):
    m = make_model("RF", seed); m.set_params(n_jobs=1)
    return m.fit(X, y)


# ------------------------------------------------------------------ main protocol arm
def one_split(ds, seed):
    sp = design_split(ds, seed)
    X, y = ds.X, ds.y
    m = fit_rf(X[sp["train"]], y[sp["train"]], seed)
    mu = {k: rf_mean_std(m, X[v])[0] for k, v in sp.items() if k != "train"}
    r_cal = np.abs(y[sp["cal"]] - mu["cal"]); r_T = np.abs(y[sp["ttest"]] - mu["ttest"])
    r_pool = np.abs(y[sp["rpool"]] - mu["rpool"])
    iqr_T = iqr(y[ds.tgt])
    q_src = conformal_q(r_cal, ALPHA); mh = headline_m(ds); q_m = conformal_q(r_pool[:mh], ALPHA)
    out = {"seed": seed, "sizes": {k: int(len(v)) for k, v in sp.items()},
           "vanilla": {"coverage": float(np.mean(r_T <= q_src)), "norm_width": float(2 * q_src / iqr_T)},
           "recal_m": {"coverage": float(np.mean(r_T <= q_m)), "norm_width": float(2 * q_m / iqr_T), "m": mh}}

    mu_x, sd_x = zfit(X[sp["train"]])
    Z = {k: (X[v] - mu_x) / sd_x for k, v in sp.items()}
    lab = np.r_[np.zeros(len(sp["train"])), np.ones(len(sp["rpool"]))]
    rng = np.random.default_rng([seed, 17])
    sigma = median_dist(Z["train"], Z["rpool"], rng)
    out["rbf_sigma"] = sigma
    auc_lab = np.r_[np.zeros(len(sp["cal"])), np.ones(len(sp["ttest"]))]
    variants = {}
    for D in D_GRID:
        phi = rff_map(D, sigma, X.shape[1], np.random.default_rng([seed, D, 7]))
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.vstack([phi(Z["train"]), phi(Z["rpool"])]), lab)
        e_cal = clf.decision_function(phi(Z["cal"])); e_T = clf.decision_function(phi(Z["ttest"]))
        auc = float(roc_auc_score(auc_lab, np.r_[e_cal, e_T]))
        variants[f"rff_{D}"] = weight_block(e_cal, e_T, r_cal, r_T, iqr_T, auc)
        if D in CLIP_D:
            variants[f"rff_{D}_clip{CLIP_Q:g}"] = weight_block(e_cal, e_T, r_cal, r_T, iqr_T, auc, clip_q=CLIP_Q)
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(np.vstack([Z["train"], Z["rpool"]]), lab)
    e_cal = clf.decision_function(Z["cal"]); e_T = clf.decision_function(Z["ttest"])
    auc = float(roc_auc_score(auc_lab, np.r_[e_cal, e_T]))
    variants["raw_LR"] = weight_block(e_cal, e_T, r_cal, r_T, iqr_T, auc)
    variants[f"raw_LR_clip{CLIP_Q:g}"] = weight_block(e_cal, e_T, r_cal, r_T, iqr_T, auc, clip_q=CLIP_Q)
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    hgb.fit(np.vstack([X[sp["train"]], X[sp["rpool"]]]), lab)
    e_cal = hgb.decision_function(X[sp["cal"]]); e_T = hgb.decision_function(X[sp["ttest"]])
    auc = float(roc_auc_score(auc_lab, np.r_[e_cal, e_T]))
    variants["raw_HGB"] = weight_block(e_cal, e_T, r_cal, r_T, iqr_T, auc)
    variants[f"raw_HGB_clip{CLIP_Q:g}"] = weight_block(e_cal, e_T, r_cal, r_T, iqr_T, auc, clip_q=CLIP_Q)
    out["weighted"] = variants
    return out


# ------------------------------------------------------------------ positive control
def tilt_split(ds, seed, lam=LAMS[0]):
    """Overlapping covariate shift with a known likelihood ratio (see module doc).
    Returns train / cal / stest (source, thinned by sigma(-lam(s-.5))), tunlab (target
    inputs for the weight model) and ttest (target test), all grouped and disjoint."""
    rng = np.random.default_rng([seed, 991, int(10 * lam)])
    s = rankdata(ds.shift) / len(ds.shift)
    order, gid = _perm_grouped(np.arange(len(ds.y)), ds.groups, rng)
    h = _cut(order, gid, 0.5)
    Sp, Tp = order[:h], order[h:]
    h2 = _cut(Tp, ds.groups[Tp], 0.5)
    TU, TT = Tp[:h2], Tp[h2:]
    Sacc = Sp[rng.random(len(Sp)) < expit(-lam * (s[Sp] - 0.5))]
    TUacc = TU[rng.random(len(TU)) < expit(lam * (s[TU] - 0.5))]
    TTacc = TT[rng.random(len(TT)) < expit(lam * (s[TT] - 0.5))]
    o, g = _perm_grouped(Sacc, ds.groups, rng)
    a = _cut(o, g, 0.6); b = _cut(o, g, 0.8)
    sp = {"train": o[:a], "cal": o[a:b], "stest": o[b:], "tunlab": TUacc, "ttest": TTacc}
    check_disjoint(sp, ds.groups)
    return sp


def one_tilt(ds, seed, lam):
    sp = tilt_split(ds, seed, lam)
    X, y = ds.X, ds.y
    s = rankdata(ds.shift) / len(ds.shift)
    m = fit_rf(X[sp["train"]], y[sp["train"]], seed)
    r_cal = np.abs(y[sp["cal"]] - m.predict(X[sp["cal"]])); r_T = np.abs(y[sp["ttest"]] - m.predict(X[sp["ttest"]]))
    iqr_T = iqr(y[sp["ttest"]])
    q_src = conformal_q(r_cal, ALPHA)
    out = {"seed": seed, "lam": lam, "sizes": {k: int(len(v)) for k, v in sp.items()},
           "unweighted": {"coverage": float(np.mean(r_T <= q_src)), "norm_width": float(2 * q_src / iqr_T)}}
    w_or_cal = np.exp(lam * s[sp["cal"]]); w_or_T = np.exp(lam * s[sp["ttest"]])
    o = wcp_eval(r_cal, w_or_cal, r_T, w_or_T, iqr_T)
    o.update({"ess_frac": ess_frac(w_or_cal), "top10_weight_share": top10_share(w_or_cal)})
    out["oracle"] = o
    mu_x, sd_x = zfit(X[sp["train"]])
    Z = {k: (X[v] - mu_x) / sd_x for k, v in sp.items()}
    lab = np.r_[np.zeros(len(sp["train"])), np.ones(len(sp["tunlab"]))]
    auc_lab = np.r_[np.zeros(len(sp["cal"])), np.ones(len(sp["ttest"]))]
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(np.vstack([Z["train"], Z["tunlab"]]), lab)
    e_cal = clf.decision_function(Z["cal"]); e_T = clf.decision_function(Z["ttest"])
    out["raw_LR"] = weight_block(e_cal, e_T, r_cal, r_T, iqr_T, float(roc_auc_score(auc_lab, np.r_[e_cal, e_T])))
    rng = np.random.default_rng([seed, 23])
    sigma = median_dist(Z["train"], Z["tunlab"], rng)
    phi = rff_map(500, sigma, X.shape[1], np.random.default_rng([seed, 500, 7]))
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.vstack([phi(Z["train"]), phi(Z["tunlab"])]), lab)
    e_cal = clf.decision_function(phi(Z["cal"])); e_T = clf.decision_function(phi(Z["ttest"]))
    out["rff_500"] = weight_block(e_cal, e_T, r_cal, r_T, iqr_T, float(roc_auc_score(auc_lab, np.r_[e_cal, e_T])))
    return out


# ------------------------------------------------------------------ legacy reproduction
def legacy_cases():
    """The three cases exactly as loaded in demo/exp5_singular_shift_diagnostic.py."""
    d = np.load(os.path.join(DEMO, "bioglass_tg.npz"))
    X, y, P = d["X"], d["y"], d["P"]
    pos = P[P > P.min()]
    glass = (X, y, np.where(P <= np.quantile(P, 0.70))[0], np.where(P >= np.quantile(pos, 0.75))[0])
    dss = {ds.name: ds for ds in design_shift_datasets() if ds.name in ("steel_yield", "polymer_Tg")}
    out = {"glass_Tg": glass}
    for k in ("steel_yield", "polymer_Tg"):
        ds = dss[k]; out[k] = (ds.X, ds.y, ds.src, ds.tgt)
    return out


def legacy_run(X, y, src, tgt, fix_rff=False, n_draws=8):
    def cq(r, a):
        n = len(r); return np.quantile(r, min(np.ceil((n + 1) * (1 - a)) / n, 1.0), method="higher")

    def wq(v, w, q):
        o = np.argsort(v); v, w = v[o], w[o]; cw = np.cumsum(w) / np.sum(w)
        return v[min(np.searchsorted(cw, q), len(v) - 1)]

    def rff_embed(Xz, D, sigma, rng):
        Wp = rng.normal(0, 1.0 / sigma, size=(Xz.shape[1], D)); b = rng.uniform(0, 2 * np.pi, size=D)
        return np.sqrt(2.0 / D) * np.cos(Xz @ Wp + b)

    rng0 = np.random.default_rng(0); perm = rng0.permutation(src)
    ntr = int(0.7 * len(perm)); tr, cal = perm[:ntr], perm[ntr:]
    m = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=4, random_state=0).fit(X[tr], y[tr])
    r_cal = np.abs(y[cal] - m.predict(X[cal])); r_tgt = np.abs(y[tgt] - m.predict(X[tgt]))
    mu_x, sd_x = X[tr].mean(0), X[tr].std(0); sd_x = np.where(sd_x < 1e-9, 1.0, sd_x)
    Zc, Zt = (X[cal] - mu_x) / sd_x, (X[tgt] - mu_x) / sd_x
    sigma = np.median(np.linalg.norm(Zc[:, None, :] - Zt[None, :, :], axis=-1)) + 1e-6
    vanilla = float(np.mean(r_tgt <= cq(r_cal, ALPHA)))
    nfs = int(np.clip(len(tgt) // 3, 10, 30)); fs = []
    for s in range(20):
        ti = np.random.default_rng(100 + s).permutation(len(tgt)); fsp, tte = ti[:nfs], ti[nfs:]
        if len(tte) < 8:
            continue
        fs.append(np.mean(r_tgt[tte] <= cq(r_tgt[fsp], ALPHA)))
    res = {"vanilla": vanilla, "fewshot": float(np.mean(fs)), "D": D_GRID, "ess": [], "wcov": [], "wcov_sd": []}
    for D in D_GRID:
        e_l, c_l = [], []
        for s in range(n_draws):
            rng = np.random.default_rng(1000 * D + s)
            if fix_rff:
                Wp = rng.normal(0, 1.0 / sigma, size=(X.shape[1], D)); b = rng.uniform(0, 2 * np.pi, size=D)
                pc = np.sqrt(2.0 / D) * np.cos(Zc @ Wp + b); pt = np.sqrt(2.0 / D) * np.cos(Zt @ Wp + b)
            else:   # original code path: two calls -> two different random maps
                pc = rff_embed(Zc, D, sigma, rng); pt = rff_embed(Zt, D, sigma, rng)
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(np.vstack([pc, pt]), np.r_[np.zeros(len(pc)), np.ones(len(pt))])
            e = np.clip(clf.predict_proba(pc)[:, 1], 1e-4, 1 - 1e-4); w = np.clip(e / (1 - e), 0, 1e6)
            e_l.append(ess_frac(w)); c_l.append(float(np.mean(r_tgt <= wq(r_cal, w, 1 - ALPHA))))
        res["ess"].append(float(np.mean(e_l))); res["wcov"].append(float(np.mean(c_l))); res["wcov_sd"].append(float(np.std(c_l)))
    return res


# ------------------------------------------------------------------ aggregation
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


def merge_parts():
    parts = {}
    for k in ("wcp", "local", "classif", "esol"):
        p = os.path.join(RES, f"R9_legacy_{k}.json")
        if os.path.exists(p):
            parts[k] = json.load(open(p))
    dump(parts, "R9_legacy.json")


def main(n_jobs=6):
    t0 = time.time()
    results = {"protocol": {"alpha": ALPHA, "seeds": SEEDS, "D_grid": D_GRID,
                            "weights": "exp(classifier log-odds), no clipping",
                            "clipped_arms": f"*_clip{CLIP_Q:g}: log-weights capped at the {CLIP_Q:g} quantile "
                                            f"of the calibration log-weights (arms: rff_{CLIP_D}, raw_LR, raw_HGB)",
                            "weight_model_training": "source-train inputs (0) vs target recal-pool inputs (1); "
                                                     "target-test never used", "tilt_lams": LAMS}}
    dsets = design_shift_datasets()
    for ds in dsets:
        rows = Parallel(n_jobs=n_jobs)(delayed(one_split)(ds, s) for s in SEEDS)
        S = {"dataset": ds.summary(), "m": headline_m(ds), "sizes_seed0": rows[0]["sizes"],
             "support": {"source_shift_max": float(ds.shift[ds.src].max()),
                         "target_shift_min": float(ds.shift[ds.tgt].min()),
                         "frac_target_beyond_source_max": float(np.mean(ds.shift[ds.tgt] > ds.shift[ds.src].max())),
                         "shift_axis_is_a_feature": True},
             "vanilla": {k: agg(rows, ["vanilla", k]) for k in ("coverage", "norm_width")},
             "recal_m": {k: agg(rows, ["recal_m", k]) for k in ("coverage", "norm_width")},
             "weighted": {}}
        for v in rows[0]["weighted"]:
            S["weighted"][v] = {k: agg(rows, ["weighted", v, k]) for k in rows[0]["weighted"][v]}
        S["per_split"] = [{"seed": r["seed"], "vanilla": r["vanilla"]["coverage"], "recal": r["recal_m"]["coverage"],
                           **{v: {"cov": r["weighted"][v]["coverage"], "ess": r["weighted"][v]["ess_frac"],
                                  "inf": r["weighted"][v]["frac_infinite"]} for v in r["weighted"]}} for r in rows]
        results[ds.name] = S
        wl = " ".join(f"{v}:{S['weighted'][v]['coverage']['mean']:.2f}/ess{S['weighted'][v]['ess_frac']['mean']:.3f}"
                      f"/inf{S['weighted'][v]['frac_infinite']['mean']:.2f}" for v in S["weighted"])
        print(f"{ds.name:12s} van {S['vanilla']['coverage']['mean']:.3f} recal {S['recal_m']['coverage']['mean']:.3f} | {wl}"
              f" | {time.time() - t0:.0f}s", flush=True)

    dump(results, f"{TAG}.json")        # checkpoint
    # positive control (overlapping, known likelihood ratio)
    results["positive_control"] = {}
    for ds in dsets:
        results["positive_control"][ds.name] = {}
        for lam in LAMS:
            save_splits(ds, split_fn=tilt_split, tag=f"{ds.name}_tilt{lam:g}", lam=lam)
            rows = Parallel(n_jobs=n_jobs)(delayed(one_tilt)(ds, s, lam) for s in SEEDS)
            S = {"sizes_seed0": rows[0]["sizes"]}
            for arm in ("unweighted", "oracle", "raw_LR", "rff_500"):
                S[arm] = {k: agg(rows, [arm, k]) for k in rows[0][arm]}
            results["positive_control"][ds.name][f"lam{lam:g}"] = S
            print(f"[tilt {lam:g}] {ds.name:12s} unw {S['unweighted']['coverage']['mean']:.3f} "
                  f"oracle {S['oracle']['coverage']['mean']:.3f} (inf {S['oracle']['frac_infinite']['mean']:.2f}, ess {S['oracle']['ess_frac']['mean']:.3f}) "
                  f"rawLR {S['raw_LR']['coverage']['mean']:.3f} (inf {S['raw_LR']['frac_infinite']['mean']:.2f}) "
                  f"rff500 {S['rff_500']['coverage']['mean']:.3f} | {time.time() - t0:.0f}s", flush=True)

    # legacy reproduction (original code path) and the same with only the RFF bug fixed
    results["legacy"] = {}
    for name, (X, y, src, tgt) in legacy_cases().items():
        results["legacy"][name] = {"original_code": legacy_run(X, y, src, tgt, fix_rff=False),
                                   "original_code_rff_bug_fixed": legacy_run(X, y, src, tgt, fix_rff=True)}
        a, b = results["legacy"][name]["original_code"], results["legacy"][name]["original_code_rff_bug_fixed"]
        print(f"[legacy] {name}: vanilla {a['vanilla']:.2f} fewshot {a['fewshot']:.2f} | ESS orig {np.round(a['ess'], 3)} "
              f"wcov {np.round(a['wcov'], 2)} | fixed ESS {np.round(b['ess'], 3)} wcov {np.round(b['wcov'], 2)}", flush=True)
    results["runtime_min"] = (time.time() - t0) / 60
    dump(results, f"{TAG}.json")
    merge_parts()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_parts()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
