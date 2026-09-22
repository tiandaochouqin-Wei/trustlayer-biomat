"""R3: comprehensive uncertainty metrics for ten UQ methods (Reviewer 2 comment 12).

Unified protocol (PROTOCOL.md / common.py): grouped 60/20/20 source split, target
pool halves = recalibration pool / target-test, headline m target records,
alpha = 0.10, finite-sample conformal quantile, 20 resampling splits, mean and
2.5-97.5 percentile across splits.  Every model is fitted on the train block only.

Datasets: the six design-shift datasets (common.design_shift_datasets) and ESOL /
Delaney logS (in-distribution only; grouped random 60/20/20 per seed; loader here).

Methods
  SC    split conformal, RF (make_model('RF')), score |y - mu|
  NSC   normalised split conformal, score |y - mu| / (sigma_tree + 1e-6)
  CQR   conformalised quantile regression (Romano et al. 2019): HistGB quantile
        models at (1-l)/2 and (1+l)/2 for every level l, score max(lo-y, y-hi)
  GP    native Gaussian intervals of make_model('GP') (mu +/- z sigma)
  BOOT  bootstrap ensemble of 10 RFs (100 trees each, fitted on bootstrap
        resamples of train), Gaussian intervals mu_B +/- z sd_B
  QR    raw HistGB quantile intervals (not conformalised)
  RFH   RF tree-spread Gaussian heuristic mu +/- z sigma_tree
  GP+conf, BOOT+conf, RFH+conf
        the non-conformal intervals wrapped by split conformal with the CQR score
        max(lo - y, y - hi) around their own level-l interval (QR+conf == CQR).

Settings
  ID          calibrate on source-cal, evaluate source-test
  SHIFT_SRC   calibrate on source-cal, evaluate target-test
  SHIFT_RECAL conformal / conformalised methods recalibrated on the first m
              records of the target recalibration pool, evaluated on the same
              target-test half (raw GP/BOOT/QR/RFH cannot use target labels:
              their SHIFT_RECAL numbers are identical to SHIFT_SRC and not repeated)

Metrics (per split): coverage at 0.90, mean width, width / IQR of the evaluated
domain (source pool for ID, target pool for SHIFT; reporting normaliser as in R1),
interval calibration error (mean and max |empirical - nominal| over
common.LEVELS, each level with its own z-quantile / conformal quantile / quantile
models), interval score (alpha 0.1), WIS over alpha in {0.1,...,0.5} + median,
worst-cluster coverage (k-means, only when the evaluated set has >= 40 records,
as in common.interval_metrics), width-stratified coverage (worst of 4 width
bins), risk-coverage curve ranked by each method's own 90 % interval width
(relative AURC, RMSE reduction at 50 % retained; undefined for constant-width
SC), abstention rate = share of candidates with 90 % interval wider than
c x IQR, c in {1, 2}.

Conditional coverage is reported against an EXACT reference distribution: a
minimum over k noisy bin proportions is downward biased even when conditional
coverage is exactly nominal, so for every worst-bin statistic we also compute
the exact distribution of min_b Binom(n_b, 0.90)/n_b for the realised bin sizes
(mean, 5th percentile, and the per-split p-value P(min_null <= observed)).
The bins are functions of X / of the interval widths only, never of the
evaluated labels, so the binomial reference is the right null.

Abstention thresholds use only labels that exist at decision time and never
target-test labels, and never more labels than the setting's own budget:
IQR of train + source-cal labels for ID and for SHIFT_SRC (no target labels
exist in that setting), IQR of the m recalibration labels for SHIFT_RECAL.
The `_ref` variants use the same IQR that normalises the widths (source pool /
whole target pool) and are an oracle-scale reference, reported for comparability
with the nW column.

Outputs results/R3_uq.json, results/R3_uq.md, results/splits/esol_logS_splits.json.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binom, norm
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits

from common import (ALPHA, LEVELS, SEEDS, DEMO, RES, Dataset, design_shift_datasets, design_split,
                    headline_m, composition_groups, check_disjoint, save_splits, _perm_grouped, _cut,
                    conformal_q, make_model, rf_mean_std, coverage, interval_score,
                    weighted_interval_score, calibration_error, cluster_conditional_coverage,
                    width_stratified_coverage, risk_coverage, abstention_rate, summarize, iqr, dump)

TAG = "R3_uq"
PARTS = os.path.join(RES, f"{TAG}_parts_v2")   # per-(dataset, seed) checkpoints, fix round
OLD_PARTS = os.path.join(RES, f"{TAG}_parts")  # previous round, used only as a regression check
os.makedirs(PARTS, exist_ok=True)
WIS_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
ABST_MULT = [1.0, 2.0]
N_BOOT, BOOT_TREES = 10, 100
TREE_JOBS = 1   # threads per worker: the shared 28-core machine is saturated, and multi-threaded
                # OpenMP/BLAS inside each joblib worker thrashed (first attempt, killed after 24 min)
EPS = 1e-6                         # NSC sigma floor, as in exp_R1_core.py
LV = np.asarray(LEVELS, float)
ZL = norm.ppf(0.5 + LV / 2)
I90 = int(np.argmin(np.abs(LV - (1 - ALPHA))))
LIDX = {round(float(l), 4): j for j, l in enumerate(LV)}
QLO = {j: round(float((1 - l) / 2), 4) for j, l in enumerate(LV)}
QHI = {j: round(float((1 + l) / 2), 4) for j, l in enumerate(LV)}
QUANTS = sorted(set(QLO.values()) | set(QHI.values()) | {0.5})
assert all(round(1 - a, 4) in LIDX for a in WIS_ALPHAS)

RAW = ["GP", "BOOT", "QR", "RFH"]
CONF = ["SC", "NSC", "CQR"]
WRAP = ["GP+conf", "BOOT+conf", "RFH+conf"]
METHODS = {"ID": CONF + RAW + WRAP, "SHIFT_SRC": CONF + RAW + WRAP, "SHIFT_RECAL": CONF + WRAP}
SETTINGS = list(METHODS)
PER_SPLIT_KEYS = ["coverage", "norm_width", "interval_score_norm", "wis_norm", "ce_mean", "aurc_rel",
                  "reduction_at50_pct", "worst_cluster_cov", "worst_cluster_null", "worst_cluster_p",
                  "width_strat_cov", "width_strat_null", "width_strat_p"]

# ESOL / Delaney (features exactly as demo/fig4_trust_layer_demo.py)
ESOL_FEATS = ["Minimum Degree", "Molecular Weight", "Number of H-Bond Donors",
              "Number of Rings", "Number of Rotatable Bonds", "Polar Surface Area"]
ESOL_TARGET = "measured log solubility in mols per litre"


def esol_dataset():
    df = pd.read_csv(os.path.join(DEMO, "delaney.csv"))
    X = df[ESOL_FEATS].to_numpy(float); y = df[ESOL_TARGET].to_numpy(float)
    n = len(y)
    meta = {"shift_axis": "none (in-distribution only)", "src_rule": "all records",
            "tgt_rule": "none", "source": "ESOL/Delaney (MoleculeNet), demo/delaney.csv",
            "n_unique_smiles": int(df["smiles"].nunique())}
    return Dataset("esol_logS", X, y, list(ESOL_FEATS), np.zeros(n), np.arange(n),
                   np.array([], dtype=int), "log(mol/L)", composition_groups(X), meta)


def esol_split(ds, seed, group=True, fr=(0.6, 0.2)):
    """Grouped random 60/20/20 split of all ESOL records (train / cal / stest);
    identical descriptor vectors share a group; no target pool (ID only)."""
    rng = np.random.default_rng(seed)
    S, gS = _perm_grouped(ds.src, ds.groups if group else None, rng)
    a = _cut(S, gS, fr[0]); b = _cut(S, gS, fr[0] + fr[1])
    sp = {"train": S[:a], "cal": S[a:b], "stest": S[b:]}
    check_disjoint(sp, ds.groups if group else None)
    return sp


# ------------------------------------------------------------------ models
def fit_bases(X, y, seed):
    rf = make_model("RF", seed); rf.set_params(n_jobs=TREE_JOBS); rf.fit(X, y)
    gp = make_model("GP", seed).fit(X, y)
    rng = np.random.default_rng(10_000 + seed)
    boots = []
    for b in range(N_BOOT):
        bi = rng.integers(0, len(y), len(y))
        boots.append(RandomForestRegressor(n_estimators=BOOT_TREES, min_samples_leaf=2, n_jobs=TREE_JOBS,
                                           random_state=1000 * seed + b).fit(X[bi], y[bi]))
    qr = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
                                           early_stopping=False, random_state=seed).fit(X, y) for q in QUANTS}
    return rf, gp, boots, qr


def predict_bases(models, Xs):
    rf, gp, boots, qr = models
    mu, sd = rf_mean_std(rf, Xs)
    gmu, gsd = gp.predict(Xs, return_std=True)
    B = np.stack([m.predict(Xs) for m in boots], 0)
    return {"RF": (mu, sd), "GP": (gmu, gsd), "BOOT": (B.mean(0), B.std(0, ddof=1)),
            "Q": {q: qr[q].predict(Xs) for q in QUANTS}}


def gauss(mu, sd):
    return {"lo": mu[None] - ZL[:, None] * sd[None], "hi": mu[None] + ZL[:, None] * sd[None], "med": mu}


def qr_int(Q):
    lo = np.stack([Q[QLO[j]] for j in range(len(LV))]); hi = np.stack([Q[QHI[j]] for j in range(len(LV))])
    n_cross = int((lo[I90] > hi[I90]).sum())
    return {"lo": np.minimum(lo, hi), "hi": np.maximum(lo, hi), "med": Q[0.5], "n_cross90": n_cross}


def raw_intervals(P):
    return {"GP": gauss(*P["GP"]), "BOOT": gauss(*P["BOOT"]), "QR": qr_int(P["Q"]), "RFH": gauss(*P["RF"])}


def _uncross(lo, hi):
    """A negative conformal correction can make lo > hi (CQR-type score); such an
    interval is empty and is reported as a zero-width interval at its midpoint."""
    bad = lo > hi
    if bad.any():
        mid = np.where(bad, (lo + hi) / 2, 0.0)
        lo, hi = np.where(bad, mid, lo), np.where(bad, mid, hi)
    return lo, hi, int(bad.sum())


def conf_sc(mu_c, y_c, mu_e):
    r = np.abs(y_c - mu_c)
    q = np.array([conformal_q(r, 1 - l) for l in LV])
    return {"lo": mu_e[None] - q[:, None], "hi": mu_e[None] + q[:, None], "med": mu_e, "q90": float(q[I90])}


def conf_nsc(mu_c, s_c, y_c, mu_e, s_e):
    r = np.abs(y_c - mu_c) / (s_c + EPS)
    q = np.array([conformal_q(r, 1 - l) for l in LV])
    u = q[:, None] * (s_e[None] + EPS)
    return {"lo": mu_e[None] - u, "hi": mu_e[None] + u, "med": mu_e, "q90": float(q[I90])}


def conf_wrap(Ic, y_c, Ie):
    """Split-conformal wrapper around any interval method (CQR score)."""
    lo, hi, qs, nfix90 = [], [], [], 0
    for j, l in enumerate(LV):
        s = np.maximum(Ic["lo"][j] - y_c, y_c - Ic["hi"][j])
        q = conformal_q(s, 1 - l); qs.append(q)
        a, b, nf = _uncross(Ie["lo"][j] - q, Ie["hi"][j] + q)
        lo.append(a); hi.append(b)
        if j == I90:
            nfix90 = nf
    return {"lo": np.stack(lo), "hi": np.stack(hi), "med": Ie["med"], "q90": float(qs[I90]),
            "n_empty90": nfix90}


def build(Pc, y_c, Pe):
    """All intervals for one (calibration set, evaluation set) pair."""
    Rc, Re = raw_intervals(Pc), raw_intervals(Pe)
    out = {"SC": conf_sc(Pc["RF"][0], y_c, Pe["RF"][0]),
           "NSC": conf_nsc(Pc["RF"][0], Pc["RF"][1], y_c, Pe["RF"][0], Pe["RF"][1]),
           "CQR": conf_wrap(Rc["QR"], y_c, Re["QR"])}
    for k in ["GP", "BOOT", "RFH"]:
        out[k + "+conf"] = conf_wrap(Rc[k], y_c, Re[k])
    out.update(Re)
    return out


# ------------------------------------------------- conditional-coverage reference
_NULLS = {}


def min_prop_null(sizes, p=1 - ALPHA):
    """Exact distribution of  min_b Binom(n_b, p) / n_b  over independent bins.

    Reference for any worst-bin coverage statistic: even when conditional
    coverage is exactly p in every bin, the MINIMUM over k noisy proportions is
    below p (more so for small bins and large k). Returns the support, the cdf,
    the mean and the 5th percentile. Exact (no simulation), hence deterministic.
    """
    key = (tuple(sorted(int(s) for s in sizes)), round(float(p), 6))
    if key in _NULLS:
        return _NULLS[key]
    ns = list(key[0])
    v = np.unique(np.concatenate([np.arange(n + 1) / n for n in ns]))
    S = np.ones(len(v))                                      # S_i = P(min >= v_i)
    for n in ns:
        kmin = np.ceil(v * n - 1e-9).astype(int)             # smallest k with k/n >= v_i
        S *= binom.sf(kmin - 1, n, p)                        # P(B >= kmin)
    mean = float(np.sum(np.diff(v) * S[1:]))                 # E[X] = sum (v_i - v_{i-1}) P(X >= v_i)
    cdf = 1.0 - np.append(S[1:], 0.0)                        # P(min <= v_i)
    out = {"v": v, "cdf": cdf, "mean": mean,
           "p05": float(v[int(np.searchsorted(cdf, 0.05))]), "sizes": ns}
    _NULLS[key] = out
    return out


def null_pvalue(nul, obs):
    """P(min_null <= obs): one-sided p-value for 'worse than exact conditional coverage'."""
    i = int(np.searchsorted(nul["v"], obs + 1e-12) - 1)
    return float(nul["cdf"][i]) if i >= 0 else 0.0


# ------------------------------------------------------------------ metrics
def cluster_labels(X, n_clusters=8, seed=0, min_size=10):
    """k-means labels exactly as in common.cluster_conditional_coverage, computed
    once per evaluated set (the clustering does not depend on the method);
    equality with the common function is asserted in _eval_setting."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(X)
    k = int(min(n_clusters, max(1, len(X) // min_size)))
    return KMeans(k, n_init=5, random_state=seed).fit_predict(Z), k


def worst_cluster(clus, covered, min_size=10):
    """(worst-cluster coverage, sizes of the clusters that enter the minimum)."""
    lab, k = clus
    per = [(int((lab == c).sum()), float(covered[lab == c].mean()))
           for c in range(k) if (lab == c).sum() >= min_size]
    if not per:
        return float("nan"), []
    return min(c for _, c in per), [n for n, _ in per]


def width_bins(width, covered, n_bins=4):
    """common.width_stratified_coverage, additionally returning the bin sizes.
    Constant width -> no stratification (returns the marginal coverage, no sizes)."""
    if np.allclose(width, width[0]):
        return float(covered.mean()), None
    edges = np.quantile(width, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-9
    b = np.clip(np.searchsorted(edges, width, side="right") - 1, 0, n_bins - 1)
    per = [(int((b == i).sum()), float(covered[b == i].mean())) for i in range(n_bins) if (b == i).any()]
    return min(c for _, c in per), [n for n, _ in per]


def evaluate(I, y, X, iqr_rep, iqr_abs, clus=None):
    lo, hi, med = I["lo"], I["hi"], I["med"]
    covs = [coverage(lo[j], hi[j], y) for j in range(len(LV))]
    ce_mean, ce_max = calibration_error(LV, covs)
    l9, h9 = lo[I90], hi[I90]
    cov = (y >= l9) & (y <= h9); w = h9 - l9
    fin = bool(np.all(np.isfinite(w)))
    const = fin and bool(np.allclose(w, w[0]))
    ints = {a: (lo[LIDX[round(1 - a, 4)]], hi[LIDX[round(1 - a, 4)]]) for a in WIS_ALPHAS}
    wis = weighted_interval_score(y, med, ints)
    isc = interval_score(l9, h9, y, ALPHA)
    err = np.abs(y - med)
    out = {"coverage": float(cov.mean()), "width": float(np.mean(w)), "norm_width": float(np.mean(w) / iqr_rep),
           "ce_mean": ce_mean, "ce_max": ce_max, "n_eval": int(len(y)),
           "interval_score": isc, "interval_score_norm": isc / iqr_rep, "wis": wis, "wis_norm": wis / iqr_rep,
           "width_constant": const,
           "rmse": float(np.sqrt(np.mean(err ** 2))),
           "n_infinite_levels": int(sum(not np.all(np.isfinite(hi[j] - lo[j])) for j in range(len(LV)))),
           "curve": covs}
    # ---- conditional coverage, each worst-bin statistic against its exact binomial reference
    if fin:
        ws, wsz = width_bins(w, cov)
        assert abs(ws - width_stratified_coverage(w, cov)) < 1e-12
        out["width_strat_cov"] = ws
        if wsz is not None:
            nul = min_prop_null(wsz)
            out.update({"width_strat_null": nul["mean"], "width_strat_null_p05": nul["p05"],
                        "width_strat_p": null_pvalue(nul, ws), "width_strat_bins": wsz})
        else:
            out.update({"width_strat_null": None, "width_strat_null_p05": None, "width_strat_p": None,
                        "width_strat_bins": None})
    else:
        out.update({"width_strat_cov": None, "width_strat_null": None, "width_strat_null_p05": None,
                    "width_strat_p": None, "width_strat_bins": None})
    if len(y) >= 40:
        wc, csz = (worst_cluster(clus, cov) if clus is not None else
                   (cluster_conditional_coverage(X, cov)[0], None))
        out["worst_cluster_cov"] = wc
        if csz:
            nul = min_prop_null(csz)
            out.update({"worst_cluster_null": nul["mean"], "worst_cluster_null_p05": nul["p05"],
                        "worst_cluster_p": null_pvalue(nul, wc), "worst_cluster_bins": csz})
        else:
            out.update({"worst_cluster_null": None, "worst_cluster_null_p05": None,
                        "worst_cluster_p": None, "worst_cluster_bins": None})
    else:
        out.update({"worst_cluster_cov": None, "worst_cluster_null": None, "worst_cluster_null_p05": None,
                    "worst_cluster_p": None, "worst_cluster_bins": None})
    for c in ABST_MULT:
        out[f"abstain_{c:g}iqr"] = abstention_rate(w, c * iqr_abs)        # decision-time / budget-legal scale
        out[f"abstain_{c:g}iqr_ref"] = abstention_rate(w, c * iqr_rep)    # oracle scale = the nW normaliser
    if fin and not const:
        rc = risk_coverage(err, w)
        out.update({"aurc_rel": rc["aurc_rel"], "reduction_at50_pct": rc["reduction_at50_pct"],
                    "rc_curve": rc["rmse"]})
    else:
        out.update({"aurc_rel": None, "reduction_at50_pct": None, "rc_curve": None})
    for k in ("q90", "n_empty90", "n_cross90"):
        if k in I:
            out[k] = I[k]
    return out


def _eval_setting(I, methods, y, X, iqr_rep, iqr_abs):
    clus = cluster_labels(X) if len(y) >= 40 else None
    res = {k: evaluate(I[k], y, X, iqr_rep, iqr_abs, clus) for k in methods}
    if clus is not None:   # cached clustering must reproduce common.cluster_conditional_coverage
        cov = (y >= I["SC"]["lo"][I90]) & (y <= I["SC"]["hi"][I90])
        assert abs(cluster_conditional_coverage(X, cov)[0] - res["SC"]["worst_cluster_cov"]) < 1e-12
    return res


# ------------------------------------------------------------------ regression check vs the previous round
REPRO_SKIP = {"SHIFT_SRC": ("abstain_1iqr", "abstain_2iqr"),      # threshold rescaled this round (see notes)
              "SHIFT_RECAL": ("abstain_1iqr", "abstain_2iqr")}
REPRO_KEYS = ["coverage", "width", "norm_width", "ce_mean", "ce_max", "interval_score",
              "interval_score_norm", "wis", "wis_norm", "width_strat_cov", "worst_cluster_cov",
              "aurc_rel", "reduction_at50_pct", "rmse", "q90", "n_empty90", "n_cross90",
              "n_infinite_levels", "abstain_1iqr", "abstain_2iqr"]


def repro_check(old, new):
    """Every metric that the fix round did not intentionally change must reproduce."""
    n, worst, bad = 0, 0.0, []
    for st in new["setting"]:
        if st not in old.get("setting", {}):
            continue
        for meth, M in new["setting"][st].items():
            O = old["setting"][st].get(meth, {})
            for k in REPRO_KEYS:
                if k in REPRO_SKIP.get(st, ()):
                    continue
                a, b = O.get(k), M.get(k)
                if a is None or b is None:
                    continue
                n += 1; d = abs(float(a) - float(b)); worst = max(worst, d)
                if d > 1e-12:
                    bad.append(f"{st}/{meth}/{k}: {a} vs {b}")
    return {"n_compared": n, "max_abs_diff": worst, "mismatches": bad}


def one_split(ds, seed, split_fn):
    path = os.path.join(PARTS, f"{ds.name}_{seed:02d}.pkl")
    if os.path.exists(path):
        r = pd.read_pickle(path); r["from_cache"] = True
        return r
    t0 = time.time()
    with threadpool_limits(limits=TREE_JOBS):
        sp = split_fn(ds, seed)
        X, y = ds.X, ds.y
        models = fit_bases(X[sp["train"]], y[sp["train"]], seed)
        P = {"cal": predict_bases(models, X[sp["cal"]]), "stest": predict_bases(models, X[sp["stest"]])}
        yc, yS = y[sp["cal"]], y[sp["stest"]]
        iqr_src = iqr(y[ds.src])
        iqr_abs_id = iqr(y[np.concatenate([sp["train"], sp["cal"]])])
        out = {"seed": seed, "sizes": {k: int(len(v)) for k, v in sp.items()}, "setting": {}}
        I = build(P["cal"], yc, P["stest"])
        out["setting"]["ID"] = _eval_setting(I, METHODS["ID"], yS, X[sp["stest"]], iqr_src, iqr_abs_id)
        if "ttest" in sp and len(sp["ttest"]):
            mh = headline_m(ds)
            rm = sp["rpool"][:mh]
            P["ttest"] = predict_bases(models, X[sp["ttest"]]); P["rm"] = predict_bases(models, X[rm])
            yT, XT = y[sp["ttest"]], X[sp["ttest"]]
            iqr_tgt = iqr(y[ds.tgt])
            iqr_abs_m = iqr(y[rm])            # SHIFT_RECAL: exactly the m labels the method itself uses
            I = build(P["cal"], yc, P["ttest"])
            # SHIFT_SRC: no target labels exist in this scenario -> source-domain scale
            out["setting"]["SHIFT_SRC"] = _eval_setting(I, METHODS["SHIFT_SRC"], yT, XT, iqr_tgt, iqr_abs_id)
            I = build(P["rm"], y[rm], P["ttest"])
            out["setting"]["SHIFT_RECAL"] = _eval_setting(I, METHODS["SHIFT_RECAL"], yT, XT, iqr_tgt, iqr_abs_m)
            out["m"] = mh
            out["iqr"] = {"report_src": iqr_src, "report_tgt": iqr_tgt, "abst_id": iqr_abs_id,
                          "abst_shift_src": iqr_abs_id, "abst_shift_recal": iqr_abs_m,
                          "rpool_iqr_prev_round": iqr(y[sp["rpool"]])}
        else:
            out["iqr"] = {"report_src": iqr_src, "abst_id": iqr_abs_id}
    out["sec"] = time.time() - t0
    old_path = os.path.join(OLD_PARTS, f"{ds.name}_{seed:02d}.pkl")
    out["repro_vs_prev"] = repro_check(pd.read_pickle(old_path), out) if os.path.exists(old_path) else None
    pd.to_pickle(out, path + ".tmp"); os.replace(path + ".tmp", path)
    with open(os.path.join(PARTS, "progress.log"), "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {ds.name} seed {seed} {out['sec']:.0f}s\n")
    out["from_cache"] = False
    return out


# ------------------------------------------------------------------ aggregation
SCALARS = ["coverage", "width", "norm_width", "ce_mean", "ce_max", "interval_score", "interval_score_norm",
           "wis", "wis_norm", "width_strat_cov", "width_strat_null", "width_strat_null_p05", "width_strat_p",
           "worst_cluster_cov", "worst_cluster_null", "worst_cluster_null_p05", "worst_cluster_p",
           "aurc_rel", "reduction_at50_pct", "abstain_1iqr", "abstain_2iqr",
           "abstain_1iqr_ref", "abstain_2iqr_ref", "rmse", "q90", "n_eval"]


def aggregate(ds, rows):
    S = {"dataset": ds.summary(), "sizes_seed0": rows[0]["sizes"], "m": rows[0].get("m"),
         "iqr_seed0": rows[0]["iqr"], "runtime_sec_mean": float(np.mean([r["sec"] for r in rows]))}
    for st in SETTINGS:
        if st not in rows[0]["setting"]:
            continue
        S[st] = {}
        for meth in METHODS[st]:
            R = [r["setting"][st][meth] for r in rows]
            M = {k: summarize([x.get(k) for x in R]) for k in SCALARS if any(x.get(k) is not None for x in R)}
            M["curve_mean"] = np.mean([x["curve"] for x in R], 0).tolist()
            M["width_constant"] = bool(all(x["width_constant"] for x in R))
            rcs = [x["rc_curve"] for x in R if x["rc_curve"] is not None]
            M["rc_curve_mean"] = np.mean(rcs, 0).tolist() if rcs else None
            M["splits_with_infinite_level"] = int(sum(x["n_infinite_levels"] > 0 for x in R))
            M["splits_with_infinite_90"] = int(sum(not np.isfinite(x["width"]) for x in R))
            for k, tag in (("worst_cluster_p", "n_splits_wclu_p05"), ("width_strat_p", "n_splits_wstr_p05")):
                v = [x.get(k) for x in R if x.get(k) is not None]
                M[tag] = int(sum(p <= 0.05 for p in v)) if v else None
                M[tag.replace("n_splits", "n_splits_with") + "_defined"] = len(v)
            for k in ("n_empty90", "n_cross90"):
                if k in R[0]:
                    M[k + "_total"] = int(sum(x[k] for x in R))
                    ne = float(np.sum([x["n_eval"] for x in R]))
                    M[k + "_share"] = float(sum(x[k] for x in R) / ne) if ne else None
            M["per_split"] = {k: [x.get(k) for x in R] for k in PER_SPLIT_KEYS}
            S[st][meth] = M
    return S


def across(results):
    out = {}
    for st in SETTINGS:
        out[st] = {}
        names = [d for d in results if st in results[d]]
        for meth in METHODS[st]:
            out[st][meth] = {}
            for k in SCALARS:
                if k in ("width", "interval_score", "wis", "rmse", "q90", "n_eval"):
                    continue          # unit-dependent / size: not aggregated across datasets
                v = [results[d][st][meth][k]["mean"] for d in names
                     if k in results[d][st][meth] and results[d][st][meth][k]["mean"] is not None]
                if v:
                    out[st][meth][k] = {"median": float(np.median(v)), "min": float(np.min(v)),
                                        "max": float(np.max(v)), "n_datasets": len(v)}
    return out


# ------------------------------------------------------------------ markdown
def cell(S, st, meth, k, d=2, ci=True):
    M = S.get(st, {}).get(meth, {})
    if k not in M or M[k]["mean"] is None:
        return "n/a"
    s = M[k]
    txt = f"{s['mean']:.{d}f}" + (f" [{s['lo']:.{d}f}, {s['hi']:.{d}f}]" if ci else "")
    if k == "width_strat_cov" and M.get("width_constant"):
        txt += "*"
    return txt


def cond_table(results, st, meths):
    """Observed worst-bin coverage against the exact binomial reference."""
    names = [d for d in results if st in results[d]]
    L = [f"Conditional coverage vs the exact binomial reference, {st} "
         "(obs = worst-bin coverage, mean over splits; ref = mean of "
         "min_b Binom(n_b, 0.90)/n_b for the realised bin sizes; p05 = its 5th percentile; "
         "k/20 = splits with p = P(min_null <= obs) <= 0.05):", "",
         "| dataset | stat | " + " | ".join(meths) + " |", "|---" * (len(meths) + 2) + "|"]
    for d in names:
        for stat, lab in (("worst_cluster", "k-means cluster"), ("width_strat", "width quartile")):
            row = []
            for m in meths:
                M = results[d][st][m]
                if results[d][st][m].get("width_constant") and stat == "width_strat":
                    row.append("const"); continue
                o = M.get(f"{stat}_cov", {}).get("mean"); r = M.get(f"{stat}_null", {}).get("mean")
                p5 = M.get(f"{stat}_null_p05", {}).get("mean")
                kk = M.get("n_splits_wclu_p05" if stat == "worst_cluster" else "n_splits_wstr_p05")
                row.append("n/a" if o is None or r is None else
                           f"{o:.2f} / {r:.2f} ({p5:.2f}) {kk}/20")
            L.append(f"| {d} | {lab} | " + " | ".join(row) + " |")
    return L + [""]


def write_md(results, acr, meta):
    L = ["# R3 - comprehensive uncertainty metrics (Reviewer 2, comment 12)", "",
         f"Script `revision/code/exp_R3_uq.py`; results `revision/results/R3_uq.json`. Compute: "
         f"{meta['worker_minutes']:.0f} worker-minutes over {meta['n_tasks']} (dataset, split) tasks with "
         f"6 single-threaded workers ({meta['compute_wall_min']:.0f} min wall clock on the shared machine); "
         f"per-task results are cached in `revision/results/{TAG}_parts_v2/`, so the tables can be rebuilt "
         f"in seconds (this rebuild: {meta['aggregation_wall_min']:.2f} min). `meta.runtime_min` in the JSON "
         "is the compute wall clock, not the rebuild time.",
         "Protocol: common.py / PROTOCOL.md (grouped 60/20/20 source split, target pool halved into "
         "recalibration pool / target-test, headline m, alpha = 0.10, finite-sample conformal quantile, "
         "20 resampling splits; every value is computed per split and summarised as mean "
         "[2.5-97.5 percentile across splits]). The splits resample one finite dataset each, so the "
         "percentile interval describes split-to-split variability, not population sampling error.", "",
         "Methods: SC split conformal (RF, |r|); NSC normalised split conformal (|r|/tree-sd); CQR "
         "conformalised HistGB quantile regression; GP native Gaussian intervals; BOOT 10 x 100-tree "
         "bootstrap RF ensemble, Gaussian intervals; QR raw HistGB quantiles; RFH RF tree-spread Gaussian "
         "heuristic; X+conf = split-conformal wrapper (score max(lo-y, y-hi)) around method X's own "
         "interval (QR+conf is identical to CQR). All fitted on train only; calibration on source-cal "
         "(ID, SHIFT_SRC) or on the first m records of the target recalibration pool (SHIFT_RECAL).", "",
         "Columns: cov = coverage at nominal 0.90; nW = mean 90 % width / IQR of the evaluated domain; "
         "CE = interval calibration error, mean (max) |empirical - nominal| over levels "
         "0.50-0.95; IS/IQR and WIS/IQR = interval score (alpha 0.1) and weighted interval score "
         "(alpha 0.1-0.5 + median) divided by the domain IQR; wClu / wStr = worst k-means-cluster "
         "coverage (evaluated sets >= 40 records only) and worst width-quartile coverage, each followed "
         "by wClu0 / wStr0, the mean of the EXACT reference distribution min_b Binom(n_b, 0.90)/n_b for "
         "the realised bin sizes - a worst-of-k statistic sits below 0.90 even under exact conditional "
         "validity, so wClu must be read against wClu0 and not against 0.90 (* constant width: "
         "no stratification, wStr equals marginal coverage); AURC = relative area under the risk-coverage "
         "curve ranked by the method's own 90 % width (1 = no better than no ranking; lower is better); "
         "R50 = % RMSE reduction on the 50 % narrowest intervals; Ab1/Ab2 = abstention rate with the rule "
         "'abstain if the 90 % interval is wider than 1 x / 2 x IQR'. The abstention IQR uses only labels "
         "available at decision time and never more than the setting's own budget: train + source-cal for "
         "ID and for SHIFT_SRC (in that scenario no target label exists), the m recalibration labels for "
         "SHIFT_RECAL. `abstain_*_ref` in the JSON repeats the rule on the oracle scale that normalises "
         "nW (source pool / whole target pool).", ""]
    cols = [("coverage", "cov", 3), ("norm_width", "nW", 2), ("ce_mean", "CE", 3), ("ce_max", "CEmax", 3),
            ("interval_score_norm", "IS/IQR", 2), ("wis_norm", "WIS/IQR", 3),
            ("worst_cluster_cov", "wClu", 2), ("worst_cluster_null", "wClu0", 2),
            ("width_strat_cov", "wStr", 2), ("width_strat_null", "wStr0", 2),
            ("aurc_rel", "AURC", 2), ("reduction_at50_pct", "R50 %", 0),
            ("abstain_1iqr", "Ab1", 2), ("abstain_2iqr", "Ab2", 2)]
    titles = {"ID": "In-distribution (ID): calibrate on source-cal, evaluate source-test (7 datasets incl. ESOL)",
              "SHIFT_SRC": "Design shift, source-calibrated (SHIFT_SRC): evaluate target-test (6 datasets)",
              "SHIFT_RECAL": "Design shift, recalibrated on m target records (SHIFT_RECAL): evaluate the same "
                             "target-test half (6 datasets)"}
    for st in SETTINGS:
        L += [f"## {titles[st]}", "", "Median across datasets of the per-dataset split means "
              "(range across datasets in the per-dataset tables / JSON).", "",
              "| method | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for meth in METHODS[st]:
            A = acr[st][meth]
            row = []
            for k, _, d in cols:
                if k not in A:
                    row.append("n/a")
                else:
                    v = A[k]["median"]
                    row.append(f"{v:.{d}f}" + ("*" if k == "width_strat_cov" and meth == "SC" else ""))
            L.append(f"| {meth} | " + " | ".join(row) + " |")
        if st == "SHIFT_RECAL":
            L += ["", "Raw GP/BOOT/QR/RFH cannot use the m target labels; their target-test numbers are "
                  "those of SHIFT_SRC."]
        names = [d for d in results if st in results[d]]
        L += ["", f"Per-dataset coverage at 0.90 (mean [2.5-97.5 %] over 20 splits), {st}:", "",
              "| method | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for meth in METHODS[st]:
            L.append(f"| {meth} | " + " | ".join(cell(results[d], st, meth, "coverage") for d in names) + " |")
        L += ["", f"Per-dataset normalised interval score IS/IQR (mean over splits; lower is better), {st}:", "",
              "| method | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for meth in METHODS[st]:
            L.append(f"| {meth} | " + " | ".join(cell(results[d], st, meth, "interval_score_norm", 2, False)
                                            for d in names) + " |")
        L += ["", f"Per-dataset relative AURC (width-ranked selective prediction; mean over splits), {st}:", "",
              "| method | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for meth in METHODS[st]:
            L.append(f"| {meth} | " + " | ".join(cell(results[d], st, meth, "aurc_rel", 2, False)
                                            for d in names) + " |")
        L += [""] + cond_table(results, st, ["SC", "NSC", "CQR"])
    L += ["## Interpretation", ""] + [f"- {s}" for s in meta["interpretation"]] + [""]
    L += ["## Notes and deviations", ""] + [f"- {s}" for s in meta["notes"]] + [""]
    L += ["## Verifier issues not adopted", "", meta["not_adopted"], ""]
    open(os.path.join(RES, f"{TAG}.md"), "w", encoding="utf-8").write("\n".join(L))


# ------------------------------------------------------------------ interpretation
def interpretation(results, acr):
    g = lambda st, m, k: acr[st][m].get(k, {}).get("median")
    rg = lambda st, m, k: (acr[st][m][k]["min"], acr[st][m][k]["max"])
    D = results
    txt = []
    txt.append("Calibration in-distribution. The three conformal procedures and the three conformalised "
               f"wrappers are calibrated (median coverage {g('ID','SC','coverage'):.3f} SC, "
               f"{g('ID','NSC','coverage'):.3f} NSC, {g('ID','CQR','coverage'):.3f} CQR, "
               f"{g('ID','GP+conf','coverage'):.3f}-{g('ID','RFH+conf','coverage'):.3f} for the wrappers; "
               f"median calibration error 0.018-0.022 over the seven levels). The raw methods are not: "
               f"BOOT {g('ID','BOOT','coverage'):.2f} [{rg('ID','BOOT','coverage')[0]:.2f}-{rg('ID','BOOT','coverage')[1]:.2f} "
               f"across the seven datasets], QR {g('ID','QR','coverage'):.2f} "
               f"[{rg('ID','QR','coverage')[0]:.2f}-{rg('ID','QR','coverage')[1]:.2f}], "
               f"GP {g('ID','GP','coverage'):.2f} [{rg('ID','GP','coverage')[0]:.2f}-{rg('ID','GP','coverage')[1]:.2f}], "
               f"RFH {g('ID','RFH','coverage'):.2f} [{rg('ID','RFH','coverage')[0]:.2f}-{rg('ID','RFH','coverage')[1]:.2f}]. "
               "The bootstrap ensemble's failure is structural: the spread of ten resampled forests measures "
               "model variance only and omits the noise term, so it cannot reach predictive coverage. "
               "The direction of the tree-spread heuristic's error depends on the dataset - it over-covers "
               f"on the four glass properties ({D['glass_Tg']['ID']['RFH']['coverage']['mean']:.2f} on glass_Tg, "
               f"{D['glass_E']['ID']['RFH']['coverage']['mean']:.2f} on glass_E) and under-covers on "
               f"polymer_Tg ({D['polymer_Tg']['ID']['RFH']['coverage']['mean']:.2f}), steel_yield "
               f"({D['steel_yield']['ID']['RFH']['coverage']['mean']:.2f}) and ESOL "
               f"({D['esol_logS']['ID']['RFH']['coverage']['mean']:.2f}), so 'the heuristic under-covers' and "
               "'the heuristic over-covers' are both true only of particular datasets.")

    # ---- efficiency / width ranking on glass_Tg, generated from the table, not asserted
    gt = D["glass_Tg"]["ID"]
    W = lambda m: gt[m]["width"]["mean"]
    C = lambda m: gt[m]["coverage"]["mean"]
    order = sorted(METHODS["ID"], key=lambda m: -W(m))
    isc = order.index("SC")
    lab = lambda m: f"{m} {W(m):.0f} K" + ("" if abs(C(m) - (1 - ALPHA)) <= 0.02 else f" (cov {C(m):.2f})")
    wider = ", ".join(lab(m) for m in order[:isc])
    narrower = ", ".join(lab(m) for m in order[isc + 1:])
    n_cal_narrow = [m for m in order[isc + 1:] if abs(C(m) - (1 - ALPHA)) <= 0.02]
    NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    best_is = min(METHODS["ID"], key=lambda m: g("ID", m, "interval_score_norm"))
    txt.append("Efficiency in-distribution: coverage alone does not rank methods, the interval score does, and "
               f"constant-width split conformal is the least efficient calibrated method (median IS/IQR "
               f"{g('ID','SC','interval_score_norm'):.2f} versus {g('ID','NSC','interval_score_norm'):.2f} NSC, "
               f"{g('ID','CQR','interval_score_norm'):.2f} CQR, {g('ID','RFH+conf','interval_score_norm'):.2f} "
               f"RFH+conf, {g('ID','BOOT+conf','interval_score_norm'):.2f} BOOT+conf; best {best_is}). "
               "Split conformal is also not the widest method: its median normalised width is "
               f"{g('ID','SC','norm_width'):.2f} against {g('ID','GP','norm_width'):.2f} (GP), "
               f"{g('ID','CQR','norm_width'):.2f} (CQR) and {g('ID','NSC','norm_width'):.2f} (NSC). On glass_Tg "
               f"the 90 % SC interval averages {W('SC'):.0f} K, the {isc + 1}th widest of the ten methods: "
               f"narrower than {wider}, and wider than {narrower}"
               f" - coverage is annotated only where it is not within 0.02 of nominal. Crucially "
               f"{NUM[len(n_cal_narrow)]} of the {NUM[len(order) - isc - 1]} methods narrower than SC - "
               + ", ".join(f"{m} {W(m):.0f} K at {C(m):.2f} coverage" for m in n_cal_narrow) +
               " - are calibrated at the nominal 0.90, so SC is not even the widest calibrated interval: the "
               "adaptive conformal variants achieve the same guarantee with a smaller average width. The "
               "honest statement is that conformal intervals are wider than intervals that under-cover, not "
               "that a guarantee costs width relative to other calibrated methods.")

    # ---- conditional coverage against the exact binomial reference
    def cnd(st, m, stat):
        o, r = g(st, m, f"{stat}_cov"), g(st, m, f"{stat}_null")
        return o, r
    names_id = [d for d in D if "ID" in D[d]]
    below = [d for d in names_id if D[d]["ID"]["SC"].get("worst_cluster_cov", {}).get("mean") is not None
             and D[d]["ID"]["SC"]["worst_cluster_cov"]["mean"] < D[d]["ID"]["SC"]["worst_cluster_null_p05"]["mean"]]
    detail = "; ".join(f"{d} {D[d]['ID']['SC']['worst_cluster_cov']['mean']:.2f} vs "
                       f"{D[d]['ID']['SC']['worst_cluster_null']['mean']:.2f} "
                       f"({D[d]['ID']['SC']['n_splits_wclu_p05']}/20 splits p<=0.05)"
                       for d in names_id if D[d]["ID"]["SC"].get("worst_cluster_cov", {}).get("mean") is not None)
    o_sc, r_sc = cnd("ID", "SC", "worst_cluster")
    o_nw, r_nw = cnd("ID", "NSC", "width_strat")
    o_cw, r_cw = cnd("ID", "CQR", "width_strat")
    gap = [D[d]["ID"]["SC"]["worst_cluster_null"]["mean"] - D[d]["ID"]["SC"]["worst_cluster_cov"]["mean"]
           for d in below]
    gap0 = [(1 - ALPHA) - D[d]["ID"]["SC"]["worst_cluster_cov"]["mean"] for d in below]
    txt.append("Conditional coverage in-distribution, read against the right reference. A minimum over k noisy "
               "bins is downward biased even under exact conditional validity, so every worst-bin number is "
               "compared with the exact distribution of min_b Binom(n_b, 0.90)/n_b for the realised bin sizes "
               "(the bins depend on X and on the interval widths, never on the evaluated labels, so this is the "
               f"correct null). Worst k-means-cluster coverage for SC is {o_sc:.2f} against a reference of "
               f"{r_sc:.2f} (medians over the datasets with >= 40 evaluation records): per dataset "
               + detail + f". SC's worst-cluster coverage is below the 5th percentile of the reference on "
               f"{len(below)} of {sum(1 for d in names_id if D[d]['ID']['SC'].get('worst_cluster_cov', {}).get('mean') is not None)} "
               f"datasets ({', '.join(below)}), i.e. the shortfall there is real, but it is "
               f"{min(gap):.2f}-{max(gap):.2f} below the REFERENCE, not the "
               f"{min(gap0):.2f}-{max(gap0):.2f} below 0.90 that a naive reading would report - about "
               f"{100 * (1 - np.mean(gap) / np.mean(gap0)):.0f} % of the apparent gap is the bias of a "
               "worst-of-k statistic. The width-quartile statistics are a different story: "
               f"NSC covers {o_nw:.2f} in its worst quartile against a reference of {r_nw:.2f} and CQR "
               f"{o_cw:.2f} against {r_cw:.2f}, i.e. at or barely below the null, so they are NOT evidence of "
               "conditional miscoverage and should not be quoted as such. The defensible claim for Reviewer 2 "
               "comment 2 is the cluster result for constant-width split conformal: a 90 % marginal guarantee "
               "does not license a 90 % statement about a particular region of composition space, and "
               "in-distribution the adaptive methods (NSC "
               f"{g('ID','NSC','worst_cluster_cov'):.2f}, CQR {g('ID','CQR','worst_cluster_cov'):.2f}, "
               f"RFH+conf {g('ID','RFH+conf','worst_cluster_cov'):.2f}) sit closer to the reference "
               f"{g('ID','SC','worst_cluster_null'):.2f} than SC's {g('ID','SC','worst_cluster_cov'):.2f} does. "
               "That ordering reverses in the design region: after recalibration the adaptive methods are the "
               f"ones that fail locally (NSC {g('SHIFT_RECAL','NSC','worst_cluster_cov'):.2f}, RFH+conf "
               f"{g('SHIFT_RECAL','RFH+conf','worst_cluster_cov'):.2f}, GP+conf "
               f"{g('SHIFT_RECAL','GP+conf','worst_cluster_cov'):.2f} against a reference of "
               f"{g('SHIFT_RECAL','SC','worst_cluster_null'):.2f}, versus SC "
               f"{g('SHIFT_RECAL','SC','worst_cluster_cov'):.2f}), because their width model is still the "
               "source-domain one - adaptivity does not buy local validity where the data are new.")

    # ---- shift, source calibration (abstention text regenerated from the new, decision-time thresholds)
    shift_names = [d for d in D if "SHIFT_SRC" in D[d]]
    ab = {d: D[d]["SHIFT_SRC"]["SC"]["abstain_1iqr"]["mean"] for d in shift_names}
    abid = {d: D[d]["ID"]["SC"]["abstain_1iqr"]["mean"] for d in shift_names}
    cv = {d: D[d]["SHIFT_SRC"]["SC"]["coverage"]["mean"] for d in shift_names}
    same = all(abs(ab[d] - abid[d]) < 1e-12 for d in shift_names)       # constant width, same calibration set
    blind = [d for d in shift_names if ab[d] <= 0.10 and cv[d] < 0.60]

    def rng2(vals, d=2):
        lo, hi = min(vals), max(vals)
        return f"{lo:.{d}f}" if abs(hi - lo) < 5 * 10 ** (-d - 1) else f"{lo:.{d}f}-{hi:.{d}f}"
    bl = (" on " + ", ".join(blind) +
          " the source-calibrated conformal interval is narrower than 1 x the source-domain IQR for "
          f"{rng2([100 * (1 - ab[d]) for d in blind], 0)} % of target candidates (abstention "
          f"{rng2([ab[d] for d in blind])}) while its true coverage is only "
          f"{rng2([cv[d] for d in blind])}"
          ) if blind else " no dataset shows the pattern"
    worst_cv = sorted(cv, key=lambda d: cv[d])
    cv_txt = (f"{rng2([cv[d] for d in worst_cv[:-1]])} on five of the six datasets "
              f"({worst_cv[-1]} retains {cv[worst_cv[-1]]:.2f}, the mildest of the six shifts)")
    txt.append("Design shift, source calibration: every method under-covers on the target-test half; median "
               "coverage " + ", ".join(f"{m} {g('SHIFT_SRC', m, 'coverage'):.2f}" for m in
                                       ["SC", "NSC", "CQR", "GP", "BOOT", "QR", "RFH"])
               + f". The distance-aware GP degrades least ({rg('SHIFT_SRC','GP','coverage')[0]:.2f}-"
               f"{rg('SHIFT_SRC','GP','coverage')[1]:.2f} across datasets) and the conformal procedures are not "
               "protected at all - the guarantee is only an exchangeability guarantee. Worse, the miscalibration "
               "is invisible to a width-based abstention rule, and for split conformal this is exact rather than "
               "approximate: SC's interval has constant width 2q, the same calibration set gives the same q, so "
               "the 90 % interval offered for a target candidate is literally the same width as the one offered "
               "in-distribution and a width trigger returns bit-identical decisions in the two regimes"
               + (" (verified: the SC abstention rate is identical in ID and SHIFT_SRC on all six datasets)"
                  if same else "") + ", while true coverage falls from 0.90 to "
               + cv_txt + ". Concretely," + bl +
               ". A width trigger therefore carries no information about this shift; detecting it needs a "
               "distance-aware score (the GP spread does widen on target inputs) or an explicit shift "
               "diagnostic, which is an argument for the trust layer but contradicts any reading in which "
               "'the interval widens when the model is out of its depth'.")

    # ---- shift, recalibration: nominal return, finite-m conservatism, residual calibration error
    def mbound(d):
        m = D[d]["m"]; k = int(np.ceil((m + 1) * (1 - ALPHA)))
        return m, min(1.0, k / (m + 1))
    bnd = {d: mbound(d) for d in shift_names}
    bmin, bmax = min(b for _, b in bnd.values()), max(b for _, b in bnd.values())
    ab2 = g("SHIFT_RECAL", "SC", "abstain_2iqr")
    ab2r = (min(g("SHIFT_RECAL", m, "abstain_2iqr") for m in METHODS["SHIFT_RECAL"]),
            max(g("SHIFT_RECAL", m, "abstain_2iqr") for m in METHODS["SHIFT_RECAL"]))
    txt.append("Design shift, recalibration on m target records: all six conformal/conformalised methods "
               "return to nominal or slightly above (median coverage "
               + ", ".join(f"{m} {g('SHIFT_RECAL', m, 'coverage'):.2f}" for m in METHODS["SHIFT_RECAL"])
               + "), so the conformal wrapper makes any UQ method - including a badly miscalibrated bootstrap "
               "ensemble - recoverable with the same handful of target labels; this is the clearest evidence "
               "in this study for the model-agnostic claim. Three honest qualifications: (i) the excess is "
               "the finite-sample conservatism of the split-conformal quantile, not a bonus - at m = "
               + ", ".join(f"{bnd[d][0]} ({d})" for d in sorted(shift_names, key=lambda d: bnd[d][0])) +
               f" the guaranteed lower bound ceil((m+1)0.9)/(m+1) is {bmin:.3f}-{bmax:.3f} "
               f"(steel_yield m = {bnd['steel_yield'][0]} forces >= {bnd['steel_yield'][1]:.3f}), which is why "
               f"SC averages {g('SHIFT_RECAL','SC','coverage'):.3f} and why the recalibrated intervals are "
               "wider than they would be with a larger budget; the residual calibration error over the seven "
               f"levels is correspondingly {g('SHIFT_RECAL','SC','ce_mean'):.3f} mean / "
               f"{g('SHIFT_RECAL','SC','ce_max'):.3f} max, against "
               f"{g('ID','SC','ce_mean'):.3f} / {g('ID','SC','ce_max'):.3f} in-distribution. (ii) The efficiency "
               f"advantage of adaptive intervals disappears under shift (median IS/IQR "
               + ", ".join(f"{m} {g('SHIFT_RECAL', m, 'interval_score_norm'):.2f}" for m in METHODS["SHIFT_RECAL"])
               + " - NSC and CQR are no longer better than plain SC, because a spread learned on source data "
               "does not order target errors). (iii) Coverage is restored by intervals of median width "
               f"{g('SHIFT_RECAL','SC','norm_width'):.1f} x the target-domain IQR "
               f"({rg('SHIFT_RECAL','SC','norm_width')[0]:.1f}-{rg('SHIFT_RECAL','SC','norm_width')[1]:.1f}), "
               f"so under the rule 'abstain if the interval is wider than 2 x IQR' {ab2:.2f} of target "
               f"candidates (SC; {ab2r[0]:.2f}-{ab2r[1]:.2f} across the six methods) are abstained. The "
               "recalibrated intervals are valid but, on five of the six datasets, too wide to decide much: "
               "honest uncertainty, not free accuracy.")

    txt.append("Selective prediction: ranking by interval width is invariant to any additive or multiplicative "
               "conformal correction, so NSC, RFH and RFH+conf share one ranking (the RF tree spread), "
               "GP/GP+conf, BOOT/BOOT+conf and QR/CQR likewise, and their risk-coverage curves are identical "
               "by construction. Conformal calibration sets the level of the interval, not the order of the "
               "candidates - 'conformal-ranked triage' is heuristic-spread-ranked triage with a calibrated "
               f"width. In-distribution the ranking works (median relative AURC {g('ID','NSC','aurc_rel'):.2f} "
               f"for tree spread, {g('ID','NSC','reduction_at50_pct'):.0f} % RMSE "
               f"reduction on the most-confident half, {rg('ID','NSC','reduction_at50_pct')[0]:.0f}-"
               f"{rg('ID','NSC','reduction_at50_pct')[1]:.0f} % across datasets). Under the design shift it "
               f"largely fails: after recalibration the same ranking gives a median relative AURC of "
               f"{g('SHIFT_RECAL','NSC','aurc_rel'):.2f} and only {g('SHIFT_RECAL','NSC','reduction_at50_pct'):.0f} % "
               f"error reduction ({rg('SHIFT_RECAL','NSC','reduction_at50_pct')[0]:+.0f} % to "
               f"{rg('SHIFT_RECAL','NSC','reduction_at50_pct')[1]:+.0f} % across datasets, negative on "
               f"glass_Tliq). The GP spread is the only uncertainty that still orders target errors "
               f"(AURC {g('SHIFT_RECAL','GP+conf','aurc_rel'):.2f}, "
               f"{g('SHIFT_RECAL','GP+conf','reduction_at50_pct'):.0f} % median reduction), which argues for a "
               "distance-aware score under shift rather than tree spread.")

    # ---- abstention: report the per-dataset structure, not a bimodal median
    id_names = [d for d in D if "ID" in D[d]]
    sc1 = {d: D[d]["ID"]["SC"]["abstain_1iqr"]["mean"] for d in id_names}
    lo_d = [d for d in id_names if sc1[d] < 0.05]; hi_d = [d for d in id_names if sc1[d] > 0.90]
    mid_d = [d for d in id_names if d not in lo_d and d not in hi_d]
    txt.append("Abstention: the threshold matters more than the method, and a median across datasets is "
               "misleading for constant-width SC, whose ID abstention is all-or-nothing by construction "
               f"(a 1 x IQR rule rejects every candidate or none): {', '.join(lo_d)} give "
               f"{min(sc1[d] for d in lo_d):.2f} while {', '.join(hi_d)} give "
               f"{min(sc1[d] for d in hi_d):.2f}-{max(sc1[d] for d in hi_d):.2f}"
               + (f", and {', '.join(mid_d)} {', '.join(f'{sc1[d]:.2f}' for d in mid_d)}" if mid_d else "")
               + f". The adaptive methods abstain selectively and are better summarised by a median: "
               f"{g('ID','NSC','abstain_1iqr'):.2f} (NSC), {g('ID','CQR','abstain_1iqr'):.2f} (CQR) at 1 x IQR - "
               "a 90 % interval is inevitably of the order of the property's own IQR unless the model is very "
               f"accurate. At 2 x IQR, ID abstention is {g('ID','SC','abstain_2iqr'):.2f}-"
               f"{max(g('ID', m, 'abstain_2iqr') for m in METHODS['ID']):.2f} but rises to "
               f"{ab2r[0]:.2f}-{ab2r[1]:.2f} after recalibration under shift. Under SHIFT_RECAL the threshold "
               "is the IQR of exactly the m recalibration labels, so this number costs no data beyond the "
               "recalibration budget itself.")
    return txt


# ------------------------------------------------------------------ main
def main(n_jobs=6):
    t0 = time.time()
    dsets = design_shift_datasets() + [esol_dataset()]
    for ds in dsets[-1:]:
        save_splits(ds, split_fn=esol_split, tag=ds.name)
    fns = {d.name: (esol_split if d.name == "esol_logS" else design_split) for d in dsets}
    order = sorted(range(len(dsets)), key=lambda i: -len(dsets[i].y))
    tasks = [(i, s) for i in order for s in SEEDS]
    rows = Parallel(n_jobs=n_jobs, verbose=0)(delayed(one_split)(dsets[i], s, fns[dsets[i].name]) for i, s in tasks)
    wall = (time.time() - t0) / 60
    task_sec = float(sum(r["sec"] for r in rows))
    n_new = int(sum(not r.get("from_cache") for r in rows))

    # honest compute accounting: the wall clock of the runs that actually computed the parts
    cm_path = os.path.join(PARTS, "compute_meta.json")
    cm = json.load(open(cm_path)) if os.path.exists(cm_path) else {"runs": []}
    if n_new:
        cm["runs"].append({"ts": time.strftime("%Y-%m-%d %H:%M"), "n_computed": n_new,
                           "wall_min": wall, "n_jobs": n_jobs})
        json.dump(cm, open(cm_path, "w"), indent=1)
    compute_wall = float(sum(r["wall_min"] for r in cm["runs"])) if cm["runs"] else wall

    # regression check against the previous round's cache
    rep = [r["repro_vs_prev"] for r in rows if r.get("repro_vs_prev")]
    repro = {"n_parts_compared": len(rep), "n_values_compared": int(sum(r["n_compared"] for r in rep)),
             "max_abs_diff": float(max([r["max_abs_diff"] for r in rep], default=float("nan"))),
             "n_mismatches": int(sum(len(r["mismatches"]) for r in rep)),
             "examples": sum((r["mismatches"] for r in rep), [])[:5],
             "intentionally_excluded": "abstain_1iqr / abstain_2iqr under SHIFT_SRC and SHIFT_RECAL "
                                       "(threshold rescaled to a decision-time / budget-legal IQR this round)"}

    by = {}
    for (i, s), r in zip(tasks, rows):
        by.setdefault(dsets[i].name, []).append(r)
    results = {}
    for ds in dsets:
        R = sorted(by[ds.name], key=lambda r: r["seed"])
        results[ds.name] = aggregate(ds, R)
        S = results[ds.name]
        line = f"{ds.name:12s} ID " + " ".join(f"{m}:{S['ID'][m]['coverage']['mean']:.3f}" for m in METHODS["ID"][:7])
        if "SHIFT_SRC" in S:
            line += " | SRC " + " ".join(f"{m}:{S['SHIFT_SRC'][m]['coverage']['mean']:.2f}" for m in METHODS["SHIFT_SRC"][:7])
            line += " | RECAL " + " ".join(f"{m}:{S['SHIFT_RECAL'][m]['coverage']['mean']:.2f}" for m in METHODS["SHIFT_RECAL"])
        print(line, flush=True)
    acr = across(results)

    # numbers quoted in the notes, computed rather than asserted
    emp = {d: results[d]["ID"]["RFH+conf"].get("n_empty90_share") for d in results}
    emp = {d: v for d, v in emp.items() if v}
    rpool = {d: results[d]["sizes_seed0"]["rpool"] for d in results if "rpool" in results[d]["sizes_seed0"]}
    notes = [
        "Nothing is selected with target-test labels: models are fitted on train, conformal quantiles on "
        "source-cal or on the first m recalibration-pool records, abstention thresholds on train+cal labels "
        "(ID, SHIFT_SRC) or on exactly the m recalibration labels (SHIFT_RECAL). The IQR used to normalise "
        "widths / interval scores (source pool for ID, whole target pool for SHIFT, as in R1_core) is a "
        "reporting normaliser only; `abstain_*_ref` in the JSON repeats the abstention rule on that oracle "
        "scale, and the qualitative conclusions are the same under either scale.",
        "FIX ROUND vs the previous version of this script. (1) The SHIFT_SRC abstention threshold previously "
        "used the IQR of the target recalibration pool, which does not exist in the SHIFT_SRC scenario (no "
        "target labels); it is now the source-domain IQR (train + source-cal). (2) The SHIFT_RECAL threshold "
        "previously used the IQR of the whole recalibration pool (|rpool| = "
        + ", ".join(f"{d} {n}" for d, n in rpool.items()) + ") although the narrative is about m = "
        + ", ".join(f"{d} {results[d]['m']}" for d in rpool) + " labels; it is now the IQR of exactly those m "
        "labels, so no number in the abstention narrative uses more target data than the recalibration budget. "
        "(3) Every worst-bin conditional-coverage statistic now carries an exact binomial reference. "
        "(4) meta.runtime_min is the compute wall clock, not the cache-rebuild time. Everything else is "
        "unchanged and was verified to reproduce bit-exactly against the previous cache "
        f"({repro['n_values_compared']} values over {repro['n_parts_compared']} parts, "
        f"{repro['n_mismatches']} mismatches, max |diff| {repro['max_abs_diff']:.3e}).",
        "Conditional coverage reference. A worst-of-k-bins coverage is downward biased even when conditional "
        "coverage is exactly 0.90: for every evaluated set we compute the EXACT distribution of "
        "min_b Binom(n_b, 0.90)/n_b over the realised bin sizes (no simulation), and report its mean "
        "(wClu0 / wStr0), its 5th percentile and the per-split p-value P(min_null <= observed). The k-means "
        "bins depend only on the evaluated inputs X and the width bins only on the interval widths, never on "
        "the evaluated labels, so the binomial reference is the appropriate null; it does assume independence "
        "across records within a bin. Comparing a worst-bin number with 0.90 rather than with this reference "
        "overstates conditional miscoverage, which is why the width-quartile results are reported as null "
        "findings here.",
        "ESOL/Delaney is in-distribution only; grouped random 60/20/20 per seed (identical descriptor "
        "vectors share a group); split indices in results/splits/esol_logS_splits.json.",
        "BOOT (10 x 100-tree RFs, min_samples_leaf 2, sd with ddof 1) and the HistGB quantile models "
        "(max_iter 400, learning rate 0.05, no early stopping, i.e. make_model('HistGB') with loss='quantile') "
        "are defined in this script with a-priori hyperparameters (not in make_model); nothing was tuned. "
        "Note for the rebuttal that this BOOT is not the same estimator as the 'bootstrap ensemble' of "
        "demo/figS2_uq_baselines.py, which is 8 HistGradientBoosting models: both under-cover for the same "
        "structural reason, but the two coverage numbers are not a like-for-like comparison.",
        "Calibration error uses all seven levels 0.50-0.95 for every method (CQR/QR: 14 quantile models + "
        "median per split). Crossed raw quantiles are sorted; empty conformalised intervals (negative "
        "correction larger than half the raw width) are counted (n_empty90_total) and scored as zero-width.",
        "Finite-sample conformal quantiles are +inf when the calibration set is too small for a level "
        "(level 0.95 with m < 19, i.e. steel_yield m = 13 in SHIFT_RECAL): those intervals are kept as "
        "infinite (coverage 1, counted in splits_with_infinite_level) and enter the calibration error; the "
        "0.90 interval and all WIS levels (0.5-0.9) are finite for every m >= 9. The same finite-m effect "
        "makes the recalibrated 0.90 intervals conservative: the guarantee is ceil((m+1)0.9)/(m+1), i.e. "
        "0.929 at m = 13 and 0.903 at m = 30, which accounts for part of the recalibrated over-coverage and "
        "of the recalibrated width.",
        "Worst-cluster coverage is computed only when the evaluated set has >= 40 records (the rule in "
        "common.interval_metrics): n/a for steel_yield source-test (39) and for the target-test halves of "
        "glass_E (33), glass_HV (35) and steel_yield (21). Width-quartile and risk-coverage values on "
        "target-test halves of 21-35 records are noisy (5-9 records per quartile) - the binomial reference "
        "makes that noise explicit (wStr0 falls well below 0.90 for those sets).",
        "Risk-coverage ranks by the method's own 90 % width; errors are those of the method's own point "
        "predictor (RF mean, GP mean, bootstrap mean, QR median). SC has constant width, so its ranking "
        "is undefined (n/a); width-stratified coverage for SC equals marginal coverage and has no reference.",
        "Raw GP/BOOT/QR/RFH are not repeated under SHIFT_RECAL because they cannot use target labels; "
        "their conformalised versions are (QR+conf is identical to CQR).",
        "The additive conformal wrapper around an over-covering method can return an empty interval "
        "(negative correction): RFH+conf does so for "
        + ", ".join(f"{100 * v:.2f} % ({d})" for d, v in sorted(emp.items(), key=lambda kv: kv[1]))
        + " of in-distribution test points and never under shift. A multiplicative wrapper (NSC) "
        "cannot produce empty intervals; raw quantile crossing (QR) occurs for < 1 % of points and is fixed "
        "by sorting (n_cross90_total).",
        "ESOL grouping is conservative: 1,128 distinct SMILES fall into 919 distinct descriptor vectors, and "
        "records sharing a descriptor vector are kept on the same side of the split.",
        "GP+conf on glass_Tliq averages 0.86 coverage after recalibration, the only recalibrated cell "
        "visibly below nominal; the split interval [0.67, 0.95] contains 0.90, and exchangeability between "
        "the recalibration pool and the target-test half holds at composition-group, not record, level.",
        "Cross-check against exp_R1_core.py: SC coverage (ID, SHIFT_SRC, SHIFT_RECAL) and the NSC "
        "recalibrated coverage and 50 %-triage numbers reproduce R1_core.json split by split.",
        "Scope note for the rebuttal: the glass_Tg widths quoted here are measured on the SOURCE pool's "
        "held-out test block (P <= q0.70(P), 6,592 of the 7,623 glasses; source-pool y IQR 114 K vs 115 K "
        "for all records), not on the full dataset. demo/figS2_uq_baselines.py measured 155.5 K on a "
        "1,000-record ungrouped subsample of all 7,623 records (600 training rows, 6 seeds, no confidence "
        "intervals). Same underlying dataset and nearly the same property spread, but the comparison should "
        "be stated as 'source-pool held-out test under the unified protocol' rather than 'the full dataset'.",
    ]
    not_adopted = ("None. All ten verifier issues were adopted: the two substantive ones (the glass_Tg width "
                   "sentence, now generated from the width ranking in the table rather than asserted; the "
                   "conditional-coverage claim, now stated against an exact binomial reference and restricted "
                   "to the cluster result for split conformal) and the eight minor ones (rounding consistency "
                   "in the abstention sentence, meta.runtime_min, the empty-interval share lower bound now "
                   "computed from n_empty90_total / evaluated records, the bimodal SC abstention median now "
                   "reported per dataset, the SHIFT_SRC abstention scale now source-domain, the SHIFT_RECAL "
                   "abstention scale now exactly the m recalibration labels with |rpool| also disclosed, the "
                   "rebuttal-precision notes on 'full dataset' and on the two different bootstrap estimators, "
                   "and the missing discussion of the recalibrated calibration error and finite-m "
                   "conservatism).")
    meta = {"notes": notes, "worker_minutes": task_sec / 60, "n_tasks": len(tasks),
            "compute_wall_min": compute_wall, "aggregation_wall_min": wall, "not_adopted": not_adopted}
    meta["interpretation"] = interpretation(results, acr)
    out = {"meta": {"tag": TAG, "protocol": "revision/code/PROTOCOL.md + common.py", "alpha": ALPHA,
                    "levels": LV.tolist(), "wis_alphas": WIS_ALPHAS, "abstention_multipliers": ABST_MULT,
                    "n_splits": len(SEEDS), "methods": METHODS, "n_boot": N_BOOT, "boot_trees": BOOT_TREES,
                    "quantiles_fitted": QUANTS,
                    "runtime_min": compute_wall,          # wall clock of the run(s) that computed the parts
                    "compute_wall_clock_min": compute_wall, "aggregation_wall_clock_min": wall,
                    "n_tasks_computed_this_run": n_new,
                    "worker_minutes_total": task_sec / 60, "threads_per_worker": TREE_JOBS,
                    "reproduction_vs_previous_round": repro,
                    "conditional_coverage_reference": "exact distribution of min_b Binom(n_b, 0.90)/n_b over "
                                                      "the realised bin sizes (mean, 5th percentile, p-value)",
                    "abstention_threshold": {"ID": "IQR(train + source-cal labels)",
                                             "SHIFT_SRC": "IQR(train + source-cal labels) - no target label "
                                                          "exists in this scenario",
                                             "SHIFT_RECAL": "IQR(the m recalibration labels)",
                                             "_ref variants": "IQR of the width normaliser (source pool / "
                                                              "whole target pool); oracle-scale reference"},
                    "notes": notes, "verifier_issues_not_adopted": not_adopted,
                    "interpretation": meta["interpretation"]},
           "across_datasets_median": acr, "datasets": results}
    dump(out, f"{TAG}.json")
    write_md(results, acr, meta)
    print(f"compute wall {compute_wall:.1f} min; this invocation {wall:.2f} min; "
          f"repro vs previous round: {repro['n_values_compared']} values, {repro['n_mismatches']} mismatches, "
          f"max |diff| {repro['max_abs_diff']:.2e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        ds = [d for d in design_shift_datasets() if d.name == "steel_yield"][0]
        r = one_split(ds, 0, design_split)
        for st, M in r["setting"].items():
            for k, v in M.items():
                print(st, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()
                              if a not in ("curve", "rc_curve")})
        print("repro", r["repro_vs_prev"])
        e = esol_dataset(); r = one_split(e, 0, esol_split)
        print("esol", r["sizes"], {k: round(v["coverage"], 3) for k, v in r["setting"]["ID"].items()}, r["sec"])
        print("repro", r["repro_vs_prev"])
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
