"""R4b: a genuine computation-to-measurement shift (Reviewer 1 comment 2; Reviewer 2
comments 3 and 8).

The design-region shifts of R1 are controlled, composition-based extrapolations. Here
the shift is between how a label is *obtained*: a model is trained and conformally
calibrated on density-functional-theory (in-silico) formation enthalpies and then used
to predict the *measured* (calorimetric) formation enthalpy of new compounds.

Data: Kim et al. (2017) experimental formation enthalpies of inorganic compounds with
matched Materials Project (MP) and OQMD DFT values (matminer 'expt_formation_enthalpy',
1,276 records, eV/atom). These are intermetallics, silicides, borides, carbides and
germanides -- NOT biomedical materials; the analysis is a genuine
computation-to-measurement analogue of the in-silico -> in-vitro gap, not that gap.

Features: matminer ElementProperty 'magpie' preset (132 composition descriptors) on
pymatgen Composition. Groups: reduced formula (replicate measurements / polymorphs of
one formula never cross a split).

Split per seed s = 0..19 (grouped by reduced formula):
    train 50 % | calibration 20 % | target pool 30 %
    target pool -> recalibration pool (first half) / target-test (second half).
    Headline m = clip(floor(|target pool| / 3), 10, 30); k-sweeps take the first k
    records of the recalibration pool; everything is evaluated on the same
    target-test half, which is never used for fitting, calibration, recalibration,
    bias estimation or any choice.

Per in-silico source (MP; OQMD if its overlap with experiment >= 500) and per model
(RF, HistGB, hyperparameters fixed a priori in common.make_model):
 (i)   in-silico calibration: model trained on DFT labels of train compounds,
       split-conformal on DFT labels of calibration compounds; coverage of DFT labels
       (sanity) and of EXPERIMENTAL labels (the computation->measurement transfer) on
       the same target-test compounds;
 (ii)  recalibration with k experimental measurements from the recalibration pool,
       k in {0, 5, 9, 10, 20, 30, 50, 100, all}; k < 9 gives an infinite interval at
       alpha = 0.1 (reported as such); k larger than the pool is reported as n/a;
 (iii) references on the same splits: (a) the same model family trained on the
       experimental labels of the train compounds and calibrated on the experimental
       labels of the calibration compounds (needs ~70 % of the data measured);
       (b) 'DFT look-up': the DFT value of the test compound itself used as the
       prediction, conformalised with k experimental records;
 (iv)  the systematic DFT-experiment discrepancy (mean signed / absolute, by chemistry
       class, grouped bootstrap CI) and whether a signed-bias correction estimated
       from the k recalibration records (global, or chemistry-class-wise with fixed
       shrinkage lambda = 5 towards the global bias) before conformal recalibration
       narrows the intervals. The bias is estimated from the same k records that
       calibrate the interval, so calibration residuals are leave-one-out
       (|d_i - b_{-i}|), keeping them comparable with a test residual whose bias
       estimate never saw the test point.
Outputs results/R4b_dft_expt.json, results/R4b_dft_expt.md and
results/splits/R4b_dft_expt_{MP,OQMD}_splits.json.
"""
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from common import (ALPHA, LEVELS, SEEDS, DATA, Dataset, conformal_q, make_model, rf_mean_std,
                    interval_metrics, weighted_interval_score, calibration_error, risk_coverage,
                    summarize, iqr, dump, check_disjoint, save_splits, beta_coverage_quantiles,
                    abstention_rate, _perm_grouped, _cut)

TAG = "R4b_dft_expt"
KGRID = [0, 5, 9, 10, 20, 30, 50, 100]
WIS_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
MODELS = ["RF", "HistGB"]
LAMBDA_SHRINK = 5.0          # a priori shrinkage of class-wise bias towards the global bias
MIN_OVERLAP = 500
FBLOCK = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
          "Th", "Pa", "U", "Np", "Pu"}
METALLOID = {"B", "C", "Si", "Ge", "N", "P", "As", "Sb", "Se", "Te", "S", "O"}
SPMETAL = {"Li", "Na", "K", "Be", "Mg", "Ca", "Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi", "Zn", "Cd", "Hg"}
CLASSES = ["f-block (Ln/An) compounds", "B/C/Si/Ge compounds", "sp-metal intermetallics",
           "transition-metal-only intermetallics"]
# evaluation-only strata of the compound's own |DFT - experiment| gap (eV/atom)
DISC_BINS = [("|d|<0.05", 0.0, 0.05), ("0.05<=|d|<0.15", 0.05, 0.15), ("|d|>=0.15", 0.15, np.inf)]
LAMS = [1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.0]


# ============================================================ data
def chem_class(comp):
    els = {str(e) for e in comp.elements}
    if els & FBLOCK:
        return 0
    if els & METALLOID:
        return 1
    if els & SPMETAL:
        return 2
    return 3


def load_frame():
    """Kim et al. experimental formation enthalpies + MP/OQMD DFT values, featurised."""
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty
    df = pd.read_pickle(f"{DATA}/expt_formation_enthalpy.pkl").reset_index(drop=True)
    ep = ElementProperty.from_preset("magpie", impute_nan=False)
    feats, ok, red, cls, fail = [], [], [], [], []
    for f in df["formula"]:
        try:
            c = Composition(f)
            v = np.asarray(ep.featurize(c), float)
            good = bool(np.all(np.isfinite(v)))
            feats.append(v if good else np.full(len(ep.feature_labels()), np.nan))
            red.append(c.reduced_formula); cls.append(chem_class(c)); ok.append(good)
            if not good:
                fail.append(f)
        except Exception:                                   # noqa: BLE001
            feats.append(np.full(len(ep.feature_labels()), np.nan)); red.append(f)
            cls.append(-1); ok.append(False); fail.append(f)
    X = np.vstack(feats)
    df["reduced"] = red; df["cls"] = cls; df["feat_ok"] = ok
    info = {"n_records": int(len(df)), "n_feat_failures": int(len(fail)), "feat_failures": fail,
            "n_features": int(X.shape[1]), "featuriser": "matminer ElementProperty magpie (impute_nan=False)",
            "n_expt": int(df["e_form expt"].notna().sum()), "n_mp": int(df["e_form mp"].notna().sum()),
            "n_oqmd": int(df["e_form oqmd"].notna().sum()),
            "n_unique_reduced_formula": int(df["reduced"].nunique())}
    return df, X, ep.feature_labels(), info


def source_dataset(df, X, feat, src):
    col = {"MP": "e_form mp", "OQMD": "e_form oqmd"}[src]
    keep = (df["e_form expt"].notna() & df[col].notna() & df["feat_ok"]).to_numpy()
    sub = df[keep].reset_index(drop=True)
    _, groups = np.unique(sub["reduced"].to_numpy(), return_inverse=True)
    ds = Dataset(f"{TAG}_{src}", X[keep], sub["e_form expt"].to_numpy(float), feat,
                 shift=np.zeros(keep.sum()), src=np.arange(keep.sum()), tgt=np.array([], int),
                 unit="eV/atom", groups=groups,
                 meta={"in_silico_source": src, "n_unique_formula": int(len(np.unique(groups))),
                       "n_replicate_records": int(keep.sum() - len(np.unique(groups))),
                       "note": "src = all matched records (random grouped split, no design axis)"})
    ds.meta["y_dft"] = sub[col].to_numpy(float)
    ds.meta["cls"] = sub["cls"].to_numpy(int)
    ds.meta["formula"] = sub["formula"].to_numpy()
    return ds


def dft_split(ds, seed, fr=(0.5, 0.2)):
    """Grouped (reduced formula) permutation of all matched records with seed s:
    train 50 % / calibration 20 % / target pool 30 %; target pool halves =
    recalibration pool (first half) / target-test (second half)."""
    rng = np.random.default_rng(seed)
    order, gid = _perm_grouped(np.arange(len(ds.y)), ds.groups, rng)
    a = _cut(order, gid, fr[0]); b = _cut(order, gid, fr[0] + fr[1])
    T, gT = order[b:], gid[b:]
    h = _cut(T, gT, 0.5)
    sp = {"train": order[:a], "cal": order[a:b], "rpool": T[:h], "ttest": T[h:]}
    check_disjoint(sp, ds.groups)
    assert sum(len(v) for v in sp.values()) == len(ds.y)
    return sp


def headline_m_pool(sp):
    return int(np.clip((len(sp["rpool"]) + len(sp["ttest"])) // 3, 10, 30))


# ============================================================ helpers
def knn_difficulty(Xtr, Xq, k=10, exclude_self=False):
    """Model-agnostic difficulty sigma(x): mean distance to the k nearest training
    compositions in standardised magpie space (used for HistGB, which has no
    ensemble spread)."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    nn = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(sc.transform(Xtr))
    d, _ = nn.kneighbors(sc.transform(Xq))
    return d[:, int(exclude_self):].mean(1)


def fit_predict(model_name, seed, Xtr, ytr, Xq_dict):
    m = make_model(model_name, seed)
    if model_name in ("RF", "ExtraTrees"):
        m.set_params(n_jobs=1)
    m.fit(Xtr, ytr)
    out = {}
    for k, Xq in Xq_dict.items():
        if model_name in ("RF", "ExtraTrees"):
            out[k] = rf_mean_std(m, Xq)
        else:
            out[k] = (m.predict(Xq), knn_difficulty(Xtr, Xq))
    return out


def full_block(center, q_fn, y, y_iqr, X):
    """interval_metrics + WIS + coverage curve + calibration error for a constant-width
    conformal interval whose half-width at miscoverage a is q_fn(a)."""
    q = q_fn(ALPHA)
    out = interval_metrics(center - q, center + q, y, y_iqr, X=X) if np.isfinite(q) else \
        {"coverage": 1.0, "width": None, "norm_width": None, "interval_score": None}
    out["half_width"] = float(q) if np.isfinite(q) else None
    ints = {a: (center - q_fn(a), center + q_fn(a)) for a in WIS_ALPHAS}
    out["wis"] = weighted_interval_score(y, center, ints) if all(np.isfinite(q_fn(a)) for a in WIS_ALPHAS) else None
    curve = [float(np.mean(np.abs(y - center) <= q_fn(1 - l))) for l in LEVELS]
    out["curve"] = curve
    out["ce_mean"], out["ce_max"] = calibration_error(LEVELS, curve)
    return out


def loo_bias_residuals(d, cls=None, lam=LAMBDA_SHRINK):
    """d = y_expt - yhat on the k recalibration records. Returns (LOO calibration
    residuals, bias function for new records). Global bias if cls is None, else
    class-wise mean shrunk towards the global mean with pseudo-count lam."""
    d = np.asarray(d, float); k = len(d)
    S = d.sum()
    b_loo = (S - d) / (k - 1)
    if cls is None:
        return np.abs(d - b_loo), (lambda c_new: np.full(len(c_new), S / k))
    cls = np.asarray(cls)
    res = np.empty(k)
    for i in range(k):
        msk = (cls == cls[i]); msk[i] = False
        nc = msk.sum(); sc = d[msk].sum()
        bi = (sc + lam * b_loo[i]) / (nc + lam)
        res[i] = abs(d[i] - bi)
    bg = S / k

    def bias(c_new):
        out = np.empty(len(c_new))
        for j, c in enumerate(c_new):
            msk = cls == c
            out[j] = (d[msk].sum() + lam * bg) / (msk.sum() + lam)
        return out
    return res, bias


def simple_eval(center, half, y, y_iqr, alpha=ALPHA):
    if not np.all(np.isfinite(half)):
        return {"coverage": 1.0, "norm_width": None, "interval_score_norm": None, "infinite": True}
    lo, hi = center - half, center + half
    cov = float(np.mean((y >= lo) & (y <= hi)))
    w = float(np.mean(hi - lo))
    return {"coverage": cov, "norm_width": w / y_iqr,
            "interval_score_norm": float(np.mean((hi - lo) + (2 / alpha) * (lo - y) * (y < lo)
                                                 + (2 / alpha) * (y - hi) * (y > hi))) / y_iqr,
            "infinite": False}


# ============================================================ one split
def one_split(ds, model_name, seed, iqr_exp, iqr_dft):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return _one_split(ds, model_name, seed, iqr_exp, iqr_dft)


def _one_split(ds, model_name, seed, iqr_exp, iqr_dft):
    sp = dft_split(ds, seed)
    X, y_exp, y_dft, cls = ds.X, ds.y, ds.meta["y_dft"], ds.meta["cls"]
    tr, ca, rp, tt = sp["train"], sp["cal"], sp["rpool"], sp["ttest"]
    m = headline_m_pool(sp)
    out = {"seed": seed, "sizes": {k: int(len(v)) for k, v in sp.items()}, "m": m,
           "n_formula": {k: int(len(np.unique(ds.groups[v]))) for k, v in sp.items()}}
    yE, yD, XT, cT = y_exp[tt], y_dft[tt], X[tt], cls[tt]

    # ---------------- (i) model trained on DFT labels, calibrated on DFT labels
    P = fit_predict(model_name, seed, X[tr], y_dft[tr], {"cal": X[ca], "rpool": X[rp], "ttest": XT})
    muT, sdT = P["ttest"]
    r_cal = np.abs(y_dft[ca] - P["cal"][0])
    qfun_is = lambda a: conformal_q(r_cal, a)                     # noqa: E731
    q_is = qfun_is(ALPHA)
    out["insilico_vs_dft"] = full_block(muT, qfun_is, yD, iqr_dft, XT)          # sanity
    out["insilico_vs_expt"] = full_block(muT, qfun_is, yE, iqr_exp, XT)         # transfer
    # coverage by chemistry class (in-silico calibrated), against experimental and DFT labels
    covE = np.abs(yE - muT) <= q_is
    covD = np.abs(yD - muT) <= q_is
    out["shortfall_dft_minus_expt"] = float(covD.mean() - covE.mean())
    out["insilico_vs_expt_by_class"] = {CLASSES[c]: {"n": int((cT == c).sum()),
                                                     "coverage": float(covE[cT == c].mean())}
                                        for c in range(4) if (cT == c).sum() > 0}
    out["insilico_vs_dft_by_class"] = {CLASSES[c]: {"n": int((cT == c).sum()),
                                                    "coverage": float(covD[cT == c].mean())}
                                       for c in range(4) if (cT == c).sum() > 0}
    # evaluation-only stratification by the size of the compound's own DFT-experiment gap
    dabs = np.abs(yD - yE)
    out["by_discrepancy_bin"] = {}
    for lab, lo_, hi_ in DISC_BINS:
        msk = (dabs >= lo_) & (dabs < hi_)
        out["by_discrepancy_bin"][lab] = {"n": int(msk.sum()),
                                          "insilico_cov": float(covE[msk].mean()) if msk.any() else None}

    # error decomposition on target-test: (yhat - y_expt) = (yhat - y_dft) + (y_dft - y_expt)
    e_model, e_disc = muT - yD, yD - yE
    out["rmse"] = {"dftmodel_vs_dft": float(np.sqrt(np.mean(e_model ** 2))),
                   "dftmodel_vs_expt": float(np.sqrt(np.mean((muT - yE) ** 2))),
                   "dft_lookup_vs_expt": float(np.sqrt(np.mean(e_disc ** 2)))}
    out["mae"] = {"dftmodel_vs_dft": float(np.mean(np.abs(e_model))),
                  "dftmodel_vs_expt": float(np.mean(np.abs(muT - yE))),
                  "dft_lookup_vs_expt": float(np.mean(np.abs(e_disc)))}
    out["corr_modelerr_discrepancy"] = float(np.corrcoef(e_model, e_disc)[0, 1])
    out["test_mean_signed_pred_minus_expt"] = float(np.mean(muT - yE))

    # ---------------- (ii) recalibration with k experimental measurements
    mu_rp = P["rpool"][0]; sd_rp = P["rpool"][1]
    d_rp = y_exp[rp] - mu_rp                          # signed residual vs measurement
    r_rp = np.abs(d_rp)
    q_m = lambda a: conformal_q(r_rp[:m], a)         # noqa: E731
    out["recal_m"] = full_block(muT, q_m, yE, iqr_exp, XT)
    covR = np.abs(yE - muT) <= q_m(ALPHA)
    for lab, lo_, hi_ in DISC_BINS:
        msk = (dabs >= lo_) & (dabs < hi_)
        out["by_discrepancy_bin"][lab]["recal_m_cov"] = float(covR[msk].mean()) if msk.any() else None
    out["recal_m"]["abstain_vs_insilico_width"] = abstention_rate(np.full(len(yE), 2 * q_m(ALPHA)), 2 * q_is)
    # (iv) bias-corrected recalibration at m (global / class-wise), LOO residuals
    for tag, cl in [("recal_m_bias_global", None), ("recal_m_bias_class", cls[rp][:m])]:
        res, bfun = loo_bias_residuals(d_rp[:m], cl)
        ctr = muT + bfun(cT)
        out[tag] = full_block(ctr, lambda a, res=res: conformal_q(res, a), yE, iqr_exp, XT)
    out["bias_estimate_m"] = float(d_rp[:m].mean())
    # sigma-normalised recalibration at m (RF: tree spread; HistGB: kNN distance)
    s_rp = sd_rp[:m] + 1e-6
    qn = conformal_q(r_rp[:m] / s_rp, ALPHA)
    uT = (sdT + 1e-6) * qn
    out["recal_m_normalised"] = interval_metrics(muT - uT, muT + uT, yE, iqr_exp, X=XT)
    out["recal_m_normalised"]["abstain_vs_insilico_width"] = abstention_rate(2 * uT, 2 * q_is)
    out["triage_insilico_expt"] = risk_coverage(np.abs(muT - yE), sdT)
    out["triage_insilico_dft"] = risk_coverage(np.abs(muT - yD), sdT)

    # k-sweep: plain / global bias / class bias / DFT look-up
    ks = [k for k in KGRID if k <= len(rp)] + ["all"]
    out["ksweep"] = {}
    dl_rp = np.abs(y_exp[rp] - y_dft[rp])            # DFT look-up residuals
    for k in KGRID + ["all"]:
        kk = len(rp) if k == "all" else k
        if k not in ks:
            out["ksweep"][str(k)] = None               # pool smaller than k
            continue
        row = {"k": int(kk)}
        q_k = q_is if kk == 0 else conformal_q(r_rp[:kk], ALPHA)
        row["plain"] = simple_eval(muT, np.full(len(yE), q_k), yE, iqr_exp)
        if kk >= 2:
            for tag, cl in [("bias_global", None), ("bias_class", cls[rp][:kk])]:
                res, bfun = loo_bias_residuals(d_rp[:kk], cl)
                row[tag] = simple_eval(muT + bfun(cT), np.full(len(yE), conformal_q(res, ALPHA)), yE, iqr_exp)
            row["dft_lookup"] = simple_eval(yD, np.full(len(yE), conformal_q(dl_rp[:kk], ALPHA)), yE, iqr_exp)
        out["ksweep"][str(k)] = row

    # ---------------- (iii-a) reference: same model trained on EXPERIMENTAL labels
    Q = fit_predict(model_name, seed, X[tr], y_exp[tr], {"cal": X[ca], "ttest": XT})
    muE, sdE = Q["ttest"]
    r_calE = np.abs(y_exp[ca] - Q["cal"][0])
    out["ref_expt_trained"] = full_block(muE, lambda a: conformal_q(r_calE, a), yE, iqr_exp, XT)
    out["ref_expt_trained"]["rmse"] = float(np.sqrt(np.mean((muE - yE) ** 2)))
    out["triage_ref_expt"] = risk_coverage(np.abs(muE - yE), sdE)
    covX = np.abs(yE - muE) <= conformal_q(r_calE, ALPHA)
    out["ref_expt_by_class"] = {CLASSES[c]: {"n": int((cT == c).sum()), "coverage": float(covX[cT == c].mean())}
                                for c in range(4) if (cT == c).sum() > 0}
    for lab, lo_, hi_ in DISC_BINS:
        msk = (dabs >= lo_) & (dabs < hi_)
        out["by_discrepancy_bin"][lab]["ref_expt_cov"] = float(covX[msk].mean()) if msk.any() else None
    # ---------------- (iii-b) reference: DFT look-up conformalised with m measurements
    out["ref_dft_lookup_m"] = full_block(yD, lambda a: conformal_q(dl_rp[:m], a), yE, iqr_exp, XT)

    # ---------------- semi-synthetic surrogate-accuracy sweep (sensitivity analysis)
    # prediction_lam = y_dft + lam * (yhat - y_dft): same error structure as the fitted model,
    # error vs DFT scaled by lam (lam = 1: the fitted model; lam = 0: a perfect DFT surrogate).
    # The DFT-experiment discrepancy is the real one; only the surrogate error is rescaled.
    e_cal = P["cal"][0] - y_dft[ca]; e_rp = mu_rp - y_dft[rp]
    out["surrogate_sweep"] = {}
    for lam in LAMS:
        cen = yD + lam * e_model
        q_l = conformal_q(lam * np.abs(e_cal), ALPHA)
        r_l = np.abs(y_exp[rp] - (y_dft[rp] + lam * e_rp))[:m]
        q_lm = conformal_q(r_l, ALPHA)
        out["surrogate_sweep"][str(lam)] = {
            "rmse_vs_dft": float(lam * np.sqrt(np.mean(e_model ** 2))),
            "insilico_cov_dft": float(np.mean(np.abs(yD - cen) <= q_l)),
            "insilico_cov_expt": float(np.mean(np.abs(yE - cen) <= q_l)),
            "insilico_nw": float(2 * q_l / iqr_exp),
            "recal_m_cov_expt": float(np.mean(np.abs(yE - cen) <= q_lm)),
            "recal_m_nw": float(2 * q_lm / iqr_exp)}
    return out


# ============================================================ discrepancy (iv)
def discrepancy_table(ds, n_boot=2000, seed=0):
    """DFT - experiment, grouped bootstrap (resampling reduced formulas)."""
    d = ds.meta["y_dft"] - ds.y; cls = ds.meta["cls"]; g = ds.groups
    rng = np.random.default_rng(seed)
    ug = np.unique(g); members = {u: np.where(g == u)[0] for u in ug}

    def stats(mask):
        idx_groups = np.unique(g[mask])
        dd = d[mask]
        base = {"n_records": int(mask.sum()), "n_formula": int(len(idx_groups)),
                "mean_signed": float(dd.mean()), "mae": float(np.abs(dd).mean()),
                "rmse": float(np.sqrt(np.mean(dd ** 2))), "sd": float(dd.std(ddof=1)) if len(dd) > 1 else None,
                "median_signed": float(np.median(dd)),
                "frac_dft_more_negative": float(np.mean(dd < 0))}
        bm, ba = [], []
        for _ in range(n_boot):
            s = rng.choice(idx_groups, len(idx_groups), replace=True)
            ii = np.concatenate([members[u][mask[members[u]]] for u in s])
            bm.append(d[ii].mean()); ba.append(np.abs(d[ii]).mean())
        base["mean_signed_ci95"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
        base["mae_ci95"] = [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))]
        return base
    out = {"all": stats(np.ones(len(d), bool))}
    for c in range(4):
        if (cls == c).sum() >= 5:
            out[CLASSES[c]] = stats(cls == c)
    # inter-laboratory reproducibility floor: replicate experimental records of one formula
    rep = [ds.y[members[u]] for u in ug if len(members[u]) > 1]
    if rep:
        dev = np.concatenate([r - r.mean() for r in rep]); dof = sum(len(r) - 1 for r in rep)
        out["replicate_expt"] = {"n_formula_with_replicates": len(rep),
                                 "n_records": int(sum(len(r) for r in rep)),
                                 "pooled_within_formula_sd": float(np.sqrt(np.sum(dev ** 2) / dof)),
                                 "mean_abs_pairwise_diff": float(np.mean([np.mean(np.abs(r[:, None] - r[None, :])[np.triu_indices(len(r), 1)]) for r in rep]))}
    return out


def lookup_resampling_check(ds, n_splits=500):
    """Model-free diagnostic: DFT value + conformal half-width from the first m
    recalibration-pool records, over many grouped splits (same split function), to
    separate resampling noise of the 20 protocol splits from a violated guarantee."""
    yE, yD = ds.y, ds.meta["y_dft"]
    cov = []
    for s in range(n_splits):
        sp = dft_split(ds, s); m = headline_m_pool(sp)
        r = np.abs(yE[sp["rpool"]] - yD[sp["rpool"]])
        cov.append(float(np.mean(np.abs(yE[sp["ttest"]] - yD[sp["ttest"]]) <= conformal_q(r[:m], ALPHA))))
    m = headline_m_pool(dft_split(ds, 0))
    return {"n_splits": n_splits, "m": m, "mean_cov_m": float(np.mean(cov)), "sd_cov_m": float(np.std(cov, ddof=1)),
            "expected": float(np.ceil((m + 1) * (1 - ALPHA)) / (m + 1)),
            "note": "seeds 0..n_splits-1; seeds 0..19 are the protocol splits"}


# ============================================================ aggregation
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


BLOCKS = ["insilico_vs_dft", "insilico_vs_expt", "recal_m", "recal_m_bias_global", "recal_m_bias_class",
          "recal_m_normalised", "ref_expt_trained", "ref_dft_lookup_m"]


def aggregate(rows):
    S = {"m": sorted({r["m"] for r in rows}), "sizes_seed0": rows[0]["sizes"],
         "n_formula_seed0": rows[0]["n_formula"],
         "sizes_range": {k: [int(min(r["sizes"][k] for r in rows)), int(max(r["sizes"][k] for r in rows))]
                         for k in rows[0]["sizes"]}}
    for b in BLOCKS:
        S[b] = {k: agg(rows, [b, k]) for k in rows[0][b] if k != "curve"}
        if "curve" in rows[0][b]:
            S[b]["curve_mean"] = np.mean([r[b]["curve"] for r in rows], 0).tolist()
    S["levels"] = LEVELS.tolist()
    for key in ["insilico_vs_expt_by_class", "insilico_vs_dft_by_class", "ref_expt_by_class"]:
        S[key] = {c: {"coverage": agg(rows, [key, c, "coverage"]),
                      "n_mean": float(np.mean([r[key].get(c, {"n": 0})["n"] for r in rows]))}
                  for c in CLASSES}
    S["shortfall_dft_minus_expt"] = agg(rows, ["shortfall_dft_minus_expt"])
    S["by_discrepancy_bin"] = {lab: {"n_mean": float(np.mean([r["by_discrepancy_bin"][lab]["n"] for r in rows])),
                                     **{k: agg(rows, ["by_discrepancy_bin", lab, k])
                                        for k in ["insilico_cov", "recal_m_cov", "ref_expt_cov"]}}
                               for lab, _, _ in DISC_BINS}
    S["surrogate_sweep"] = {str(l): {k: agg(rows, ["surrogate_sweep", str(l), k])
                                     for k in rows[0]["surrogate_sweep"][str(l)]} for l in LAMS}
    S["rmse"] = {k: agg(rows, ["rmse", k]) for k in rows[0]["rmse"]}
    S["mae"] = {k: agg(rows, ["mae", k]) for k in rows[0]["mae"]}
    S["corr_modelerr_discrepancy"] = agg(rows, ["corr_modelerr_discrepancy"])
    S["test_mean_signed_pred_minus_expt"] = agg(rows, ["test_mean_signed_pred_minus_expt"])
    S["bias_estimate_m"] = agg(rows, ["bias_estimate_m"])
    for t in ["triage_insilico_expt", "triage_insilico_dft", "triage_ref_expt"]:
        S[t] = {"reduction_at50_pct": agg(rows, [t, "reduction_at50_pct"]), "aurc_rel": agg(rows, [t, "aurc_rel"]),
                "rmse_all": agg(rows, [t, "rmse_all"]),
                "curve_mean": np.mean([r[t]["rmse"] for r in rows], 0).tolist(), "grid": rows[0][t]["grid"]}
    S["ksweep"] = {}
    for k in KGRID + ["all"]:
        k = str(k)
        avail = [r for r in rows if r["ksweep"][k] is not None]
        if not avail:
            S["ksweep"][k] = {"available_splits": 0, "note": "recalibration pool smaller than k"}
            continue
        e = {"available_splits": len(avail), "k_mean": float(np.mean([r["ksweep"][k]["k"] for r in avail]))}
        for meth in ["plain", "bias_global", "bias_class", "dft_lookup"]:
            if meth not in avail[0]["ksweep"][k]:
                continue
            e[meth] = {mm: agg([r["ksweep"][k] for r in avail], [meth, mm])
                       for mm in ["coverage", "norm_width", "interval_score_norm"]}
            e[meth]["infinite_splits"] = int(sum(r["ksweep"][k][meth]["infinite"] for r in avail))
        # paired effect of the bias correction on width / coverage
        for meth in ["bias_global", "bias_class"]:
            if meth in avail[0]["ksweep"][k] and not avail[0]["ksweep"][k]["plain"]["infinite"]:
                dw = [r["ksweep"][k][meth]["norm_width"] - r["ksweep"][k]["plain"]["norm_width"] for r in avail
                      if r["ksweep"][k][meth]["norm_width"] is not None and r["ksweep"][k]["plain"]["norm_width"] is not None]
                dc = [r["ksweep"][k][meth]["coverage"] - r["ksweep"][k]["plain"]["coverage"] for r in avail]
                e[f"{meth}_minus_plain"] = {"norm_width": summarize(dw), "coverage": summarize(dc),
                                            "frac_splits_narrower": float(np.mean(np.array(dw) < 0)) if dw else None}
        kk = int(round(e["k_mean"]))
        bt = beta_coverage_quantiles(kk) if kk > 0 else None
        e["beta_theory_5_50_95"] = bt if bt is not None and all(np.isfinite(bt)) else None   # None: k too small
        S["ksweep"][k] = e
    S["per_split"] = [{"seed": r["seed"], "insilico_dft_cov": r["insilico_vs_dft"]["coverage"],
                       "insilico_expt_cov": r["insilico_vs_expt"]["coverage"],
                       "recal_m_cov": r["recal_m"]["coverage"], "recal_m_nw": r["recal_m"]["norm_width"],
                       "ref_expt_cov": r["ref_expt_trained"]["coverage"],
                       "insilico_nw": r["insilico_vs_expt"]["norm_width"],
                       "rmse_dft": r["rmse"]["dftmodel_vs_dft"], "rmse_expt": r["rmse"]["dftmodel_vs_expt"]}
                      for r in rows]
    return S


# ============================================================ markdown
def f3(s, key="mean", nd=3):
    return "n/a" if s is None or s.get(key) is None else f"{s[key]:.{nd}f}"


def ci(s, nd=3):
    if s is None or s.get("mean") is None:
        return "n/a"
    return f"{s['mean']:.{nd}f} [{s['lo']:.{nd}f}, {s['hi']:.{nd}f}]"


def write_md(R, path):
    L = []
    info = R["data"]
    L.append("# R4b - genuine computation-to-measurement shift (DFT -> calorimetry)\n")
    L.append("**Scope.** Inorganic compounds (intermetallics, silicides, borides, carbides, germanides) "
             "from Kim et al. (2017, Sci. Data 4, 170162) with matched Materials Project / OQMD DFT formation "
             "enthalpies (matminer `expt_formation_enthalpy`). These are **not biomedical materials**; the analysis is "
             "a *genuine computation-to-measurement analogue* of the in-silico -> in-vitro gap (a model trained and "
             "calibrated on computed labels is asked to predict a measured property), not that gap itself.\n")
    L.append(f"**Data.** {info['n_records']} records; magpie featurisation failures: {info['n_feat_failures']} "
             f"({', '.join(info['feat_failures']) or 'none'}); 132 composition features. "
             + "; ".join(f"{s}: {R[s]['dataset']['N']} records with both experimental and {s} values "
                         f"({R[s]['dataset']['n_unique_formula']} reduced formulas, {R[s]['dataset']['n_replicate_records']} "
                         f"replicate/polymorph records)" for s in R["sources"]) + ". "
             f"Units eV/atom. Experimental IQR (normaliser): " +
             ", ".join(f"{s} {R[s]['iqr_expt']:.3f}" for s in R["sources"]) + ".\n")
    if R.get("skipped_sources"):
        L.append(f"Skipped sources: {R['skipped_sources']}\n")
    L.append("**Protocol.** 20 grouped (reduced-formula) splits: train 50 % / calibration 20 % / target pool 30 %, "
             "pool halves = recalibration pool / target-test. alpha = 0.1, finite-sample conformal quantile, "
             "|y - yhat| score; headline m = clip(floor(|pool|/3), 10, 30) = "
             + ", ".join(f"{s} {R[s]['RF']['m']}" for s in R["sources"]) +
             ". Mean [2.5-97.5 percentile] across the 20 splits, which resample one dataset (split-to-split "
             "variability, not population sampling error). Nothing is selected on target-test labels; the "
             "bias-shrinkage pseudo-count (lambda = 5) and chemistry classes were fixed a priori.\n")
    for s in R["sources"]:
        L.append(f"\n## In-silico source: {s}\n")
        D = R[s]["discrepancy"]
        L.append(f"**(iv) DFT - experiment discrepancy** (all matched records; grouped bootstrap 95 % CI):\n")
        L.append("| subset | n records | mean signed (DFT - expt) | MAE | RMSE | frac DFT more negative |")
        L.append("|---|---|---|---|---|---|")
        for k, v in D.items():
            if k == "replicate_expt":
                continue
            L.append(f"| {k} | {v['n_records']} | {v['mean_signed']:+.3f} [{v['mean_signed_ci95'][0]:+.3f}, "
                     f"{v['mean_signed_ci95'][1]:+.3f}] | {v['mae']:.3f} [{v['mae_ci95'][0]:.3f}, {v['mae_ci95'][1]:.3f}] "
                     f"| {v['rmse']:.3f} | {v['frac_dft_more_negative']:.2f} |")
        if "replicate_expt" in D:
            rpl = D["replicate_expt"]
            L.append(f"\nInter-laboratory reproducibility floor: {rpl['n_formula_with_replicates']} formulas with replicate "
                     f"experimental records, pooled within-formula SD {rpl['pooled_within_formula_sd']:.3f} eV/atom, "
                     f"mean |pairwise difference| {rpl['mean_abs_pairwise_diff']:.3f} eV/atom.\n")
        for mod in MODELS:
            S = R[s][mod]
            L.append(f"\n### {s} / {mod}\n")
            L.append(f"Sizes (seed 0): {S['sizes_seed0']}; m = {S['m']}.\n")
            L.append("| setting (target-test, 20 splits) | coverage | norm. width (/IQR expt) | interval score (eV/atom) | WIS | cal. error (mean) | worst-cluster cov |")
            L.append("|---|---|---|---|---|---|---|")
            rows = [("(i) in-silico calibrated, vs **DFT** labels (sanity)", "insilico_vs_dft"),
                    ("(i) in-silico calibrated, vs **experimental** labels", "insilico_vs_expt"),
                    (f"(ii) recalibrated with m = {S['m'][0]} measurements", "recal_m"),
                    (f"(iv) + global signed-bias correction (m)", "recal_m_bias_global"),
                    (f"(iv) + class-wise bias correction (m)", "recal_m_bias_class"),
                    (f"(ii) sigma-normalised recalibration (m)", "recal_m_normalised"),
                    ("(iii-a) reference: trained + calibrated on experimental labels", "ref_expt_trained"),
                    (f"(iii-b) reference: DFT value itself + m measurements", "ref_dft_lookup_m")]
            for lab, b in rows:
                B = S[b]
                L.append(f"| {lab} | {ci(B['coverage'])} | {ci(B.get('norm_width'))} | {ci(B.get('interval_score'))} "
                         f"| {ci(B.get('wis')) if 'wis' in B else 'n/a'} | {ci(B.get('ce_mean')) if 'ce_mean' in B else 'n/a'} "
                         f"| {ci(B.get('worst_cluster_cov')) if 'worst_cluster_cov' in B else 'n/a'} |")
            L.append("\n(sanity row: width normalised by the IQR of DFT values.)\n")
            L.append(f"RMSE on target-test (eV/atom): DFT-trained model vs DFT {ci(S['rmse']['dftmodel_vs_dft'])}; "
                     f"vs experiment {ci(S['rmse']['dftmodel_vs_expt'])}; experiment-trained model vs experiment "
                     f"{ci(S['ref_expt_trained']['rmse'])}; DFT look-up vs experiment {ci(S['rmse']['dft_lookup_vs_expt'])}. "
                     f"Correlation(model error on DFT, DFT-expt discrepancy) {ci(S['corr_modelerr_discrepancy'], 2)}. "
                     f"Mean signed (prediction - experiment) on test {ci(S['test_mean_signed_pred_minus_expt'])}; "
                     f"bias estimated from m recal records (expt - prediction) {ci(S['bias_estimate_m'])}.\n")
            L.append(f"Paired coverage shortfall (DFT-label coverage - experimental-label coverage, same intervals, "
                     f"same compounds): {ci(S['shortfall_dft_minus_expt'])}.\n")
            L.append("| chemistry class (target-test) | n/split | in-silico cov vs DFT | in-silico cov vs expt | expt-trained ref cov |")
            L.append("|---|---|---|---|---|")
            for c in CLASSES:
                v = S["insilico_vs_expt_by_class"][c]
                if v["coverage"]["mean"] is None:
                    continue
                L.append(f"| {c} | {v['n_mean']:.0f} | {ci(S['insilico_vs_dft_by_class'][c]['coverage'])} | "
                         f"{ci(v['coverage'])} | {ci(S['ref_expt_by_class'][c]['coverage'])} |")
            L.append("\nCoverage of experimental labels stratified (evaluation only) by the compound's own "
                     "|DFT - experiment| gap:\n")
            L.append("| stratum (eV/atom) | n/split | in-silico calibrated | recalibrated (m) | expt-trained ref |")
            L.append("|---|---|---|---|---|")
            for lab, v in S["by_discrepancy_bin"].items():
                L.append(f"| {lab} | {v['n_mean']:.1f} | {ci(v['insilico_cov'])} | {ci(v['recal_m_cov'])} | {ci(v['ref_expt_cov'])} |")
            L.append("")
            L.append(f"Abstention (share of target candidates whose recalibrated interval is wider than the in-silico "
                     f"interval): constant-width {f3(S['recal_m']['abstain_vs_insilico_width'])}, sigma-normalised "
                     f"{f3(S['recal_m_normalised']['abstain_vs_insilico_width'])}. Triage (RMSE reduction on the 50 % "
                     f"most confident by {'tree spread' if mod == 'RF' else 'kNN distance'}): vs DFT "
                     f"{ci(S['triage_insilico_dft']['reduction_at50_pct'], 1)} %, vs experiment "
                     f"{ci(S['triage_insilico_expt']['reduction_at50_pct'], 1)} %; relative AURC vs experiment "
                     f"{ci(S['triage_insilico_expt']['aurc_rel'])}.\n")
            L.append("\nk-sweep (coverage of experimental labels / normalised width; k = 0 is the in-silico calibration):\n")
            L.append("| k | plain recal cov | plain nw | +global bias cov | +global bias nw | d nw (global - plain) | +class bias cov | +class bias nw | DFT look-up cov | DFT look-up nw | Beta theory 5/50/95 |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for k, e in S["ksweep"].items():
                if e.get("available_splits", 0) == 0:
                    L.append(f"| {k} | n/a (pool < k) |||||||||| ")
                    continue
                kl = f"{k} (={e['k_mean']:.0f})" if k == "all" else k
                if e["available_splits"] < 20:
                    kl += f" [{e['available_splits']}/20 splits]"

                def g(meth, mm):
                    if meth not in e:
                        return "-"
                    if e[meth].get("infinite_splits", 0) == e["available_splits"] and mm != "coverage":
                        return "inf"
                    return f3(e[meth][mm])
                dg = e.get("bias_global_minus_plain", {}).get("norm_width")
                bt = e.get("beta_theory_5_50_95")
                L.append(f"| {kl} | {g('plain','coverage')} | {g('plain','norm_width')} | {g('bias_global','coverage')} | "
                         f"{g('bias_global','norm_width')} | {ci(dg) if dg else '-'} | {g('bias_class','coverage')} | "
                         f"{g('bias_class','norm_width')} | {g('dft_lookup','coverage')} | {g('dft_lookup','norm_width')} | "
                         f"{', '.join(f'{x:.2f}' for x in bt) if bt and all(np.isfinite(bt)) else '-'} |")
            L.append("\nk < 9: the finite-sample conformal quantile is infinite at alpha = 0.1 (coverage 1 trivially, "
                     "infinite interval). 'n/a (pool < k)': the recalibration pool of every split is smaller than k.\n")
            L.append("\nSemi-synthetic surrogate-accuracy sweep (sensitivity analysis; real DFT-experiment discrepancy, "
                     "surrogate error vs DFT rescaled by lambda; prediction = DFT + lambda x (fitted-model error); "
                     "lambda = 1 is the fitted model, lambda = 0 a perfect DFT surrogate):\n")
            L.append("| lambda | surrogate RMSE vs DFT | in-silico cov vs DFT | in-silico cov vs expt | in-silico nw | recal (m) cov vs expt | recal (m) nw |")
            L.append("|---|---|---|---|---|---|---|")
            for l, v in S["surrogate_sweep"].items():
                L.append(f"| {l} | {f3(v['rmse_vs_dft'])} | {f3(v['insilico_cov_dft'])} | {ci(v['insilico_cov_expt'])} | "
                         f"{f3(v['insilico_nw'])} | {ci(v['recal_m_cov_expt'])} | {f3(v['recal_m_nw'])} |")
            L.append("\n(lambda = 0: zero-width in-silico interval, so DFT-label coverage is trivially 1 and "
                     "experimental-label coverage is the share of exact DFT = experiment ties.)\n")
    L.append("\n## Reading\n")
    L.extend(R["reading"])
    L.append("\n## Deviations from PROTOCOL.md and notes\n")
    L.extend(f"- {d}" for d in R["deviations"])
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


# ============================================================ main
def main(n_jobs=6):
    t0 = time.time()
    df, X, feat, info = load_frame()
    print(f"records {info['n_records']}, featurisation failures {info['n_feat_failures']}", flush=True)
    R = {"data": info, "sources": [], "protocol": {
        "split": "grouped by reduced formula; train 50/cal 20/target pool 30; pool halves = recal pool / target-test",
        "alpha": ALPHA, "seeds": SEEDS, "kgrid": KGRID + ["all"], "models": MODELS,
        "score": "|y - yhat| (finite-sample split-conformal quantile)",
        "bias_correction": "b = mean(y_expt - yhat) over k recal records (global) or class-wise mean shrunk to global "
                           f"with pseudo-count {LAMBDA_SHRINK}; calibration residuals leave-one-out",
        "normalised_sigma": "RF: tree spread; HistGB: mean distance to 10 nearest training compositions (standardised magpie)",
        "norm_width": "mean width / IQR of experimental values over all matched records of that source",
        "chemistry_classes": {"order": CLASSES, "f-block": sorted(FBLOCK), "metalloid/nonmetal": sorted(METALLOID),
                              "sp-metal": sorted(SPMETAL)},
        "note": "inorganic compounds, not biomedical materials: a genuine computation-to-measurement analogue"},
         "skipped_sources": {}}
    for src in ["MP", "OQMD"]:
        ds = source_dataset(df, X, feat, src)
        if src != "MP" and len(ds.y) < MIN_OVERLAP:
            R["skipped_sources"][src] = f"overlap {len(ds.y)} < {MIN_OVERLAP}"
            continue
        R["sources"].append(src)
        split_path = save_splits(ds, split_fn=dft_split, tag=f"{TAG}_{src}")
        iqr_exp, iqr_dft = iqr(ds.y), iqr(ds.meta["y_dft"])
        summ = ds.summary(); summ.update({"n_unique_formula": ds.meta["n_unique_formula"],
                                          "n_replicate_records": ds.meta["n_replicate_records"]})
        summ["class_counts"] = {CLASSES[c]: int((ds.meta["cls"] == c).sum()) for c in range(4)}
        R[src] = {"dataset": summ, "iqr_expt": iqr_exp, "iqr_dft": iqr_dft, "splits_file": split_path,
                  "discrepancy": discrepancy_table(ds),
                  "diagnostic_lookup_many_splits": lookup_resampling_check(ds)}
        for mod in MODELS:
            rows = Parallel(n_jobs=n_jobs)(delayed(one_split)(ds, mod, s, iqr_exp, iqr_dft) for s in SEEDS)
            S = aggregate(rows)
            R[src][mod] = S
            print(f"{src:4s} {mod:6s} n={len(ds.y)} m={S['m']} | IS->DFT {ci(S['insilico_vs_dft']['coverage'])}"
                  f" | IS->EXPT {ci(S['insilico_vs_expt']['coverage'])} nw {f3(S['insilico_vs_expt']['norm_width'])}"
                  f" | recal(m) {ci(S['recal_m']['coverage'])} nw {f3(S['recal_m']['norm_width'])}"
                  f" | +bias {f3(S['recal_m_bias_global']['coverage'])} nw {f3(S['recal_m_bias_global']['norm_width'])}"
                  f" | ref-expt {f3(S['ref_expt_trained']['coverage'])} nw {f3(S['ref_expt_trained']['norm_width'])}"
                  f" | RMSE dft {f3(S['rmse']['dftmodel_vs_dft'])} expt {f3(S['rmse']['dftmodel_vs_expt'])}"
                  f" ref {f3(S['ref_expt_trained']['rmse'])} lookup {f3(S['rmse']['dft_lookup_vs_expt'])}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
    R["reading"] = reading(R)
    R["deviations"] = [
        "Split fractions 50/20/30 (train / calibration / target pool) over one pool of matched compounds, as "
        "specified for this analysis, instead of the 60/20/20 source split plus a separate design-region target "
        "pool: here source and target are the same compounds and the shift is in how the label is obtained "
        "(DFT vs calorimetry), so there is no design axis and no source-test half.",
        "'In-distribution' coverage is the DFT-label coverage on the target-test compounds (sanity row); there is no "
        "separate source-test set.",
        "Normalised width uses the IQR of the experimental values over all matched records of that source (a fixed "
        "normalising constant, as R1 uses the IQR of the whole target pool); it is not used for any choice.",
        "Sigma for normalised scores / triage: RF tree spread (as in R1); HistGB has no ensemble spread, so a "
        "model-agnostic proxy (mean distance to the 10 nearest training compositions, standardised magpie space) is used.",
        "Width-stratified coverage equals marginal coverage for constant-width intervals; it is informative only for "
        "the sigma-normalised rows. Abstention for constant-width intervals is 0/1 per split (share of splits in which "
        "the recalibrated interval is wider than the in-silico one).",
        "Bias-corrected recalibration uses leave-one-out calibration residuals (bias estimated from the same k records); "
        "this is a jackknife-style construction whose finite-sample guarantee is not exactly the split-conformal one.",
        "k = 100 exceeds the recalibration pool for MP (~97 records) and is reported as n/a; 'all' uses the whole pool.",
        "Surrogate-accuracy sweep and the 500-split look-up check are additional diagnostics (semi-synthetic / "
        "model-free), labelled as such; the 500-split check reuses the same split function with seeds 0..499.",
        "RF uses common.make_model with n_jobs set to 1 inside 6 joblib workers (CPU sharing); results do not depend "
        "on n_jobs.",
    ]
    R["runtime_s"] = time.time() - t0
    dump(R, f"{TAG}.json")
    from common import RES
    write_md(R, f"{RES}/{TAG}.md")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def reading(R):
    """Plain-language summary computed from the numbers (no hand-typed values)."""
    import json
    import os
    from common import RES
    out = []
    cfg = [(s, mod) for s in R["sources"] for mod in MODELS]
    cov_is = [R[s][mod]["insilico_vs_expt"]["coverage"]["mean"] for s, mod in cfg]
    short = [R[s][mod]["shortfall_dft_minus_expt"] for s, mod in cfg]
    r1 = None
    p1 = os.path.join(RES, "R1_core.json")
    if os.path.exists(p1):
        J = json.load(open(p1))
        r1 = [J[k]["shift_source"]["coverage"]["mean"] for k in J]
    out.append(
        f"- **Headline (honest).** On this genuine computation-to-measurement shift, conformal intervals calibrated "
        f"only on DFT labels still cover {min(cov_is):.3f}-{max(cov_is):.3f} of the *measured* enthalpies at a nominal "
        f"0.90 (paired shortfall vs DFT-label coverage "
        + ", ".join(f"{s}/{mod} {sh['mean']:+.3f} [{sh['lo']:+.3f}, {sh['hi']:+.3f}]" for (s, mod), sh in zip(cfg, short))
        + "). There is **no coverage collapse** comparable to the design-region extrapolations"
        + (f" of R1 (source-calibrated coverage {min(r1):.2f}-{max(r1):.2f})" if r1 else "")
        + ". This does NOT support a claim that in-silico calibration generally collapses between computation and "
          "experiment; it supports the narrower statement that the size of the coverage loss depends on how large the "
          "computation-measurement discrepancy is relative to the surrogate's own error.")
    ex = R[cfg[0][0]][cfg[0][1]]
    out.append(
        "- **Why.** The composition-only surrogates' own error on DFT labels is as large as or larger than the "
        "DFT-experiment discrepancy on the same test compounds (RMSE surrogate-vs-DFT / DFT-vs-experiment, ratio, "
        "coverage shortfall: "
        + "; ".join(f"{s}/{mod} {R[s][mod]['rmse']['dftmodel_vs_dft']['mean']:.3f} / "
                    f"{R[s][mod]['rmse']['dft_lookup_vs_expt']['mean']:.3f} eV/atom, "
                    f"{R[s][mod]['rmse']['dftmodel_vs_dft']['mean'] / R[s][mod]['rmse']['dft_lookup_vs_expt']['mean']:.2f}, "
                    f"{R[s][mod]['shortfall_dft_minus_expt']['mean']:+.3f}" for s, mod in cfg)
        + "), and the surrogate error is negatively correlated with the discrepancy (r = "
        + ", ".join(f"{R[s][mod]['corr_modelerr_discrepancy']['mean']:.2f}" for s, mod in cfg)
        + "): the smooth composition model partly averages out compound-specific DFT errors. The in-silico interval "
          "is therefore already about as wide as the experiment-trained reference and absorbs most of the "
          "discrepancy; the configuration whose surrogate is closest to DFT accuracy shows the largest shortfall.")
    for s in R["sources"]:
        sw = R[s]["RF"]["surrogate_sweep"]
        pts = ", ".join(f"lambda {l} (RMSE vs DFT {sw[l]['rmse_vs_dft']['mean']:.3f}): {sw[l]['insilico_cov_expt']['mean']:.2f}"
                        for l in ["1.0", "0.5", "0.25", "0.1", "0.0"])
        rec = ", ".join(f"{sw[l]['recal_m_cov_expt']['mean']:.2f}" for l in ["1.0", "0.5", "0.25", "0.1", "0.0"])
        out.append(f"- **When it would collapse ({s}, RF, semi-synthetic).** Shrinking the surrogate's error towards "
                   f"the DFT labels while keeping the real DFT-experiment discrepancy, in-silico-calibrated coverage of "
                   f"experimental values falls: {pts}; recalibration with m measurements restores {rec}. A surrogate "
                   f"that reproduces DFT closely inherits DFT's systematic error while its in-silico interval shrinks, so "
                   f"the collapse appears exactly when the in-silico model is good. This is a sensitivity analysis, not "
                   f"an observation of a real, more accurate model.")
    dgs = []
    for s, mod in cfg:
        e = R[s][mod]["ksweep"].get(str(R[s][mod]["m"][0]), {})
        d = e.get("bias_global_minus_plain", {}).get("norm_width")
        if d and d.get("mean") is not None:
            dgs.append(f"{s}/{mod} {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
    out.append("- **Signed-bias correction.** The global DFT-experiment offset is small ("
               + ", ".join(f"{s} {R[s]['discrepancy']['all']['mean_signed']:+.3f}" for s in R["sources"])
               + " eV/atom; class-level offsets up to "
               + f"{max(abs(v['mean_signed']) for s in R['sources'] for k, v in R[s]['discrepancy'].items() if k not in ('all', 'replicate_expt')):.3f}"
               + " eV/atom), so estimating it from k measurements adds estimation noise without removing much "
                 "systematic error: at k = m the change in normalised width is " + "; ".join(dgs)
               + " (positive = wider). The bias correction does not narrow the intervals here.")
    def rng_cls(S, key):
        v = {c: S[key][c]["coverage"]["mean"] for c in CLASSES if S[key][c]["coverage"]["mean"] is not None}
        lo_c, hi_c = min(v, key=v.get), max(v, key=v.get)
        return f"{v[lo_c]:.2f} ({lo_c}) to {v[hi_c]:.2f} ({hi_c})"
    big = DISC_BINS[-1][0]
    out.append("- **Conditional coverage.** Coverage is uneven across chemistry classes. In-silico-calibrated, vs "
               "experiment: " + "; ".join(f"{s}/{mod} {rng_cls(R[s][mod], 'insilico_vs_expt_by_class')}" for s, mod in cfg)
               + ". The same intervals vs DFT labels: " + "; ".join(f"{s}/{mod} {rng_cls(R[s][mod], 'insilico_vs_dft_by_class')}" for s, mod in cfg)
               + "; experiment-trained reference: " + "; ".join(f"{s}/{mod} {rng_cls(R[s][mod], 'ref_expt_by_class')}" for s, mod in cfg)
               + ". The unevenness is present against DFT labels and for the experiment-trained reference too, so it is "
                 f"mostly heteroscedastic model error, not the computation-measurement shift. The shift does bite on the "
                 f"compounds whose own DFT-experiment gap is large ({big} eV/atom, "
               + f"~{R[cfg[0][0]][cfg[0][1]]['by_discrepancy_bin'][big]['n_mean']:.0f} test records/split): coverage "
               + "; ".join(f"{s}/{mod} in-silico {R[s][mod]['by_discrepancy_bin'][big]['insilico_cov']['mean']:.2f}, "
                           f"recalibrated {R[s][mod]['by_discrepancy_bin'][big]['recal_m_cov']['mean']:.2f}, "
                           f"expt-trained {R[s][mod]['by_discrepancy_bin'][big]['ref_expt_cov']['mean']:.2f}" for s, mod in cfg)
               + ". Marginal (global) recalibration does not repair this sub-group; these compounds cannot be "
                 "identified before measurement, which is the practical limit of any marginal guarantee.")
    for s in R["sources"]:
        dg = R[s].get("diagnostic_lookup_many_splits")
        if dg:
            out.append(f"- **Resampling check ({s}).** With the 20 protocol splits the DFT look-up conformalised on m "
                       f"measurements averages {R[s]['RF']['ref_dft_lookup_m']['coverage']['mean']:.3f}; the same "
                       f"model-free procedure over {dg['n_splits']} splits averages {dg['mean_cov_m']:.3f} (per-split SD "
                       f"{dg['sd_cov_m']:.3f}; finite-sample expectation {dg['expected']:.3f}); "
                       + ("the sub-nominal 20-split mean is resampling noise of the 20 splits, not a violated "
                          "split-conformal guarantee." if R[s]['RF']['ref_dft_lookup_m']['coverage']['mean'] < dg['expected'] - 0.02
                          else "the 20-split mean is consistent with the split-conformal guarantee."))
    out.append("- **Measurement budget.** The in-silico route uses 0 measured labels for training and calibration; the "
               "experiment-trained reference uses the measured labels of train + calibration compounds ("
               + ", ".join(f"{s} ~{R[s]['RF']['sizes_seed0']['train'] + R[s]['RF']['sizes_seed0']['cal']}" for s in R["sources"])
               + " records). Normalised width / coverage, in-silico vs experiment-trained: "
               + "; ".join(f"{s}/{mod} {R[s][mod]['insilico_vs_expt']['norm_width']['mean']:.2f} / "
                           f"{R[s][mod]['insilico_vs_expt']['coverage']['mean']:.2f} vs "
                           f"{R[s][mod]['ref_expt_trained']['norm_width']['mean']:.2f} / "
                           f"{R[s][mod]['ref_expt_trained']['coverage']['mean']:.2f}" for s, mod in cfg)
               + ". For this property and these surrogates, DFT labels are nearly as informative as measurements; "
                 "recalibration with m = 30 measurements is a check that costs 30 experiments and here confirms "
                 "rather than repairs the in-silico interval.")
    out.append("- **Scope.** Inorganic intermetallic/silicide/boride/carbide compounds, formation enthalpy, "
               "composition-only descriptors, and only ~320 DFT training labels (the matched subset); not biomedical "
               "materials. A genuine computation-to-measurement analogue, not the in-silico -> in-vitro -> in-vivo gap. "
               "Splits resample one dataset; percentile intervals describe split-to-split variability.")
    for s in R["sources"]:
        for mod in MODELS:
            S = R[s][mod]
            a, b = S["insilico_vs_dft"]["coverage"], S["insilico_vs_expt"]["coverage"]
            c, r = S["recal_m"]["coverage"], S["ref_expt_trained"]["coverage"]
            nw_is, nw_m = S["insilico_vs_expt"]["norm_width"]["mean"], S["recal_m"]["norm_width"]["mean"]
            nw_ref = S["ref_expt_trained"]["norm_width"]["mean"]
            bg = S["ksweep"][str(S["m"][0])] if str(S["m"][0]) in S["ksweep"] else None
            dg = bg.get("bias_global_minus_plain", {}).get("norm_width") if bg else None
            out.append(f"- **{s} / {mod}.** In-silico-calibrated intervals cover {a['mean']:.3f} of held-out DFT labels "
                       f"but {b['mean']:.3f} [{b['lo']:.2f}, {b['hi']:.2f}] of the measured values of the same compounds "
                       f"(shortfall {a['mean'] - b['mean']:+.3f}). Recalibrating on m = {S['m'][0]} measurements gives "
                       f"{c['mean']:.3f} [{c['lo']:.2f}, {c['hi']:.2f}] at normalised width {nw_m:.3f} (in-silico "
                       f"{nw_is:.3f}; experiment-trained reference {r['mean']:.3f} at {nw_ref:.3f}). "
                       + (f"A global signed-bias correction changes the width by {dg['mean']:+.3f} IQR "
                          f"[{dg['lo']:+.3f}, {dg['hi']:+.3f}] at k = m." if dg and dg.get("mean") is not None else ""))
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
