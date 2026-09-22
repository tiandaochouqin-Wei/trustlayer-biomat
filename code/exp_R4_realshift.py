"""R4_realshift -- GENUINE (non-constructed) distribution shifts in SciGlass.

Answers Reviewer 1 comment 2 and Reviewer 2 comments 3 and 8. Binding protocol:
revision/code/PROTOCOL.md + common.py. Design follows the scoping study
revision/audit/A2_sciglass_explore.md (sections 2.1, 2.5, 2.6), upgraded to the
protocol (grouped splits, 60/20/20 source, recalibration pool / target-test halves,
alpha = 0.1, finite-sample conformal quantile, 20 evaluations, mean and 2.5-97.5
percentile across evaluations, full metric set).

Data: bioactive SciGlass subset (Si, Ca, Na > 0) via common.glass_dataset for Tg,
Young's modulus, microhardness and liquidus temperature (25 element atomic-fraction
features; composition groups = common.composition_groups at 1e-6). Metadata: first
author (normalised to a laboratory key exactly as A2 section 2.1), publication year,
ChemicalAnalysis flag, publication code Kod = SciGlass ID // 1e8.

Model: common.make_model("RF", seed) (300 trees, min_samples_leaf = 2), with
n_jobs lowered to 1 because evaluations run in parallel (hyperparameters unchanged).

(A) PRIMARY   repeated grouped 5-fold leave-laboratory-out (5 folds of whole
    laboratories balanced by record count, 4 repeats = 20 evaluations).
    Source (other labs) -> composition-grouped 60/20/20 train / source-cal / source-test
    (i  row-cal, the usual i.i.d. practice) and, separately, a laboratory-grouped
    60/20/20 split (ii lab-cal, calibration set = whole source laboratories).
    Target fold -> (a) composition-grouped record halves recal-pool / target-test;
                   (b) laboratory-grouped halves (recal and test from different labs),
                       recal records taken either spread over the pool's labs
                       ("spread", headline) or in laboratory order ("clustered",
                       i.e. one or two partner labs supply all k records).
    (iii) target recalibration k in {0, 9, 10, 20, 30, 40} and headline m;
    (iv)  lab-specific recalibration (k = 10 of the lab's own records);
    per-laboratory conditional coverage; weighted conformal (covariate-shift fix);
    covariate-vs-label decomposition (domain AUC, density-ratio reweighted source
    RMSE, ESS/n); identical-composition cross-lab check.
(B) SECONDARY temporal (Tg >= 2016, E >= 2016, microhardness >= 2012, liquidus >= 2017),
    20 seeds, same metrics, plus (ii-t) "recent-cal": train on the earliest 60 % of the
    source by year, calibrate on the latest 20 %.
(C) TERTIARY  nominal -> chemically analysed composition (protocol + laboratory shift),
    20 seeds, same metrics.
(R) REFERENCE ungrouped random-row split (20 seeds): same-lab versus cross-lab
    identical-composition errors (intra- versus inter-laboratory scatter).
(D) SENSITIVITY phosphorus design shift of R1 (common.design_split, identical source
    split and model): target recal pool / target-test grouped by publication
    (Kod; also Author+Year, the proxy used by the split audit A1), 20 seeds,
    compared with the composition-grouped recalibration of results/R1_core.json.
(E) 45S5 Bioglass illustration (Tg records within +-1 mol% per oxide).

Outputs: results/R4_realshift.json, results/R4_realshift.md, results/R4_realshift.log,
results/splits/R4_<family>_<dataset>_splits.json.
Run:  python -W ignore exp_R4_realshift.py           (full, about 30-40 min on 6 workers)
      python -W ignore exp_R4_realshift.py --quick   (smoke test, microhardness only)
      python -W ignore exp_R4_realshift.py --md-only (rewrite the .md from the .json)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import binom
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import (ALPHA, LEVELS, GLASS_PROPS, SPLITS, RES, SEEDS, glass_dataset, _sciglass_frame,
                    _perm_grouped, _cut, design_split, check_disjoint, headline_m, conformal_q,
                    make_model, rf_mean_std, interval_metrics, weighted_interval_score,
                    calibration_error, risk_coverage, abstention_rate, summarize, iqr, dump,
                    beta_coverage_quantiles, composition_groups, interval_score)

TAG = "R4_realshift"
N_JOBS = 6            # joblib workers (shared machine: <= 6)
TREE_JOBS = 1         # threads per tree model (<= 4); 6 x 1 = 6 cores
KGRID = [0, 9, 10, 20, 30, 40]
WIS_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
N_FOLDS, N_REPEATS = 5, 4
TEMPORAL_CUT = {"Tg": 2016, "YoungModulus": 2016, "Microhardness": 2012, "Tliquidus": 2017}
MIN_LAB = 20          # laboratories with >= 20 evaluated records enter per-lab statistics
LAB_K = 10            # lab-specific recalibration size
LAB_DRAWS = 10        # random draws of the lab's own k records per evaluation
MATCH_DEC = 3         # identical-composition check at 1e-3 atomic fraction (A2)
BIOGLASS = {"SiO2": 0.461, "Na2O": 0.244, "CaO": 0.269, "P2O5": 0.026}
# Split audit A1 section 6.3, for the (D) comparison: publication-grouped recalibration of the
# phosphorus design shift under the SUBMITTED Table-3 protocol (70/30 source, m unchanged,
# Author+Year publication proxy, whole publications drawn into the pool, 10 seeds):
# property -> (random-recal mean, publication-grouped mean, share of splits < 0.90 random, grouped)
A1_PUBRECAL = {"Tg": (0.930, 0.911, 0.20, 0.30), "YoungModulus": (0.951, 0.911, 0.10, 0.30),
               "Microhardness": (0.938, 0.804, 0.20, 0.80), "Tliquidus": (0.947, 0.923, 0.10, 0.30)}
LOG = os.path.join(RES, f"{TAG}.log")
T0 = time.time()


def log(*a):
    msg = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


# ============================================================ laboratory key (A2 2.1)
def fold_ascii(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace("`", "").replace("'", "")
    s = re.sub(r"[^a-z\-\. ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def author_key(s):
    """Laboratory proxy = surname + first initial (A2 section 2.1, verbatim):
    ASCII-fold, lower-case, drop punctuation and generational suffixes, fold
    transliteration endings (-ii/-iy/-yi/-y/-ij -> -i)."""
    s = fold_ascii(str(s))
    toks = s.replace(".", ". ").split()
    if not toks:
        return "unknown"
    sur = toks[0].strip(".-,")
    rest = [t for t in toks[1:] if t.strip(".") not in ("iii", "ii", "jr", "sr")]
    ini = rest[0][0] if rest else ""
    sur = re.sub(r"(ii|iy|yi|y|ij)$", "i", sur)
    return sur + ("_" + ini if ini else "")


# ============================================================ data
_CACHE = {}
PROP_INFO = {p: (lab, u) for p, lab, u in GLASS_PROPS}


def get_data(prop):
    if prop in _CACHE:
        return _CACHE[prop]
    label, unit = PROP_INFO[prop]
    ds = glass_dataset(prop, label, unit)
    df = _sciglass_frame()
    ok = df[("property", prop)].notna().to_numpy()
    ids = df.index.to_numpy()[ok].astype(np.int64)
    author = np.array([str(a) for a in ds.meta["author"]])
    year = np.asarray(ds.meta["year"], float)
    D = {"prop": prop, "label": label, "unit": unit, "ds": ds, "X": ds.X, "y": ds.y,
         "g": ds.groups, "ids": ids, "author": author,
         "lab": np.array([author_key(a) for a in author]),
         "pub": ids // 10 ** 8, "year": year,
         "chem": np.asarray(ds.meta["chem_analysis"], bool),
         "key3": composition_groups(ds.X, MATCH_DEC),
         "compounds": df.loc[ok, "compounds"][list(BIOGLASS)].to_numpy(float) if prop == "Tg" else None}
    D["pub_ay"] = pd.factorize(pd.Series(author) + "|" + pd.Series(year.astype(int).astype(str)))[0]
    _CACHE[prop] = D
    return D


# ============================================================ splits
def _cut_nearest(order, gid, frac, min_pos=1):
    """Group boundary closest to frac*len (never splits a group; both parts non-empty)."""
    n = len(order)
    b = np.r_[np.where(gid[1:] != gid[:-1])[0] + 1]
    b = b[(b >= min_pos) & (b < n)]
    if len(b) == 0:
        return n
    return int(b[np.argmin(np.abs(b - frac * n))])


def lab_folds(lab, k, rng):
    """Whole laboratories -> k folds balanced by record count. Labs are processed
    roughly largest first (size multiplied by a U(0.5, 1.5) jitter so that repeats
    differ in which large labs are held out together) and each goes to the currently
    lightest fold."""
    s = pd.Series(lab).value_counts()
    labs, sizes = s.index.to_numpy(), s.to_numpy().astype(float)
    order = np.argsort(-sizes * rng.uniform(0.5, 1.5, len(sizes)), kind="stable")
    load = np.zeros(k)
    fold_of = {}
    for i in order:
        f = int(np.argmin(load + rng.random(k) * 1e-6))
        fold_of[labs[i]] = f
        load[f] += sizes[i]
    return np.array([fold_of[L] for L in lab])


def _drop_overlap(a, ref, g):
    """Remove from a the records whose composition group also occurs in ref."""
    return a[~np.isin(g[a], g[ref])]


def make_shift_split(D, src, tgt, seed, time_cal=False):
    """Split family for one evaluation of a genuine shift (source pool src, target tgt).
    train/cal/stest            composition-grouped 60/20/20 of the source   (i  row-cal)
    train_lab/cal_lab/stest_lab laboratory-grouped 60/20/20 of the source   (ii lab-cal);
                                cal/stest records whose composition is in train dropped
    train_time/stest_time/cal_time  (temporal only) earliest 60 % / next 20 % / latest 20 %
                                of the source by year (ii-t recent-cal); overlaps dropped
    rpool_a/ttest_a            composition-grouped halves of the target     (a)
    rpool_b_spread/rpool_b_clustered/ttest_b  laboratory-grouped halves (b); the pool is
                                ordered spread over its labs (composition-grouped
                                permutation) or clustered (laboratory order); test records
                                sharing a composition with the pool dropped."""
    rng = np.random.default_rng(seed)
    g, lab = D["g"], D["lab"]
    S, gS = _perm_grouped(src, g, rng)
    a, b = _cut(S, gS, 0.6), _cut(S, gS, 0.8)
    sp = {"train": S[:a], "cal": S[a:b], "stest": S[b:]}
    L, gL = _perm_grouped(src, lab, rng)
    a = _cut_nearest(L, gL, 0.6)
    b = _cut_nearest(L, gL, 0.8, min_pos=a + 1)
    tr, ca, st = L[:a], L[a:b], L[b:]
    ca = _drop_overlap(ca, tr, g)
    st = _drop_overlap(_drop_overlap(st, tr, g), ca, g)
    sp.update(train_lab=tr, cal_lab=ca, stest_lab=st)
    if time_cal:
        order = src[np.lexsort((rng.random(len(src)), D["year"][src]))]
        n = len(order)
        a, b = int(round(0.6 * n)), int(round(0.8 * n))
        tr, st, ca = order[:a], order[a:b], order[b:]
        ca = _drop_overlap(ca, tr, g)
        st = _drop_overlap(_drop_overlap(st, tr, g), ca, g)
        sp.update(train_time=tr, stest_time=st, cal_time=ca)
    T, gT = _perm_grouped(tgt, g, rng)
    h = _cut(T, gT, 0.5)
    sp["rpool_a"], sp["ttest_a"] = T[:h], T[h:]
    TL, gTL = _perm_grouped(tgt, lab, rng)
    h = _cut_nearest(TL, gTL, 0.5)
    pool_c, test_b = TL[:h], TL[h:]
    test_b = _drop_overlap(test_b, pool_c, g)
    pool_s = _perm_grouped(pool_c, g, rng)[0]
    sp.update(rpool_b_spread=pool_s, rpool_b_clustered=pool_c, ttest_b=test_b)
    # ---- protocol checks
    check_disjoint({k: sp[k] for k in ("train", "cal", "stest")}, g)
    check_disjoint({k: sp[k] for k in ("train_lab", "cal_lab", "stest_lab")}, g)
    check_disjoint({k: sp[k] for k in ("train_lab", "cal_lab", "stest_lab")}, lab)
    if time_cal:
        check_disjoint({k: sp[k] for k in ("train_time", "cal_time", "stest_time")}, g)
    check_disjoint({"rpool_a": sp["rpool_a"], "ttest_a": sp["ttest_a"]}, g)
    check_disjoint({"rpool_b": pool_s, "ttest_b": test_b}, g)
    check_disjoint({"rpool_b": pool_s, "ttest_b": test_b}, lab)
    check_disjoint({"source": src, "target": tgt})
    assert set(pool_s.tolist()) == set(pool_c.tolist())
    return sp


# ============================================================ helpers
def fit_rf(X, y, seed):
    return make_model("RF", seed).set_params(n_jobs=TREE_JOBS).fit(X, y)


def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else None


def cblock(idx, mu, rc, y, X, iqr_ref, full=True):
    """Constant-width split-conformal interval mu +- q(rc), evaluated on idx."""
    q = conformal_q(rc, ALPHA)
    yy, mm = y[idx], mu[idx]
    e = np.abs(yy - mm)
    if not np.isfinite(q):
        return {"coverage": 1.0, "infinite": 1.0, "n_cal": int(len(rc)), "n_eval": int(len(idx))}
    out = interval_metrics(mm - q, mm + q, yy, iqr_ref, X=X[idx] if full else None)
    out.update(q=float(q), infinite=0.0, n_cal=int(len(rc)), n_eval=int(len(idx)))
    if full:
        out["wis"] = weighted_interval_score(yy, mm, {a: (mm - conformal_q(rc, a), mm + conformal_q(rc, a))
                                                     for a in WIS_ALPHAS})
        covs = [float(np.mean(e <= conformal_q(rc, 1 - lv))) for lv in LEVELS]
        out["ce_mean"], out["ce_max"] = calibration_error(LEVELS, covs)
    return out


def kentry(rc, e, iqr_ref, y_eval, mu_eval):
    q = conformal_q(rc, ALPHA)
    d = {"coverage": float(np.mean(e <= q)), "infinite": float(not np.isfinite(q))}
    if np.isfinite(q):
        d.update(width=float(2 * q), norm_width=float(2 * q / iqr_ref),
                 interval_score=interval_score(mu_eval - q, mu_eval + q, y_eval))
    return d


def wcp_q(rc, wc, wt, alpha=ALPHA):
    """Vectorised Tibshirani et al. (2019) weighted conformal quantile (identical to
    common.weighted_conformal_q, one test weight per entry of wt)."""
    o = np.argsort(rc)
    s, cw = rc[o], np.cumsum(np.asarray(wc, float)[o])
    j = np.searchsorted(cw, (1 - alpha) * (cw[-1] + np.asarray(wt, float)), side="left")
    return np.where(j < len(s), s[np.minimum(j, len(s) - 1)], np.inf)


def domain(Xs, Xt, seed):
    """Cross-fitted (5-fold) RF domain classifier source(0) vs target(1); AUC and
    density-ratio weights w = p/(1-p) (class_weight='balanced' -> prior-corrected)."""
    Xd = np.vstack([Xs, Xt])
    d = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    p = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xd, d):
        c = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                   n_jobs=TREE_JOBS, random_state=seed).fit(Xd[tr], d[tr])
        p[te] = c.predict_proba(Xd[te])[:, 1]
    auc = float(roc_auc_score(d, p))
    pc = np.clip(p, 0.01, 0.99)
    w = pc / (1 - pc)
    return auc, w[:len(Xs)], w[len(Xs):]


def ess_frac(w):
    return float((w.sum() ** 2) / (w ** 2).sum() / len(w))


def per_lab(idx, lab, covmap, min_n=MIN_LAB):
    labs, cnt = np.unique(lab[idx], return_counts=True)
    rows = []
    for L, c in zip(labs, cnt):
        if c < min_n:
            continue
        sel = idx[lab[idx] == L]
        row = {"lab": str(L), "n": int(c)}
        for mth, cov in covmap.items():
            row[mth] = float(cov[sel].mean())
        row["p_below_0.8_if_exact"] = float(binom.cdf(int(np.ceil(0.8 * c - 1e-9)) - 1, c, 1 - ALPHA))
        rows.append(row)
    return rows


def per_lab_summary(rows, methods):
    out = {"n_labs": len(rows)}
    if not rows:
        return out
    out["expected_share_below_0.8_if_exact"] = float(np.mean([r["p_below_0.8_if_exact"] for r in rows])) \
        if "p_below_0.8_if_exact" in rows[0] else None
    for mth in methods:
        v = np.array([r[mth] for r in rows if mth in r], float)
        if len(v):
            out[mth] = {"median": float(np.median(v)), "share_below_0.8": float(np.mean(v < 0.8)),
                        "q25": float(np.percentile(v, 25)), "min": float(v.min())}
    return out


def lab_specific(tgt, excl, schemes, r_row, lab, g, rng):
    """(iv) For each target laboratory with >= MIN_LAB records: recalibrate on k = LAB_K
    of its own records (composition-grouped draw; k rises to the next composition
    boundary) and evaluate on its remaining records, excluding the fold-level recal
    set. Other schemes are evaluated on exactly the same records."""
    labs, cnt = np.unique(lab[tgt], return_counts=True)
    rows = []
    for L, c in zip(labs, cnt):
        if c < MIN_LAB:
            continue
        ii = tgt[lab[tgt] == L]
        acc = {"lab10": []}
        acc.update({s: [] for s in schemes})
        wid, kus, nev = [], [], []
        for _ in range(LAB_DRAWS):
            o, go = _perm_grouped(ii, g, rng)
            pos = min(LAB_K, len(o))
            while pos < len(o) and go[pos] == go[pos - 1]:
                pos += 1
            rec, ev = o[:pos], o[pos:]
            ev = ev[~np.isin(ev, excl)]
            if len(ev) < 5:
                continue
            qk = conformal_q(r_row[rec])
            acc["lab10"].append(float(np.mean(r_row[ev] <= qk)))
            for s, (rs, qs) in schemes.items():
                acc[s].append(float(np.mean(rs[ev] <= qs)))
            wid.append(2 * qk)
            kus.append(pos)
            nev.append(len(ev))
        if not wid:
            continue
        row = {"lab": str(L), "n": int(c), "k_used_mean": float(np.mean(kus)), "n_eval_mean": float(np.mean(nev)),
               "width_lab10": float(np.mean(wid)), "lab10_p10_over_draws": float(np.percentile(acc["lab10"], 10))}
        for s, v in acc.items():
            row[s] = float(np.mean(v))
        rows.append(row)
    return rows


def match_stats(train, ev, D, r, q):
    """Identical-composition check (1e-3): evaluation records whose composition occurs
    in the training set, split into same-laboratory and cross-laboratory-only matches;
    model error, coverage and a model-free 'lookup' error (|y - mean y of the matching
    training records|)."""
    k3, lab, y = D["key3"], D["lab"], D["y"]
    tk = pd.DataFrame({"k": k3[train], "L": lab[train], "y": y[train]})
    s_k = tk.groupby("k").y.agg(["sum", "count"])
    s_kl = tk.groupby(["k", "L"]).y.agg(["sum", "count"])
    kl_set = set(s_kl.index.tolist())
    matched = np.isin(k3[ev], s_k.index.to_numpy())
    same = np.array([(k3[i], lab[i]) in kl_set for i in ev], bool)
    cross = matched & ~same
    out = {"n_eval": int(len(ev)), "frac_matched": float(matched.mean()),
           "frac_same_lab_match": float(same.mean()), "frac_cross_lab_only_match": float(cross.mean())}
    for name, sel in (("unmatched", ~matched), ("same_lab", same), ("cross_lab_only", cross), ("matched", matched)):
        ii = ev[sel]
        d = {"n": int(len(ii))}
        if len(ii) >= 10:
            d.update(rmse_model=rmse(r[ii]), coverage=float(np.mean(r[ii] <= q)))
            if name in ("same_lab", "cross_lab_only"):
                sk = s_k.loc[k3[ii]].to_numpy()
                skl = np.array([s_kl.loc[(k3[i], lab[i])].to_numpy() if (k3[i], lab[i]) in kl_set else [0.0, 0.0]
                                for i in ii])
                if name == "same_lab":
                    look = skl[:, 0] / skl[:, 1]
                else:
                    look = sk[:, 0] / sk[:, 1]
                d["rmse_lookup"] = rmse(y[ii] - look)
        out[name] = d
    return out


# ============================================================ (A)(B)(C) evaluation
def evaluate_shift(prop, sp, seed, kind):
    D = get_data(prop)
    X, y, g, lab = D["X"], D["y"], D["g"], D["lab"]
    src = np.concatenate([sp["train"], sp["cal"], sp["stest"]])
    tgt = np.concatenate([sp["rpool_a"], sp["ttest_a"]])
    iqr_S, iqr_T = iqr(y[src]), iqr(y[tgt])
    mh = int(np.clip(len(tgt) // 3, 10, 30))
    O = {"seed": seed, "kind": kind, "m": mh, "iqr_source": iqr_S, "iqr_target": iqr_T,
         "sizes": {k: int(len(v)) for k, v in sp.items() if not k.startswith("_")},
         "n_labs": {"source": int(len(np.unique(lab[src]))), "target": int(len(np.unique(lab[tgt]))),
                    "rpool_b": int(len(np.unique(lab[sp["rpool_b_spread"]]))),
                    "ttest_b": int(len(np.unique(lab[sp["ttest_b"]])))},
         "target_frac_labs_unseen": float(np.mean(~np.isin(lab[tgt], lab[src]))),
         "target_frac_comp_in_source": float(np.mean(np.isin(g[tgt], g[src]))),
         "y_mean": {"source": float(y[src].mean()), "target": float(y[tgt].mean())}}

    # ---- models and calibration schemes
    m_row = fit_rf(X[sp["train"]], y[sp["train"]], seed)
    mu, sd = rf_mean_std(m_row, X)
    r = np.abs(y - mu)
    schemes = {"row": (mu, r, r[sp["cal"]], sp["stest"])}
    m_lab = fit_rf(X[sp["train_lab"]], y[sp["train_lab"]], seed)
    mu_l = m_lab.predict(X)
    r_l = np.abs(y - mu_l)
    schemes["lab"] = (mu_l, r_l, r_l[sp["cal_lab"]], sp["stest_lab"])
    if "train_time" in sp:
        m_t = fit_rf(X[sp["train_time"]], y[sp["train_time"]], seed)
        mu_t = m_t.predict(X)
        r_t = np.abs(y - mu_t)
        schemes["time"] = (mu_t, r_t, r_t[sp["cal_time"]], sp["stest_time"])
    q = {s: float(conformal_q(v[2])) for s, v in schemes.items()}
    O["q"] = q
    # realised fractions of the laboratory-grouped (and recent) source split: laboratories
    # are indivisible, so these depart from the nominal 60/20/20 (reported in the .md)
    ns = float(len(src))
    O["split_frac"] = {"row_train": len(sp["train"]) / ns, "row_cal": len(sp["cal"]) / ns,
                       "row_stest": len(sp["stest"]) / ns,
                       "lab_train": len(sp["train_lab"]) / ns, "lab_cal": len(sp["cal_lab"]) / ns,
                       "lab_stest": len(sp["stest_lab"]) / ns,
                       "lab_n_labs_train": int(len(np.unique(lab[sp["train_lab"]]))),
                       "lab_n_labs_cal": int(len(np.unique(lab[sp["cal_lab"]])))}
    if "train_time" in sp:
        O["split_frac"].update(time_train=len(sp["train_time"]) / ns, time_cal=len(sp["cal_time"]) / ns,
                               time_stest=len(sp["stest_time"]) / ns)
    O["rmse"] = {"source_cal": rmse(r[sp["cal"]]), "source_test": rmse(r[sp["stest"]]),
                 "target_all": rmse(r[tgt]), "target_test_a": rmse(r[sp["ttest_a"]]),
                 "target_test_b": rmse(r[sp["ttest_b"]]) if len(sp["ttest_b"]) else None,
                 "lab_model_source_test": rmse(r_l[sp["stest_lab"]]), "lab_model_target_all": rmse(r_l[tgt])}
    if "time" in schemes:
        O["rmse"]["time_model_target_all"] = rmse(r_t[tgt])
    O["id"] = {s: cblock(v[3], v[0], v[2], y, X, iqr_S) for s, v in schemes.items()}
    b_ok = len(sp["ttest_b"]) >= 20 and len(sp["rpool_b_spread"]) >= 10
    O["b_variant_available"] = float(b_ok)
    evalsets = {"all": tgt, "test_a": sp["ttest_a"]}
    if b_ok:
        evalsets["test_b"] = sp["ttest_b"]
    for es, idx in evalsets.items():
        O[f"tgt_{es}"] = {s: cblock(idx, v[0], v[2], y, X, iqr_T) for s, v in schemes.items()}

    # ---- (iii) target recalibration, variants (a) and (b)
    pools = {"a": (sp["rpool_a"], sp["ttest_a"])}
    if b_ok:
        pools.update(b_spread=(sp["rpool_b_spread"], sp["ttest_b"]),
                     b_clustered=(sp["rpool_b_clustered"], sp["ttest_b"]))
    for v, (pool, te) in pools.items():
        mm = min(mh, len(pool))
        blk = {"m_used": mm, "pool_size": int(len(pool)),
               "n_labs_in_recal_m": int(len(np.unique(lab[pool[:mm]]))),
               "m": cblock(te, mu, r[pool[:mm]], y, X, iqr_T),
               "ksweep": {str(k): kentry(r[sp["cal"]] if k == 0 else r[pool[:k]], r[te], iqr_T, y[te], mu[te])
                          for k in KGRID if k <= len(pool)}}
        qm = conformal_q(r[pool[:mm]])
        blk["width_ratio_m_vs_row"] = float(qm / q["row"]) if np.isfinite(qm) else None
        O[f"recal_{v}"] = blk
    q_m_a = conformal_q(r[sp["rpool_a"][:mh]])
    O["width_ratio"] = {"lab_vs_row": q["lab"] / q["row"], "recal_a_m_vs_row": float(q_m_a / q["row"])}
    if "time" in schemes:
        O["width_ratio"]["time_vs_row"] = q["time"] / q["row"]

    # ---- locally adaptive (normalised) score: width-stratified coverage, triage, abstention
    se = sd + 1e-6
    te = sp["ttest_a"]
    pa = sp["rpool_a"][:mh]
    qn_src = conformal_q(r[sp["cal"]] / se[sp["cal"]])
    qn_m = conformal_q(r[pa] / se[pa])
    lo_m, hi_m = mu[te] - se[te] * qn_m, mu[te] + se[te] * qn_m
    rc_id = risk_coverage(r[sp["stest"]], sd[sp["stest"]])
    rc_t = risk_coverage(r[te], sd[te])
    O["normalised"] = {
        "source_cal_on_test_a": interval_metrics(mu[te] - se[te] * qn_src, mu[te] + se[te] * qn_src, y[te], iqr_T),
        "recal_a_m": interval_metrics(lo_m, hi_m, y[te], iqr_T),
        "abstain_recal_a_m_vs_id_width": abstention_rate(hi_m - lo_m, 2 * q["row"]),
        "triage_id": {k: rc_id[k] for k in ("aurc_rel", "reduction_at50_pct", "rmse_all", "rmse_at50")},
        "triage_target_test_a": {k: rc_t[k] for k in ("aurc_rel", "reduction_at50_pct", "rmse_all", "rmse_at50")}}

    # ---- covariate-shift correction (weighted conformal) and decomposition (A2 2.5)
    rc = r[sp["cal"]]
    auc, wc, wt = domain(X[sp["cal"]], X[tgt], seed)
    ess = ess_frac(wc)
    rm_cal, rm_t = rmse(rc), rmse(r[tgt])
    rm_rw = float(np.sqrt(np.sum(wc * rc ** 2) / np.sum(wc)))
    # Covariate share of the excess mean squared error. Three exclusion flags are
    # recorded separately and all of them are reported in the .md, because two of
    # them (no excess error; reweighted residuals exceeding the target residuals)
    # are selections on the outcome and would otherwise bias the reported share
    # downwards.  "ess_only" keeps the two filters that do not condition on the
    # size of the estimate (ESS/n >= 0.05, positive excess error).
    ex_ess = float(ess < 0.05)
    ex_noexc = float(not (rm_t > rm_cal))
    ex_over = float(rm_rw > rm_t)
    raw = (rm_rw ** 2 - rm_cal ** 2) / (rm_t ** 2 - rm_cal ** 2) if rm_t != rm_cal else None
    share_ess = raw if (not ex_ess and not ex_noexc) else None
    share = share_ess if not ex_over else None
    O["decomp"] = {"domain_auc": auc, "ess_frac": ess, "rmse_source_cal": rm_cal, "rmse_reweighted": rm_rw,
                   "rmse_target": rm_t, "covariate_share": share, "covariate_share_ess_only": share_ess,
                   "excl_ess": ex_ess, "excl_no_excess": ex_noexc, "excl_reweighted_over_target": ex_over,
                   "cov_source_cal_reweighted": float(np.sum(wc * (rc <= q["row"])) / np.sum(wc)),
                   "cov_target_row": float(np.mean(r[tgt] <= q["row"]))}
    qw = wcp_q(rc, wc, wt)
    pos_all = np.arange(len(tgt))
    pos_te = np.arange(len(sp["rpool_a"]), len(tgt))
    O["weighted"] = {}
    for es, pos in (("all", pos_all), ("test_a", pos_te)):
        idx, qq = tgt[pos], qw[pos]
        fin = np.isfinite(qq)
        O["weighted"][es] = {"coverage": float(np.mean(r[idx] <= qq)), "frac_infinite": float(1 - fin.mean()),
                             "width_finite": float(np.mean(2 * qq[fin])) if fin.any() else None,
                             "norm_width_finite": float(np.mean(2 * qq[fin]) / iqr_T) if fin.any() else None}

    # ---- identical-composition cross-lab check and novel-composition sensitivity
    O["identical"] = match_stats(sp["train"], tgt, D, r, q["row"])
    nov = ~np.isin(g[te], g[src])
    O["novel_comp_test_a"] = {"frac_novel": float(nov.mean()), "n_novel": int(nov.sum())}
    if nov.sum() >= 10:
        O["novel_comp_test_a"].update(cov_row=float(np.mean(r[te][nov] <= q["row"])),
                                      cov_lab=float(np.mean(r_l[te][nov] <= q["lab"])),
                                      cov_recal_a_m=float(np.mean(r[te][nov] <= q_m_a)))

    # ---- per-laboratory conditional coverage
    covmap = {s: v[1] <= q[s] for s, v in schemes.items()}
    covmap["recal_a_m"] = r <= q_m_a
    pl_te = per_lab(sp["ttest_a"], lab, covmap)
    pl_all = per_lab(tgt, lab, {s: covmap[s] for s in schemes})
    pl_te10 = per_lab(sp["ttest_a"], lab, covmap, min_n=10)
    pl_all10 = per_lab(tgt, lab, {s: covmap[s] for s in schemes}, min_n=10)
    O["per_lab_test_a"] = per_lab_summary(pl_te, list(covmap))
    O["per_lab_all"] = per_lab_summary(pl_all, list(schemes))
    O["per_lab_test_a_min10"] = per_lab_summary(pl_te10, list(covmap))
    O["per_lab_all_min10"] = per_lab_summary(pl_all10, list(schemes))
    # ---- (iv) lab-specific recalibration
    sch = {s: (v[1], q[s]) for s, v in schemes.items()}
    sch["recal_a_m"] = (r, q_m_a)
    ls = lab_specific(tgt, sp["rpool_a"][:mh], sch, r, lab, g, np.random.default_rng(50000 + seed))
    O["lab_specific"] = per_lab_summary(ls, ["lab10"] + list(sch))
    if ls:
        O["lab_specific"]["width_lab10_over_row_median"] = float(np.median([x["width_lab10"] for x in ls]) / (2 * q["row"]))
        O["lab_specific"]["lab10_p10_over_draws_median"] = float(np.median([x["lab10_p10_over_draws"] for x in ls]))
    O["_lists"] = {"per_lab_test_a": pl_te, "per_lab_all": pl_all, "per_lab_test_a_min10": pl_te10,
                   "per_lab_all_min10": pl_all10, "lab_specific": ls}
    return O


# ============================================================ (R) reference random rows
def reference_rows(prop, seed):
    """Ungrouped random-row 60/20/20 split of the whole property set (NOT the protocol
    split; used only to measure intra- vs inter-laboratory error at identical
    composition and the optimism of ungrouped splitting)."""
    D = get_data(prop)
    X, y = D["X"], D["y"]
    n = len(y)
    perm = np.random.default_rng(70000 + seed).permutation(n)
    tr, ca, te = perm[:int(0.6 * n)], perm[int(0.6 * n):int(0.8 * n)], perm[int(0.8 * n):]
    m = fit_rf(X[tr], y[tr], seed)
    r = np.abs(y - m.predict(X))
    q = conformal_q(r[ca])
    out = {"seed": seed, "coverage_test": float(np.mean(r[te] <= q)), "rmse_test": rmse(r[te]),
           "width": float(2 * q), "frac_test_comp_in_train_1e-6": float(np.mean(np.isin(D["g"][te], D["g"][tr])))}
    out["identical"] = match_stats(tr, te, D, r, q)
    return out


# ============================================================ (D) P-shift, publication-grouped
def pshift_pub_split(D, seed, proxy):
    """Target (P design region) halves grouped by publication; recal pool ordered
    spread (composition-grouped permutation of the pool) or clustered (publication
    order); test records sharing a composition with the pool dropped."""
    ds = D["ds"]
    gp = D["pub"] if proxy == "kod" else D["pub_ay"]
    rng = np.random.default_rng(30000 + seed)
    T, gT = _perm_grouped(ds.tgt, gp, rng)
    h = _cut_nearest(T, gT, 0.5)
    pool_c, test = T[:h], T[h:]
    n_drop = int(np.isin(D["g"][test], D["g"][pool_c]).sum())
    test = _drop_overlap(test, pool_c, D["g"])
    pool_s = _perm_grouped(pool_c, D["g"], rng)[0]
    check_disjoint({"rpool": pool_s, "ttest": test}, D["g"])
    check_disjoint({"rpool": pool_s, "ttest": test}, gp)
    return {"rpool_spread": pool_s, "rpool_clustered": pool_c, "ttest": test, "_n_drop": n_drop}


def pshift_eval(prop, seed, pub_splits):
    D = get_data(prop)
    ds = D["ds"]
    X, y, g = D["X"], D["y"], D["g"]
    sp = design_split(ds, seed)                       # identical to R1_core
    m = make_model("RF", seed).set_params(n_jobs=TREE_JOBS).fit(X[sp["train"]], y[sp["train"]])
    mu = rf_mean_std(m, X)[0]
    r = np.abs(y - mu)
    rc = r[sp["cal"]]
    mh = headline_m(ds)
    iqr_T = iqr(y[ds.tgt])
    O = {"seed": seed, "m": mh}
    O["comp_grouped"] = {"source_cal": cblock(sp["ttest"], mu, rc, y, X, iqr_T),
                         "recal_m": cblock(sp["ttest"], mu, r[sp["rpool"][:mh]], y, X, iqr_T),
                         "n_pool": int(len(sp["rpool"])), "n_test": int(len(sp["ttest"]))}
    for proxy, ps in pub_splits.items():
        gp = D["pub"] if proxy == "kod" else D["pub_ay"]
        te = ps["ttest"]
        res = {"n_pubs_target": int(len(np.unique(gp[ds.tgt]))), "n_pool": int(len(ps["rpool_clustered"])),
               "n_test": int(len(te)), "n_pubs_pool": int(len(np.unique(gp[ps["rpool_clustered"]]))),
               "n_pubs_test": int(len(np.unique(gp[te]))), "n_test_dropped_comp_overlap": ps["_n_drop"],
               "compgrouped_share_test_rows_pub_in_pool": float(np.mean(np.isin(gp[sp["ttest"]], gp[sp["rpool"]]))),
               "compgrouped_share_test_rows_pub_in_recal_m": float(np.mean(np.isin(gp[sp["ttest"]], gp[sp["rpool"][:mh]]))),
               "source_cal": cblock(te, mu, rc, y, X, iqr_T)}
        for order in ("spread", "clustered"):
            pool = ps[f"rpool_{order}"]
            mm = min(mh, len(pool))
            b = cblock(te, mu, r[pool[:mm]], y, X, iqr_T)
            b["m_used"] = mm
            b["n_pubs_in_recal"] = int(len(np.unique(gp[pool[:mm]])))
            res[f"recal_m_{order}"] = b
        O[f"pub_{proxy}"] = res
    if "kod" in pub_splits and "author_year" in pub_splits:
        a, b = pub_splits["kod"], pub_splits["author_year"]
        same = (len(a["ttest"]) == len(b["ttest"]) and np.array_equal(np.sort(a["ttest"]), np.sort(b["ttest"]))
                and len(a["rpool_clustered"]) == len(b["rpool_clustered"])
                and np.array_equal(a["rpool_clustered"], b["rpool_clustered"]))
        O["proxy_compare"] = {
            "splits_identical": float(same),
            "n_groups_kod": int(len(np.unique(D["pub"][ds.tgt]))),
            "n_groups_author_year": int(len(np.unique(D["pub_ay"][ds.tgt]))),
            "abs_diff_spread": abs(O["pub_kod"]["recal_m_spread"]["coverage"]
                                   - O["pub_author_year"]["recal_m_spread"]["coverage"]),
            "abs_diff_clustered": abs(O["pub_kod"]["recal_m_clustered"]["coverage"]
                                      - O["pub_author_year"]["recal_m_clustered"]["coverage"])}
    return O


# ============================================================ (E) 45S5 Bioglass
def bioglass_index(D, tol=0.01):
    Cp = D["compounds"]
    m = np.ones(len(Cp), bool)
    for j, (ox, v) in enumerate(BIOGLASS.items()):
        m &= np.abs(Cp[:, j] - v) <= tol
    m &= Cp.sum(1) >= 0.98
    return np.where(m)[0]


def nominal_45s5(feat):
    atoms = {"Si": 0.461, "Na": 2 * 0.244, "Ca": 0.269, "P": 2 * 0.026,
             "O": 2 * 0.461 + 0.244 + 0.269 + 5 * 0.026}
    tot = sum(atoms.values())
    x = np.zeros(len(feat))
    for e, v in atoms.items():
        x[feat.index(e)] = v / tot
    return x[None, :]


def bioglass_split(D, seed, idx45):
    """Tg set minus the 45S5 window; composition-grouped 60/20/20 (row-cal) and
    laboratory-grouped 60/20/20 (lab-cal) of the remainder."""
    g, lab = D["g"], D["lab"]
    keep = np.setdiff1d(np.arange(len(D["y"])), idx45)
    keep = keep[~np.isin(g[keep], g[idx45])]
    rng = np.random.default_rng(80000 + seed)
    S, gS = _perm_grouped(keep, g, rng)
    a, b = _cut(S, gS, 0.6), _cut(S, gS, 0.8)
    L, gL = _perm_grouped(keep, lab, rng)
    a2 = _cut_nearest(L, gL, 0.6)
    b2 = _cut_nearest(L, gL, 0.8, min_pos=a2 + 1)
    trl, cal = L[:a2], _drop_overlap(L[a2:b2], L[:a2], g)
    sp = {"train": S[:a], "cal": S[a:b], "stest": S[b:], "train_lab": trl, "cal_lab": cal, "bioglass_45S5": idx45}
    check_disjoint({k: sp[k] for k in ("train", "cal", "stest", "bioglass_45S5")}, g)
    check_disjoint({k: sp[k] for k in ("train_lab", "cal_lab", "bioglass_45S5")}, g)
    return sp


def bioglass_eval(seed, sp):
    D = get_data("Tg")
    X, y = D["X"], D["y"]
    idx = sp["bioglass_45S5"]
    x0 = nominal_45s5(D["ds"].feat)
    out = {"seed": seed}
    for s, (tr, ca) in {"row": ("train", "cal"), "lab": ("train_lab", "cal_lab")}.items():
        m = fit_rf(X[sp[tr]], y[sp[tr]], seed)
        q = conformal_q(np.abs(y[sp[ca]] - m.predict(X[sp[ca]])))
        p = m.predict(X[idx])
        p0 = float(m.predict(x0)[0])
        # how far the prediction at the nominal composition is from the training data:
        # the +-1 mol% window is removed, but chemically similar glasses are not
        d1 = np.abs(X[sp[tr]] - x0).sum(1)
        out[s + "_train_distance"] = {
            "nearest_L1_atomic_fraction": float(d1.min()),
            "n_train_L1_le_0.02": int((d1 <= 0.02).sum()), "n_train_L1_le_0.05": int((d1 <= 0.05).sum()),
            "n_train_L1_le_0.10": int((d1 <= 0.10).sum()), "n_train": int(len(d1)),
            "y_mean_of_train_L1_le_0.05": float(y[sp[tr]][d1 <= 0.05].mean()) if (d1 <= 0.05).any() else None}
        out[s] = {"q": float(q), "width": float(2 * q), "pred_nominal": p0,
                  "pred_records_mean": float(p.mean()), "pred_records_sd": float(p.std(ddof=1)),
                  "coverage_records": float(np.mean(np.abs(y[idx] - p) <= q)),
                  "coverage_records_nominal_centre": float(np.mean(np.abs(y[idx] - p0) <= q)),
                  "bias_mean_y_minus_pred": float(np.mean(y[idx] - p)),
                  "rmse_records": rmse(y[idx] - p)}
    return out


def bioglass_describe(D, idx):
    y, lab, yr = D["y"][idx], D["lab"][idx], D["year"][idx]
    lm = pd.Series(y).groupby(lab).mean()
    mad = np.median(np.abs(y - np.median(y)))
    recs = sorted([{"author": str(D["author"][i]), "lab_key": str(D["lab"][i]), "year": int(D["year"][i]),
                    "Tg_K": float(D["y"][i]), "analysed": bool(D["chem"][i]), "kod": int(D["pub"][i]),
                    "sciglass_id": int(D["ids"][i]),
                    "SiO2_Na2O_CaO_P2O5": [round(float(v), 4) for v in D["compounds"][i]]} for i in idx],
                  key=lambda d: (d["year"], d["Tg_K"]))
    idx2 = bioglass_index(D, 0.02)
    return {"rule": "|x_ox - x_ref| <= 0.01 mole fraction for SiO2 0.461, Na2O 0.244, CaO 0.269, P2O5 0.026; "
                    "sum of the four >= 0.98 (as A2)",
            "n_records_within_2molpct": int(len(idx2)),
            "n_records_within_2molpct_outside_window": int(len(np.setdiff1d(idx2, idx))),
            "n_records": int(len(idx)), "n_labs": int(len(np.unique(lab))), "n_publications": int(len(np.unique(D["pub"][idx]))),
            "n_analysed": int(D["chem"][idx].sum()), "n_distinct_Tg_values": int(len(np.unique(y))),
            "year_min": int(yr.min()), "year_max": int(yr.max()),
            "Tg_min_K": float(y.min()), "Tg_median_K": float(np.median(y)), "Tg_max_K": float(y.max()),
            "Tg_p05_p10_p90_p95_K": [float(v) for v in np.percentile(y, [5, 10, 90, 95])],
            "Tg_sd_K": float(y.std(ddof=1)), "Tg_robust_sd_K_1.4826MAD": float(1.4826 * mad),
            "sd_of_lab_means_K": float(lm.std(ddof=1)), "n_labs_with_ge2_records": int((pd.Series(lab).value_counts() >= 2).sum()),
            "records": recs}


# ============================================================ aggregation
def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if str(k).startswith("_"):
            continue
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, p))
        elif isinstance(v, (bool, np.bool_)):
            out[p] = float(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[p] = float(v)
    return out


def agg(rows):
    flat = [flatten(r) for r in rows]
    keys = list(dict.fromkeys(k for f in flat for k in f))
    S = {}
    for k in keys:
        vals = [f.get(k) for f in flat]
        s = summarize(vals)
        if s["n"]:
            a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
            s["min"], s["max"] = float(a.min()), float(a.max())
            s["median"] = float(np.median(a))
            if k.endswith("coverage"):
                s["frac_below_0.90"] = float(np.mean(a < 0.90))
        S[k] = s
    return S, flat


def pooled_per_repeat(rows, list_key, methods, n_per_repeat=N_FOLDS):
    """Lab-out: every lab is held out once per repeat -> pool per-lab rows per repeat."""
    reps = {}
    for r in rows:
        reps.setdefault(r["seed"] // n_per_repeat, []).extend(r["_lists"][list_key])
    per = {rep: per_lab_summary(v, methods) for rep, v in sorted(reps.items())}
    out = {"n_labs": summarize([p["n_labs"] for p in per.values()])}
    for mth in methods:
        out[mth] = {"median": summarize([p[mth]["median"] for p in per.values() if mth in p]),
                    "share_below_0.8": summarize([p[mth]["share_below_0.8"] for p in per.values() if mth in p])}
    if list_key != "lab_specific":
        out["expected_share_below_0.8_if_exact"] = summarize([p["expected_share_below_0.8_if_exact"] for p in per.values()])
    return out


def pooled_coverage_per_repeat(rows, block="tgt_all", schemes=("row", "lab")):
    """Lab-out: pooled coverage of all records of a repeat (each record held out once)."""
    reps = {}
    for r in rows:
        rep = r["seed"] // N_FOLDS
        for s in schemes:
            c, n = r[block][s]["coverage"], r[block][s]["n_eval"]
            a = reps.setdefault(rep, {}).setdefault(s, [0.0, 0])
            a[0] += c * n
            a[1] += n
    return {s: summarize([reps[k][s][0] / reps[k][s][1] for k in reps]) for s in schemes}


# ============================================================ splits persistence
def save_family(tag, D, splits, protocol, extra=None):
    ds = D["ds"]
    path = os.path.join(SPLITS, f"{tag}_splits.json")
    summ = {"name": ds.name, "N": int(len(ds.y)), "n_features": int(ds.X.shape[1]),
            "n_unique_compositions": int(len(np.unique(ds.groups))), "unit": ds.unit,
            "source": "SciGlass via glasspy (bioactive subset Si, Ca, Na > 0)",
            "index_space": "row positions of common.glass_dataset(prop) arrays (SciGlass order, property non-missing); "
                           "sciglass_id gives the SciGlass ID of every row",
            "laboratory_key": "normalised first author (A2 section 2.1)"}
    obj = {"dataset": summ, "protocol": protocol,
           "splits": {str(s): {k: np.asarray(v).tolist() for k, v in sp.items() if not k.startswith("_")}
                      for s, sp in splits.items()},
           "sciglass_id": D["ids"].tolist()}
    if extra:
        obj.update(extra)
    with open(path, "w") as fh:
        json.dump(obj, fh)
    return path


# ============================================================ driver
def run_task(kind, prop, e, sp):
    t = time.time()
    if kind in ("A", "B", "C"):
        out = evaluate_shift(prop, sp, e, kind)
    elif kind == "R":
        out = reference_rows(prop, e)
    elif kind == "D":
        out = pshift_eval(prop, e, sp)
    elif kind == "E":
        out = bioglass_eval(e, sp)
    else:
        raise ValueError(kind)
    out["_time_s"] = time.time() - t
    return kind, prop, e, out


def describe_prop(D):
    lab = D["lab"]
    vc = pd.Series(lab).value_counts()
    return {"n": int(len(D["y"])), "n_labs": int(len(vc)), "n_publications_kod": int(len(np.unique(D["pub"]))),
            "n_unique_compositions_1e-6": int(len(np.unique(D["g"]))), "labs_ge20": int((vc >= MIN_LAB).sum()),
            "largest_lab": [str(vc.index[0]), int(vc.iloc[0])], "year_range": [int(D["year"].min()), int(D["year"].max())],
            "n_analysed": int(D["chem"].sum())}


def describe_shift(D, src, tgt):
    lab, y, g = D["lab"], D["y"], D["g"]
    return {"n_source": int(len(src)), "n_target": int(len(tgt)),
            "n_labs_source": int(len(np.unique(lab[src]))), "n_labs_target": int(len(np.unique(lab[tgt]))),
            "n_pubs_target": int(len(np.unique(D["pub"][tgt]))),
            "target_frac_from_labs_unseen_in_source": float(np.mean(~np.isin(lab[tgt], lab[src]))),
            "target_frac_comp_in_source_1e-6": float(np.mean(np.isin(g[tgt], g[src]))),
            "target_frac_comp_in_source_1e-3": float(np.mean(np.isin(D["key3"][tgt], D["key3"][src]))),
            "y_mean_sd_source": [float(y[src].mean()), float(y[src].std())],
            "y_mean_sd_target": [float(y[tgt].mean()), float(y[tgt].std())],
            "year_median_source_target": [float(np.median(D["year"][src])), float(np.median(D["year"][tgt]))]}


PROTO_SHIFT = make_shift_split.__doc__


def main():
    quick = "--quick" in sys.argv
    if "--md-only" in sys.argv:
        write_md(json.load(open(os.path.join(RES, f"{TAG}.json"))))
        return
    open(LOG, "w").close()
    props = ["Microhardness"] if quick else [p for p, _, _ in GLASS_PROPS]
    seeds = SEEDS[:2] if quick else SEEDS
    tasks, meta = [], {"config": {
        "alpha": ALPHA, "model": "common.make_model('RF', seed): RandomForestRegressor(n_estimators=300, "
                                 "min_samples_leaf=2), n_jobs=1", "kgrid": KGRID, "headline_m": "clip(|T|//3, 10, 30)",
        "n_folds": N_FOLDS, "n_repeats": N_REPEATS, "temporal_cutoffs": TEMPORAL_CUT, "min_lab_records": MIN_LAB,
        "lab_specific_k": LAB_K, "lab_specific_draws_per_evaluation": LAB_DRAWS,
        "identical_composition_rounding": f"1e-{MATCH_DEC}", "wis_alphas": WIS_ALPHAS, "levels": LEVELS.tolist(),
        "domain_classifier": "RandomForestClassifier(200, min_samples_leaf=5, class_weight='balanced'), 5-fold cross-fit, "
                             "p clipped to [0.01, 0.99], w = p/(1-p)",
        "summary_statistics": "mean and 2.5-97.5 percentile across the 20 evaluations (lab-out: 4 repeats x 5 folds; "
                              "other families: 20 seeds). Evaluations resample one finite dataset and are not "
                              "independent samples of the population.",
        "quick": quick}, "data": {}}
    for prop in props:
        D = get_data(prop)
        meta["data"][prop] = {"overall": describe_prop(D)}
        # (A) leave-laboratory-out
        spA, fold_info = {}, []
        for rep in range(N_REPEATS):
            fold = lab_folds(D["lab"], N_FOLDS, np.random.default_rng(9000 + rep))
            fold_info.append({"repeat": rep, "fold_records": np.bincount(fold, minlength=N_FOLDS).tolist(),
                              "fold_labs": [int(len(np.unique(D["lab"][fold == f]))) for f in range(N_FOLDS)]})
            for f in range(N_FOLDS):
                e = rep * N_FOLDS + f
                if quick and e not in seeds:
                    continue
                tgt, src = np.where(fold == f)[0], np.where(fold != f)[0]
                assert not np.intersect1d(D["lab"][tgt], D["lab"][src]).size
                spA[e] = make_shift_split(D, src, tgt, e)
                spA[e]["_fold"] = f
        meta["data"][prop]["A_folds"] = fold_info
        save_family(f"R4_labout_{D['label']}", D, spA,
                    "Leave-laboratory-out, 5 folds of whole labs (balanced by records) x 4 repeats; key e = 5*repeat + fold. "
                    + PROTO_SHIFT)
        tasks += [("A", prop, e, sp) for e, sp in spA.items()]
        # (B) temporal
        yr = D["year"]
        srcB, tgtB = np.where(yr < TEMPORAL_CUT[prop])[0], np.where(yr >= TEMPORAL_CUT[prop])[0]
        meta["data"][prop]["B_temporal"] = describe_shift(D, srcB, tgtB)
        spB = {s: make_shift_split(D, srcB, tgtB, s, time_cal=True) for s in seeds}
        save_family(f"R4_temporal_{D['label']}", D, spB,
                    f"Temporal: source year < {TEMPORAL_CUT[prop]}, target year >= {TEMPORAL_CUT[prop]}. " + PROTO_SHIFT)
        tasks += [("B", prop, s, sp) for s, sp in spB.items()]
        # (C) nominal -> analysed
        srcC, tgtC = np.where(~D["chem"])[0], np.where(D["chem"])[0]
        meta["data"][prop]["C_analysed"] = describe_shift(D, srcC, tgtC)
        spC = {s: make_shift_split(D, srcC, tgtC, s) for s in seeds}
        save_family(f"R4_analysed_{D['label']}", D, spC,
                    "Nominal -> chemically analysed: source ChemicalAnalysis False, target True. " + PROTO_SHIFT)
        tasks += [("C", prop, s, sp) for s, sp in spC.items()]
        # (R) reference
        tasks += [("R", prop, s, None) for s in seeds]
        # (D) P-shift publication-grouped
        spD = {s: {proxy: pshift_pub_split(D, s, proxy) for proxy in ("kod", "author_year")} for s in seeds}
        save_family(f"R4_pshift_pubgrouped_{D['label']}", D,
                    {f"{s}_{proxy}": v for s, d in spD.items() for proxy, v in d.items()},
                    "P design shift (common.design_split source split and target region, seed s, identical to R1). "
                    "Target halves grouped by publication; key '<seed>_<proxy>' with proxy kod (SciGlass ID // 1e8) or "
                    "author_year (Author+Year, the A1 audit proxy). " + (pshift_pub_split.__doc__ or ""))
        tasks += [("D", prop, s, spD[s]) for s in seeds]
    # (E) 45S5
    if "Tg" in props:
        D = get_data("Tg")
        idx45 = bioglass_index(D)
        meta["E_45S5"] = {"description": bioglass_describe(D, idx45)}
        spE = {s: bioglass_split(D, s, idx45) for s in seeds}
        save_family("R4_bioglass45S5_glass_Tg", D, spE, bioglass_split.__doc__)
        tasks += [("E", "Tg", s, sp) for s, sp in spE.items()]
    order = {"A": 0, "B": 1, "C": 2, "D": 3, "R": 4, "E": 5}
    size = {"Tliquidus": 0, "Tg": 1, "YoungModulus": 2, "Microhardness": 3}
    tasks.sort(key=lambda t: (size[t[1]], order[t[0]]))
    log(f"{len(tasks)} tasks; props={props}; n_jobs={N_JOBS}")
    res = {}
    with parallel_config(backend="loky", inner_max_num_threads=1):
        gen = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(delayed(run_task)(*t) for t in tasks)
        for i, (kind, prop, e, out) in enumerate(gen):
            res.setdefault(kind, {}).setdefault(prop, []).append(out)
            if kind in "ABC":
                rb = out.get("recal_b_spread", {}).get("m", {}).get("coverage")
                log(f"{i + 1}/{len(tasks)} {kind} {prop} e{e}: row {out['tgt_test_a']['row']['coverage']:.3f} "
                    f"lab {out['tgt_test_a']['lab']['coverage']:.3f} recal_a {out['recal_a']['m']['coverage']:.3f} "
                    f"recal_b {rb if rb is None else round(rb, 3)} ({out['_time_s']:.0f}s)")
            else:
                log(f"{i + 1}/{len(tasks)} {kind} {prop} e{e} ({out['_time_s']:.0f}s)")
    results = assemble(meta, res, props)
    dump(results, f"{TAG}.json")
    write_md(results)
    log("done")


def assemble(meta, res, props):
    R = dict(meta)
    fam_names = {"A": "A_labout", "B": "B_temporal", "C": "C_analysed"}
    for kind, fname in fam_names.items():
        R[fname] = {}
        for prop in props:
            rows = sorted(res[kind][prop], key=lambda o: o["seed"])
            S, flat = agg(rows)
            methods = [s for s in ("row", "lab", "time") if s in rows[0]["q"]]
            blk = {"n_evaluations": len(rows), "summary": S, "per_eval": flat,
                   "per_lab_lists": [{"seed": r["seed"], **{k: v for k, v in r["_lists"].items()}} for r in rows],
                   "ksweep_beta_theory_5_50_95": {str(k): beta_coverage_quantiles(k) for k in KGRID if k > 0}}
            if kind == "A":
                blk["pooled_per_repeat"] = {
                    "coverage_all_records": pooled_coverage_per_repeat(rows),
                    "per_lab_all": pooled_per_repeat(rows, "per_lab_all", methods),
                    "per_lab_test_a": pooled_per_repeat(rows, "per_lab_test_a", methods + ["recal_a_m"]),
                    "lab_specific": pooled_per_repeat(rows, "lab_specific", ["lab10"] + methods + ["recal_a_m"])}
            R[fname][prop] = blk
    R["reference_random_rows"] = {}
    for prop in props:
        rows = sorted(res["R"][prop], key=lambda o: o["seed"])
        R["reference_random_rows"][prop] = {"summary": agg(rows)[0]}
    R["D_pshift_pubgrouped"] = {}
    r1 = json.load(open(os.path.join(RES, "R1_core.json")))
    for prop in props:
        rows = sorted(res["D"][prop], key=lambda o: o["seed"])
        S, flat = agg(rows)
        label = PROP_INFO[prop][0]
        r1d = r1.get(label, {})
        r1_split = {p["seed"]: p for p in r1d.get("per_split", [])}
        repro = [abs(o["comp_grouped"]["recal_m"]["coverage"] - r1_split[o["seed"]]["recal_cov"])
                 for o in rows if o["seed"] in r1_split]
        R["D_pshift_pubgrouped"][prop] = {
            "summary": S, "per_eval": flat,
            "R1_core_reference": {"shift_recal_coverage": r1d.get("shift_recal", {}).get("coverage"),
                                  "shift_source_coverage": r1d.get("shift_source", {}).get("coverage"),
                                  "shift_recal_norm_width": r1d.get("shift_recal", {}).get("norm_width"),
                                  "m": r1d.get("m")},
            "max_abs_diff_vs_R1_per_split_recal_cov": float(max(repro)) if repro else None}
    if "E" in res:
        rows = sorted(res["E"]["Tg"], key=lambda o: o["seed"])
        R["E_45S5"]["model"] = agg(rows)[0]
    return R


# ============================================================ markdown
def _f(s, key="mean", nd=3):
    if s is None or s.get(key) is None:
        return "–"
    return f"{s[key]:.{nd}f}"


def _ci(s, nd=3):
    if s is None or s.get("mean") is None:
        return "–"
    return f"{s['mean']:.{nd}f} [{s['lo']:.{nd}f}, {s['hi']:.{nd}f}]"


def _g(S, key):
    return S.get(key)


def write_md(R):
    L = []
    P = [p for p in ("Tg", "YoungModulus", "Microhardness", "Tliquidus") if p in R.get("A_labout", {})]
    U = {p: PROP_INFO[p][1] for p in P}
    nd_u = {"Tg": 1, "YoungModulus": 2, "Microhardness": 3, "Tliquidus": 1}
    A, B, Cc = R.get("A_labout", {}), R.get("B_temporal", {}), R.get("C_analysed", {})
    L.append("# R4_realshift — genuine (non-constructed) shifts in SciGlass\n")
    L.append("Script `revision/code/exp_R4_realshift.py`; numbers from `revision/results/R4_realshift.json`. "
             "Reviewer 1 comment 2; Reviewer 2 comments 3 and 8. Protocol: `revision/code/PROTOCOL.md` "
             "(α = 0.1, finite-sample split-conformal quantile, grouped splits, 60/20/20 source, target halves "
             "= recalibration pool / target-test). Model: RF (`common.make_model('RF')`, 300 trees, "
             "min_samples_leaf = 2). Values are mean [2.5–97.5 percentile] over 20 evaluations "
             "(leave-lab-out: 4 repeats × 5 folds; other families: 20 seeds). The evaluations resample one finite "
             "dataset; the percentile range describes split-to-split variability, not population sampling error. "
             "'Laboratory' = normalised first-author key (A2 §2.1), a proxy.\n")
    # ---------------- headline statements
    L.append("## 0. Bottom line\n")
    L += headline_statements(R, P)
    # ---------------- data
    L.append("\n## 1. Data and shift definitions\n")
    L.append("| Property | n | labs | publications (Kod) | labs ≥ 20 rec. | largest lab | temporal cutoff: n source / target (target labs; share from labs unseen before) | analysed: n (labs; share from labs with no nominal record) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for p in P:
        d = R["data"][p]
        o, t, c = d["overall"], d["B_temporal"], d["C_analysed"]
        L.append(f"| {p} | {o['n']} | {o['n_labs']} | {o['n_publications_kod']} | {o['labs_ge20']} | {o['largest_lab'][0]} ({o['largest_lab'][1]}) | "
                 f"≥ {R['config']['temporal_cutoffs'][p]}: {t['n_source']} / {t['n_target']} ({t['n_labs_target']}; {t['target_frac_from_labs_unseen_in_source']:.2f}) | "
                 f"{c['n_target']} ({c['n_labs_target']}; {c['target_frac_from_labs_unseen_in_source']:.2f}) |")
    L.append("")
    L.append("Leave-lab-out fold sizes (records per fold, min–max over folds and repeats): " + "; ".join(
        f"{p} {min(min(fi['fold_records']) for fi in R['data'][p]['A_folds'])}–{max(max(fi['fold_records']) for fi in R['data'][p]['A_folds'])}"
        for p in P) + ".\n")
    L += scheme_definitions(R, P)
    # ---------------- family tables
    for fam, title, F in (("A", "2. PRIMARY — leave-laboratory-out (repeated grouped 5-fold, 4 repeats)", A),
                          ("B", "3. SECONDARY — temporal (future publications)", B),
                          ("C", "4. TERTIARY — nominal → chemically analysed composition (protocol + laboratory shift)", Cc)):
        if not F:
            continue
        L.append(f"\n## {title}\n")
        L += family_tables(fam, F, P, nd_u, U)
    # ---------------- reference
    if R.get("reference_random_rows"):
        L.append("\n## 5. Identical compositions: intra- versus inter-laboratory error\n")
        L.append("Reference = ungrouped random-row 60/20/20 split (not the protocol split; used only here). Records whose composition "
                 "(1e-3 atomic fraction) occurs in the training set, split by whether a training record of that composition comes from "
                 "the same laboratory. 'Lookup' = |y − mean y of the matching training records| (model-free). Leave-lab-out matches are "
                 "cross-laboratory by construction.\n")
        L.append("| Property | random rows: same-lab match n / RMSE model / RMSE lookup | random rows: cross-lab-only match n / RMSE model / RMSE lookup | random rows: matched / unmatched RMSE | lab-out: matched frac / RMSE model / RMSE lookup / row-cal cov. | lab-out: unmatched RMSE / cov. | lab-out ÷ random rows, RMSE: matched / unmatched | implied lab component √(lab-out² − same-lab²) (model) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for p in P:
            S = R["reference_random_rows"][p]["summary"]
            SA = A[p]["summary"]
            nd = nd_u[p]
            sl, cl = _g(S, "identical.same_lab.rmse_model"), _g(SA, "identical.matched.rmse_model")
            rm_ref, ru_ref = _g(S, "identical.matched.rmse_model"), _g(S, "identical.unmatched.rmse_model")
            ru = _g(SA, "identical.unmatched.rmse_model")
            imp = "–"
            if sl and cl and sl["mean"] is not None and cl["mean"] is not None and cl["mean"] > sl["mean"]:
                imp = f"{np.sqrt(cl['mean'] ** 2 - sl['mean'] ** 2):.{nd}f}"
            rat = lambda a, b: f"{a['mean'] / b['mean']:.2f}" if a and b and a.get("mean") and b.get("mean") else "–"
            L.append(f"| {p} | {_f(_g(S,'identical.same_lab.n'),nd=0)} / {_f(sl,nd=nd)} / {_f(_g(S,'identical.same_lab.rmse_lookup'),nd=nd)} | "
                     f"{_f(_g(S,'identical.cross_lab_only.n'),nd=0)} / {_f(_g(S,'identical.cross_lab_only.rmse_model'),nd=nd)} / {_f(_g(S,'identical.cross_lab_only.rmse_lookup'),nd=nd)} | "
                     f"{_f(rm_ref,nd=nd)} / {_f(ru_ref,nd=nd)} | "
                     f"{_f(_g(SA,'identical.frac_matched'))} / {_f(cl,nd=nd)} / {_f(_g(SA,'identical.cross_lab_only.rmse_lookup'),nd=nd)} / {_f(_g(SA,'identical.matched.coverage'))} | "
                     f"{_f(ru,nd=nd)} / {_f(_g(SA,'identical.unmatched.coverage'))} | {rat(cl, rm_ref)} / {rat(ru, ru_ref)} | {imp} |")
        L.append("")
        L.append("Reading the last two columns together: repeating an identical composition in another laboratory already costs a large part of "
                 "the error (same-lab versus cross-lab-only RMSE in the first two columns), but the *shift penalty* of holding out whole "
                 "laboratories is not confined to compositions the model has already seen — relative to the ungrouped random-row reference the "
                 "RMSE rises by a similar factor on matched and on unmatched records (ratios above). Inter-laboratory disagreement at fixed "
                 "composition and genuine novelty therefore contribute comparably; § 0 states the claim in that form.\n")
        L.append("Ungrouped random-row reference, for comparison with the composition-grouped source-test: " + "; ".join(
            f"{p} coverage {_f(_g(R['reference_random_rows'][p]['summary'],'coverage_test'))}, RMSE {_f(_g(R['reference_random_rows'][p]['summary'],'rmse_test'),nd=nd_u[p])}"
            for p in P) + ".\n")
    # ---------------- P-shift pub grouped
    if R.get("D_pshift_pubgrouped"):
        L.append("\n## 6. SENSITIVITY — phosphorus design shift (R1) with publication-grouped recalibration\n")
        L.append("Source split, model and target region identical to `R1_core` (`common.design_split`, same seeds; the composition-grouped "
                 "recalibration reproduces R1 per split, see last column). Publication-grouped: the target region is split into halves of whole "
                 "publications; recalibration records (headline m) come from the pool publications and the target-test contains only other "
                 "publications. 'spread' = the m records are spread over the pool's publications; 'clustered' = taken in publication order "
                 "(few publications). Coverage mean [2.5–97.5 pct] (share of splits < 0.90).\n")
        L.append("| Property (m) | target pubs (Kod) | composition-grouped (R1 protocol) | test rows sharing a publication with the recal pool / with the m records | Kod-grouped, spread | Kod-grouped, clustered | Author+Year-grouped, spread | Author+Year-grouped, clustered | source-cal on Kod-grouped test | norm. width comp → Kod spread | max abs. diff. vs R1 per split |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for p in P:
            DD = R["D_pshift_pubgrouped"][p]
            S = DD["summary"]
            def cc(k):
                s = _g(S, k)
                return f"{_ci(s)} ({s['frac_below_0.90']:.2f})" if s and s.get("mean") is not None else "–"
            L.append(f"| {p} ({_f(_g(S,'m'),nd=0)}) | {_f(_g(S,'pub_kod.n_pubs_target'),nd=0)} | {cc('comp_grouped.recal_m.coverage')} | "
                     f"{_f(_g(S,'pub_kod.compgrouped_share_test_rows_pub_in_pool'),nd=2)} / {_f(_g(S,'pub_kod.compgrouped_share_test_rows_pub_in_recal_m'),nd=2)} | "
                     f"{cc('pub_kod.recal_m_spread.coverage')} | {cc('pub_kod.recal_m_clustered.coverage')} | "
                     f"{cc('pub_author_year.recal_m_spread.coverage')} | {cc('pub_author_year.recal_m_clustered.coverage')} | "
                     f"{_ci(_g(S,'pub_kod.source_cal.coverage'))} | {_f(_g(S,'comp_grouped.recal_m.norm_width'),nd=2)} → {_f(_g(S,'pub_kod.recal_m_spread.norm_width'),nd=2)} | "
                     f"{DD['max_abs_diff_vs_R1_per_split_recal_cov'] if DD['max_abs_diff_vs_R1_per_split_recal_cov'] is not None else float('nan'):.2g} |")
        L.append("")
        L += pshift_proxy_and_a1(R, P)
    # ---------------- 45S5
    if R.get("E_45S5"):
        E = R["E_45S5"]
        d = E["description"]
        L.append("\n## 7. 45S5 Bioglass illustration (Tg)\n")
        L.append(f"Rule: {d['rule']}.\n")
        L.append("| n records | labs | publications | analysed | years | Tg min / median / max (K) | 5–95 % (K) | SD (K) | robust SD 1.4826·MAD (K) | SD of lab means (K) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        pc = d["Tg_p05_p10_p90_p95_K"]
        L.append(f"| {d['n_records']} | {d['n_labs']} | {d['n_publications']} | {d['n_analysed']} | {d['year_min']}–{d['year_max']} | "
                 f"{d['Tg_min_K']:.1f} / {d['Tg_median_K']:.1f} / {d['Tg_max_K']:.1f} | {pc[0]:.1f}–{pc[3]:.1f} | {d['Tg_sd_K']:.1f} | "
                 f"{d['Tg_robust_sd_K_1.4826MAD']:.1f} | {d['sd_of_lab_means_K']:.1f} |")
        L.append("")
        if "model" in E:
            S = E["model"]
            L.append("Model (20 seeds): the RF is trained without any record of the ±1 mol% window **and** without any record sharing a "
                     "composition group with it, so none of the 45S5 records above is in the training set. This is **not** an extrapolation "
                     f"test: glasses chemically close to 45S5 but outside the window remain in the training data "
                     f"({d['n_records_within_2molpct_outside_window']} further Tg records lie within ±2 mol% per oxide, and the nearest "
                     f"training composition is {_f(_g(S,'row_train_distance.nearest_L1_atomic_fraction'),nd=3)} away in summed |Δ atomic "
                     f"fraction|, with {_f(_g(S,'row_train_distance.n_train_L1_le_0.05'),nd=0)} training records within 0.05). The table "
                     "gives the source-calibrated 90 % interval at the 45S5 composition. Two coverages are reported: 'own-composition "
                     "prediction' centres each record's interval on the model's prediction for **that record's** composition (the records "
                     "differ slightly within the window), 'nominal-centred' centres every interval on the prediction at the nominal "
                     "46.1–24.4–26.9–2.6 composition.\n")
            L.append("| Calibration | prediction at nominal 45S5 (K) | interval half-width q (K) | interval at nominal (K) | coverage of the 45S5 records (own-composition prediction) | coverage, nominal-centred | mean y − prediction (K) | RMSE on records (K) |")
            L.append("|---|---|---|---|---|---|---|---|")
            for s, name in (("row", "row-cal (i.i.d.)"), ("lab", "lab-cal (whole labs)")):
                pn, qq = _g(S, f"{s}.pred_nominal"), _g(S, f"{s}.q")
                L.append(f"| {name} | {_ci(pn,1)} | {_ci(qq,1)} | {pn['mean']-qq['mean']:.1f}–{pn['mean']+qq['mean']:.1f} | "
                         f"{_ci(_g(S,f'{s}.coverage_records'))} | {_ci(_g(S,f'{s}.coverage_records_nominal_centre'))} | "
                         f"{_ci(_g(S,f'{s}.bias_mean_y_minus_pred'),1)} | {_ci(_g(S,f'{s}.rmse_records'),1)} |")
            L.append("")
            L.append(f"For comparison, ±1.645 × SD = ±{1.645*d['Tg_sd_K']:.1f} K; ±1.645 × robust SD = ±{1.645*d['Tg_robust_sd_K_1.4826MAD']:.1f} K; "
                     f"±1.645 × SD of lab means = ±{1.645*d['sd_of_lab_means_K']:.1f} K.\n")
    L.append("\n## 8. Notes and deviations\n")
    L += notes()
    L.append("\n## 9. Verifier issues and how they were handled\n")
    L += verifier_log()
    with open(os.path.join(RES, f"{TAG}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def scheme_definitions(R, P):
    """Explicit definition of the four calibration schemes compared in sections 2-4,
    including the fact that lab-cal and recent-cal also refit the model, and the
    realised (not nominal) source fractions of the laboratory-grouped split."""
    fams = [R.get(k, {}) for k in ("A_labout", "B_temporal", "C_analysed")]

    def rng(key, nd=2):
        v = []
        for F in fams:
            for p in F:
                s = F[p]["summary"].get(key)
                if s and s.get("min") is not None:
                    v += [s["min"], s["max"]]
        return f"{min(v):.{nd}f}–{max(v):.{nd}f}" if v else "–"

    big = max(((R["data"][p]["overall"]["largest_lab"][1] / R["data"][p]["overall"]["n"], p,
                R["data"][p]["overall"]["largest_lab"][0], R["data"][p]["overall"]["largest_lab"][1],
                R["data"][p]["overall"]["n"]) for p in P), default=None)
    L = ["**Calibration schemes compared in §§ 2–4.** All of them use α = 0.1 and the finite-sample "
         "split-conformal quantile; they differ in which records calibrate it and, for (ii) and (ii-t), also in "
         "which records train the model.\n",
         "* **(i) row-cal** — the usual i.i.d. practice. One RF is trained on the composition-grouped 60 % of the "
         "source pool; the conformal quantile is taken from the composition-grouped 20 % source-calibration split. "
         "Every number labelled 'row-cal', every k-sweep and every target recalibration in (iii) and (iv) uses "
         "**this** model; only the calibration residuals change.",
         "* **(ii) lab-cal** — the source pool is re-split 60/20/20 by *whole laboratories*. This arm therefore "
         "**refits the model** (a second RF on the laboratory-grouped 60 %) as well as changing the calibration "
         "set, so the lab-cal coverages and the width ratio (ii)/(i) mix a calibration effect with a model effect "
         "and are not a pure calibration comparison. Because laboratories are indivisible the realised fractions "
         f"depart from 60/20/20: train {rng('split_frac.lab_train')}, calibration {rng('split_frac.lab_cal')}, "
         f"source-test {rng('split_frac.lab_stest')} of the source pool (min–max over all evaluations and "
         "properties; the row-cal split is exact by construction"
         + (f", and the most unbalanced property is {big[1]}, whose largest laboratory {big[2]} alone holds {big[3]} of "
            f"{big[4]} records)." if big else ").")
         + f" The laboratory-grouped calibration set contains {rng('split_frac.lab_n_labs_cal', 0)} laboratories.",
         "* **(ii-t) recent-cal** (temporal family only) — trains on the earliest 60 % of the source by year and "
         "calibrates on the latest 20 %; the middle 20 % is its source-test. It also refits the model. Realised "
         f"fractions: train {rng('split_frac.time_train')}, calibration {rng('split_frac.time_cal')}.",
         "* **(iii) target recalibration (recal k / recal m)** — the row-cal model with the conformal quantile "
         "recomputed from k (or the headline m) labelled records of the target recalibration pool.",
         "* **(iv) lab-specific recalibration** — the row-cal model with the quantile recomputed from k = 10 "
         "records of the *same* held-out laboratory as the evaluated records.\n"]
    return L


def a1_verdict(prop, c_comp, c_clustered):
    """Verdict of the (D) arm that corresponds to the split audit A1 section 6.3."""
    if prop not in A1_PUBRECAL or c_comp is None or c_clustered is None:
        return None
    d_r4 = c_comp - c_clustered
    d_a1 = A1_PUBRECAL[prop][0] - A1_PUBRECAL[prop][1]
    if d_r4 < 0.02:
        return "no material drop under this protocol"
    if abs(d_r4 - d_a1) <= 0.01:
        return "direction confirmed, magnitude comparable"
    return "direction confirmed, magnitude milder" if d_r4 < d_a1 else "direction confirmed, magnitude larger"


def pshift_proxy_and_a1(R, P):
    """(D) two paragraphs the table cannot carry: (1) whether the Kod and Author+Year
    columns are really two independent proxies, (2) the explicit verdict on the A1
    audit's microhardness finding (0.94 -> 0.80)."""
    D = R["D_pshift_pubgrouped"]
    L = []
    ident, differ = [], []
    for p in P:
        S = D[p]["summary"]
        si = _g(S, "proxy_compare.splits_identical")
        if si is None or si.get("mean") is None:
            continue
        nk = _f(_g(S, "proxy_compare.n_groups_kod"), nd=0)
        na = _f(_g(S, "proxy_compare.n_groups_author_year"), nd=0)
        n_ev = len(D[p]["per_eval"])
        n_same = int(round(si["mean"] * n_ev))
        if n_same == n_ev:
            ident.append(f"{p} ({nk} groups under both proxies, identical target halves in {n_same}/{n_ev} seeds)")
        else:
            ds = _g(S, "proxy_compare.abs_diff_spread")
            dc = _g(S, "proxy_compare.abs_diff_clustered")
            mk, ma = _g(S, "pub_kod.recal_m_spread.coverage"), _g(S, "pub_author_year.recal_m_spread.coverage")
            differ.append(f"{p} (identical in {n_same}/{n_ev} seeds; {nk} Kod groups vs {na} Author+Year groups; "
                          f"per-split max |Kod − Author+Year| coverage difference {_f(ds,'max',3)} spread, "
                          f"{_f(dc,'max',3)} clustered; the two 'spread' means are {_f(mk,nd=4)} and {_f(ma,nd=4)}, "
                          f"equal only after rounding to three decimals)")
    if ident or differ:
        s = "**The two publication proxies are not always independent.** "
        if ident:
            s += ("For " + "; ".join(ident) + " — the Kod and Author+Year partitions of the design region are the *same* "
                  "partition, so the corresponding column pairs above are identical by construction, not by coincidence. ")
        if differ:
            s += ("For " + "; ".join(differ) + " — the splits do differ; where the two means nevertheless agree to three "
                  "decimals the per-split differences cancel, which is a coincidence of rounding and not a copy error.")
        L.append(s + "\n")
    rows = [p for p in P if p in A1_PUBRECAL]
    if rows:
        L.append("**Verdict on the split audit A1 (§ 6.3).** A1 reported, for the *submitted* Table-3 protocol (70/30 source, "
                 "Author+Year publication proxy, whole publications drawn into the recalibration pool, 10 seeds), that "
                 "publication-grouped recalibration lowers the recalibrated coverage from " +
                 ", ".join((f"**{p} {A1_PUBRECAL[p][0]:.2f} → {A1_PUBRECAL[p][1]:.2f}**" if p == "Microhardness"
                            else f"{p} {A1_PUBRECAL[p][0]:.2f} → {A1_PUBRECAL[p][1]:.2f}") for p in rows) +
                 ", with the share of splits below 0.90 rising from " +
                 ", ".join(f"{p} {A1_PUBRECAL[p][2]:.0%} → {A1_PUBRECAL[p][3]:.0%}" for p in rows) + ". Because A1 drew "
                 "whole publications into the pool, the arm of this table that corresponds to it is the **clustered** "
                 "ordering (the m records come from the pool's first publications); the 'spread' arm is a weaker, "
                 "more favourable perturbation. Under the unified protocol of this script (composition-grouped source "
                 "60/20/20, 20 seeds, Kod proxy):\n")
        L.append("| Property | R4 composition-grouped | R4 Kod-grouped, clustered (A1's arm) | share of splits < 0.90 (comp. → clustered) | A1 (10 seeds, Author+Year, Table-3 protocol) | verdict |")
        L.append("|---|---|---|---|---|---|")
        verdicts = {}
        for p in rows:
            S = D[p]["summary"]
            c0, c1 = _g(S, "comp_grouped.recal_m.coverage"), _g(S, "pub_kod.recal_m_clustered.coverage")
            if not (c0 and c1 and c0.get("mean") is not None):
                continue
            verd = a1_verdict(p, c0["mean"], c1["mean"])
            verdicts[p] = verd
            L.append(f"| {p} | {_ci(c0)} | {_ci(c1)} | {c0['frac_below_0.90']:.2f} → {c1['frac_below_0.90']:.2f} | "
                     f"{A1_PUBRECAL[p][0]:.3f} → {A1_PUBRECAL[p][1]:.3f} ({A1_PUBRECAL[p][3]:.0%} of splits < 0.90) | {verd} |")
        S = D.get("Microhardness", {}).get("summary")
        if S and "Microhardness" in verdicts:
            c0, c1 = _g(S, "comp_grouped.recal_m.coverage"), _g(S, "pub_kod.recal_m_clustered.coverage")
            cs = _g(S, "pub_kod.recal_m_spread.coverage")
            L.append("")
            L.append(f"For the property A1 singled out, microhardness, the answer is therefore: "
                     f"**{verdicts['Microhardness']}.** Here {_ci(c0)} → {_ci(c1)} clustered ({c1['frac_below_0.90']:.0%} of splits "
                     f"below 0.90) and {_ci(cs)} spread ({cs['frac_below_0.90']:.0%} below 0.90), against A1's "
                     f"0.94 → 0.80 with 80 % of splits below 0.90. A1 used the submitted 70/30 source split, m = 23 and "
                     f"10 seeds, whereas this table uses the unified protocol, so the two magnitudes are not expected to "
                     f"match exactly. "
                     + ("The qualitative conclusion is A1's: recalibration records drawn from the same publications as "
                        "the evaluated records overstate the recovery.\n"
                        if verdicts["Microhardness"].startswith("direction confirmed")
                        else "Under the unified protocol A1's effect does not reproduce at this magnitude; the number "
                             "above, not A1's, is the one to quote for this protocol.\n"))
    return L


def family_tables(fam, F, P, nd_u, U):
    L = []
    has_time = fam == "B"
    # main table
    L.append("**Accuracy and marginal coverage on the target-test half (a) (same records for every method).** Coverage mean [2.5–97.5 pct]; "
             "normalised width = mean width / IQR of the target property values.\n")
    hdr = ("| Property | RMSE source-test → target-test | ID coverage row-cal (source-test) | (i) row-cal cov. | (ii) lab-cal cov. |"
           + (" (ii-t) recent-cal cov. |" if has_time else "") +
           " weighted conformal cov. (inf. share) | (iii-a) recal m cov. | norm. width (i) / (ii) / (iii-a) | width ratio (ii)/(i), (iii-a)/(i) |")
    L.append(hdr)
    L.append("|" + "---|" * (hdr.count("|") - 1))
    for p in P:
        S = F[p]["summary"]
        nd = nd_u[p]
        row = (f"| {p} (m={_f(_g(S,'m'),nd=0)}) | {_f(_g(S,'rmse.source_test'),nd=nd)} → {_f(_g(S,'rmse.target_test_a'),nd=nd)} {U[p]} | "
               f"{_ci(_g(S,'id.row.coverage'))} | {_ci(_g(S,'tgt_test_a.row.coverage'))} | {_ci(_g(S,'tgt_test_a.lab.coverage'))} | ")
        if has_time:
            row += f"{_ci(_g(S,'tgt_test_a.time.coverage'))} | "
        row += (f"{_ci(_g(S,'weighted.test_a.coverage'))} ({_f(_g(S,'weighted.test_a.frac_infinite'),nd=2)}) | {_ci(_g(S,'recal_a.m.coverage'))} | "
                f"{_f(_g(S,'tgt_test_a.row.norm_width'),nd=2)} / {_f(_g(S,'tgt_test_a.lab.norm_width'),nd=2)} / {_f(_g(S,'recal_a.m.norm_width'),nd=2)} | "
                f"{_f(_g(S,'width_ratio.lab_vs_row'),nd=2)}, {_f(_g(S,'width_ratio.recal_a_m_vs_row'),nd=2)} |")
        L.append(row)
    L.append("")
    if fam == "A":
        L.append("Pooled over all records of a repeat (each record held out exactly once per repeat; mean [min–max-like 2.5–97.5 pct] over 4 repeats): " + "; ".join(
            f"{p} row-cal {_ci(F[p]['pooled_per_repeat']['coverage_all_records']['row'])}, lab-cal {_ci(F[p]['pooled_per_repeat']['coverage_all_records']['lab'])}"
            for p in P) + ".\n")
    # interval score / WIS / CE / worst cluster
    L.append("**Further metrics on target-test (a)** (interval score and WIS in property units; CE = mean |empirical − nominal| over levels 0.50–0.95; "
             "worst-cluster = lowest coverage among k-means clusters of the evaluated compositions).\n")
    L.append("| Property | method | interval score | WIS | CE mean | worst-cluster cov. |")
    L.append("|---|---|---|---|---|---|")
    meths = [("row", "tgt_test_a.row"), ("lab", "tgt_test_a.lab")] + ([("recent", "tgt_test_a.time")] if has_time else []) + [("recal m (a)", "recal_a.m")]
    for p in P:
        S = F[p]["summary"]
        nd = nd_u[p]
        for name, k in meths:
            L.append(f"| {p} | {name} | {_f(_g(S,k+'.interval_score'),nd=nd)} | {_f(_g(S,k+'.wis'),nd=nd)} | {_f(_g(S,k+'.ce_mean'))} | {_f(_g(S,k+'.worst_cluster_cov'))} |")
    L.append("")
    # k-sweep
    L.append("**Target recalibration k-sweep on the target-test half.** (a) record-level halves (composition-grouped); (b) laboratory-grouped halves "
             "(recalibration and test records from different laboratories), recal records spread over the pool's labs or clustered "
             "(laboratory order). Coverage mean [2.5–97.5 pct]; normalised width in parentheses. k = 0 is the row-calibrated source quantile.\n")
    ks = [str(k) for k in KGRID]
    L.append("| Property | variant | " + " | ".join(f"k={k}" for k in ks) + " | headline m |")
    L.append("|---|---|" + "---|" * (len(ks) + 1))
    for p in P:
        S = F[p]["summary"]
        for v, name in (("a", "(a) records"), ("b_spread", "(b) labs, spread"), ("b_clustered", "(b) labs, clustered")):
            cells = []
            for k in ks:
                c = _g(S, f"recal_{v}.ksweep.{k}.coverage")
                w = _g(S, f"recal_{v}.ksweep.{k}.norm_width")
                cells.append(f"{_ci(c)} ({_f(w, nd=2)})" if c else "–")
            c, w = _g(S, f"recal_{v}.m.coverage"), _g(S, f"recal_{v}.m.norm_width")
            fb = f"; <0.90 in {c['frac_below_0.90']:.2f}" if c and c.get("mean") is not None else ""
            L.append(f"| {p} | {name} | " + " | ".join(cells) + f" | {_ci(c)} ({_f(w, nd=2)}{fb}) |")
    L.append("")
    L.append("Pool sizes (records; labs in pool / test, mean): " + "; ".join(
        f"{p} (a) {_f(_g(F[p]['summary'],'sizes.rpool_a'),nd=0)}, (b) {_f(_g(F[p]['summary'],'sizes.rpool_b_spread'),nd=0)} "
        f"({_f(_g(F[p]['summary'],'n_labs.rpool_b'),nd=0)} / {_f(_g(F[p]['summary'],'n_labs.ttest_b'),nd=0)} labs), "
        f"labs in the m clustered records {_f(_g(F[p]['summary'],'recal_b_clustered.n_labs_in_recal_m'),nd=1)}" for p in P) + ".\n")
    # per-lab
    L.append("**Per-laboratory conditional coverage** (laboratories with ≥ 20 records in the evaluated set; median over labs / share of labs "
             "with coverage < 0.80; mean over evaluations). 'exact' = share expected < 0.80 if every lab were covered at exactly 0.90 "
             "(binomial, given its n).\n")
    ms = ["row", "lab"] + (["time"] if has_time else [])
    L.append("| Property | set | labs | " + " | ".join(f"{m} median / <0.8" for m in ms) + " | recal m (a) median / <0.8 | exact <0.8 |")
    L.append("|---|---|---|" + "---|" * (len(ms) + 2))
    for p in P:
        S = F[p]["summary"]
        for st, name in (("per_lab_all", "whole target, ≥ 20"), ("per_lab_test_a", "target-test (a), ≥ 20"),
                         ("per_lab_all_min10", "whole target, ≥ 10"), ("per_lab_test_a_min10", "target-test (a), ≥ 10")):
            cells = [f"{_f(_g(S,f'{st}.{m}.median'),nd=2)} / {_f(_g(S,f'{st}.{m}.share_below_0.8'),nd=2)}" for m in ms]
            rc = f"{_f(_g(S,f'{st}.recal_a_m.median'),nd=2)} / {_f(_g(S,f'{st}.recal_a_m.share_below_0.8'),nd=2)}" if st.startswith("per_lab_test_a") else "–"
            L.append(f"| {p} | {name} | {_f(_g(S,f'{st}.n_labs'),nd=1)} | " + " | ".join(cells) + f" | {rc} | {_f(_g(S,f'{st}.expected_share_below_0.8_if_exact'),nd=2)} |")
    L.append("")
    if fam == "A":
        L.append("Pooled per repeat (every held-out lab once per repeat; mean over 4 repeats), whole target: " + "; ".join(
            f"{p} ({_f(F[p]['pooled_per_repeat']['per_lab_all']['n_labs'],nd=0)} labs) row {_f(F[p]['pooled_per_repeat']['per_lab_all']['row']['median'],nd=2)} / "
            f"{_f(F[p]['pooled_per_repeat']['per_lab_all']['row']['share_below_0.8'],nd=2)}, lab-cal {_f(F[p]['pooled_per_repeat']['per_lab_all']['lab']['median'],nd=2)} / "
            f"{_f(F[p]['pooled_per_repeat']['per_lab_all']['lab']['share_below_0.8'],nd=2)} (exact {_f(F[p]['pooled_per_repeat']['per_lab_all']['expected_share_below_0.8_if_exact'],nd=2)})"
            for p in P) + ".\n")
    # lab-specific
    L.append("**(iv) Lab-specific recalibration** ('a new laboratory measures 10 glasses'): per target laboratory with ≥ 20 records, "
             f"k = {LAB_K} of its own records (composition-grouped; {LAB_DRAWS} draws per evaluation) recalibrate the row-cal model; coverage on "
             "the lab's remaining records (excluding the fold-level recalibration records). All methods on the same records. "
             "Median over labs / share of labs < 0.80.\n")
    L.append("| Property | labs | lab-specific k=10 | " + " | ".join(ms) + " | recal m (a) | width lab-specific / row-cal (median) | lab-specific p10 over draws (median lab) |")
    L.append("|---|---|---|" + "---|" * (len(ms) + 3))
    for p in P:
        S = F[p]["summary"]
        cells = [f"{_f(_g(S,f'lab_specific.{m}.median'),nd=2)} / {_f(_g(S,f'lab_specific.{m}.share_below_0.8'),nd=2)}" for m in ["lab10"] + ms + ["recal_a_m"]]
        L.append(f"| {p} | {_f(_g(S,'lab_specific.n_labs'),nd=1)} | " + " | ".join(cells) +
                 f" | {_f(_g(S,'lab_specific.width_lab10_over_row_median'),nd=2)} | {_f(_g(S,'lab_specific.lab10_p10_over_draws_median'),nd=2)} |")
    L.append("")
    # decomposition
    L.append("**Covariate versus label/measurement shift** (A2 §2.5). Source-cal residual RMSE → the same residuals density-ratio reweighted to the "
             "target compositions (cross-fitted RF domain classifier) → actual target RMSE. Covariate share = (rw² − cal²)/(target² − cal²), a rough "
             "indicator. **Full exclusion rule** (the trimmed column): a split is dropped if (1) ESS/n < 0.05, (2) the target RMSE does not exceed "
             "the source-cal RMSE (no excess error to decompose), or (3) the reweighted RMSE exceeds the target RMSE (A2's caveat that the "
             "indicator is then meaningless). Rules (2) and (3) condition on the outcome, so the 'ESS-filter only' column (rules 1 and 2, i.e. no "
             "selection on the size of the estimate) and the per-reason exclusion counts are given alongside; the trimmed column can only be the "
             "lower of the two. A split may fail more than one condition, so the three counts do not add up to the number of splits removed "
             "(compare the n of the two share columns). 'Reweighted cov.' = coverage the row-cal quantile would have if only covariates "
             "shifted; 'actual' = row-cal coverage on the whole target.\n")
    L.append("| Property | domain AUC | ESS/n | RMSE cal → reweighted → target | covariate share, trimmed (n) | covariate share, ESS filter only: mean / median / max (n) | splits excluded: ESS / no excess / rw > target | reweighted cov. → actual cov. | novel-composition subset of target-test (a): share / row-cal / lab-cal / recal m |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    over = []
    for p in P:
        S = F[p]["summary"]
        nd = nd_u[p]
        n_ev = F[p].get("n_evaluations", 20)
        eo = _g(S, "decomp.excl_reweighted_over_target")
        n_over = int(round((eo["mean"] if eo and eo.get("mean") is not None else 0) * n_ev))
        n_ess = int(round((_g(S, "decomp.excl_ess") or {"mean": 0})["mean"] * n_ev))
        n_nox = int(round((_g(S, "decomp.excl_no_excess") or {"mean": 0})["mean"] * n_ev))
        eso = _g(S, "decomp.covariate_share_ess_only")
        if n_over >= n_ev / 2:
            over.append((p, n_over, n_ev))
        L.append(f"| {p} | {_f(_g(S,'decomp.domain_auc'))} | {_f(_g(S,'decomp.ess_frac'),nd=2)} | "
                 f"{_f(_g(S,'decomp.rmse_source_cal'),nd=nd)} → {_f(_g(S,'decomp.rmse_reweighted'),nd=nd)} → {_f(_g(S,'decomp.rmse_target'),nd=nd)} | "
                 f"{_f(_g(S,'decomp.covariate_share'),nd=2)} (n={(_g(S,'decomp.covariate_share') or {}).get('n', 0)} of {n_ev}) | "
                 f"{_f(eso,nd=2)} / {_f(eso,'median',2)} / {_f(eso,'max',2)} (n={(eso or {}).get('n', 0)} of {n_ev}) | "
                 f"{n_ess} / {n_nox} / {n_over} | "
                 f"{_f(_g(S,'decomp.cov_source_cal_reweighted'))} → {_f(_g(S,'decomp.cov_target_row'))} | "
                 f"{_f(_g(S,'novel_comp_test_a.frac_novel'),nd=2)} / {_f(_g(S,'novel_comp_test_a.cov_row'))} / {_f(_g(S,'novel_comp_test_a.cov_lab'))} / {_f(_g(S,'novel_comp_test_a.cov_recal_a_m'))} |")
    L.append("")
    if over:
        L.append("Where the trimmed column rests on few splits, it is **not** evidence that the covariate share is small: " + "; ".join(
            f"for {p} the reweighted residuals *exceed* the target residuals in {n} of {tot} splits, so the indicator is uninformative there "
            f"(reweighting over-explains the excess error rather than under-explaining it)" for p, n, tot in over) +
            ". Such properties are excluded from any range quoted in § 0 and the reason is stated there.\n")
    L.append("Identical compositions (1e-3) in the target whose composition occurs in the source training set: " + "; ".join(
        f"{p} share {_f(_g(F[p]['summary'],'identical.frac_matched'),nd=3)} (cross-lab-only {_f(_g(F[p]['summary'],'identical.frac_cross_lab_only_match'),nd=3)}), "
        f"RMSE matched {_f(_g(F[p]['summary'],'identical.matched.rmse_model'),nd=nd_u[p])} vs unmatched {_f(_g(F[p]['summary'],'identical.unmatched.rmse_model'),nd=nd_u[p])} {U[p]}, "
        f"row-cal cov. matched {_f(_g(F[p]['summary'],'identical.matched.coverage'))}" for p in P) + ".\n")
    # selective / abstention
    L.append("**Selective prediction and locally adaptive intervals** (normalised score |r|/RF tree SD; target-test (a)): RMSE reduction on "
             "the 50 % most confident (ID → target), width-stratified coverage (worst width quartile) and abstention rate (share of "
             "target candidates whose recalibrated adaptive interval is wider than the in-distribution conformal width).\n")
    L.append("| Property | RMSE reduction @50 % ID → target | adaptive source-cal cov. | adaptive recal m cov. | worst width-quartile cov. (recal) | abstention |")
    L.append("|---|---|---|---|---|---|")
    for p in P:
        S = F[p]["summary"]
        L.append(f"| {p} | {_f(_g(S,'normalised.triage_id.reduction_at50_pct'),nd=1)} % → {_f(_g(S,'normalised.triage_target_test_a.reduction_at50_pct'),nd=1)} % | "
                 f"{_ci(_g(S,'normalised.source_cal_on_test_a.coverage'))} | {_ci(_g(S,'normalised.recal_a_m.coverage'))} | "
                 f"{_f(_g(S,'normalised.recal_a_m.wsc_width_strat'))} | {_f(_g(S,'normalised.abstain_recal_a_m_vs_id_width'),nd=2)} |")
    L.append("")
    return L


def headline_statements(R, P):
    L = []
    fam = [("A", "A_labout", "held-out laboratories"), ("B", "B_temporal", "future publications (temporal)"),
           ("C", "C_analysed", "chemically analysed compositions")]

    def rng_str(vals, nd=2):
        v = [x for x in vals if x is not None]
        return f"{min(v):.{nd}f}–{max(v):.{nd}f}" if v else "–"

    for code, key, name in fam:
        F = R.get(key, {})
        if not F:
            continue
        g = lambda p, k: (F[p]["summary"].get(k) or {}).get("mean")
        id_c = rng_str([g(p, "id.row.coverage") for p in P])
        row = rng_str([g(p, "tgt_test_a.row.coverage") for p in P])
        lab = rng_str([g(p, "tgt_test_a.lab.coverage") for p in P])
        wcp = rng_str([g(p, "weighted.test_a.coverage") for p in P])
        ra = rng_str([g(p, "recal_a.m.coverage") for p in P])
        rbs = rng_str([g(p, "recal_b_spread.m.coverage") for p in P])
        rbc = rng_str([g(p, "recal_b_clustered.m.coverage") for p in P])
        k10 = rng_str([g(p, "recal_a.ksweep.10.coverage") for p in P])
        wl = rng_str([g(p, "width_ratio.lab_vs_row") for p in P], 1)
        wr = rng_str([g(p, "width_ratio.recal_a_m_vs_row") for p in P], 1)
        rm = "; ".join(f"{p} {g(p,'rmse.source_test'):.3g}→{g(p,'rmse.target_test_a'):.3g}" for p in P)
        s = (f"* **{name.capitalize()}** ({'repeated grouped 5-fold, 20 evaluations' if code == 'A' else '20 seeds'}). "
             f"In-distribution row-cal coverage {id_c}; on the target-test half row-cal (i.i.d.) coverage falls to {row} across the four properties "
             f"(RMSE source-test→target-test: {rm}). Laboratory-grouped calibration gives {lab} at {wl}× the row-cal width")
        if code == "B":
            s += f"; recent-period calibration gives {rng_str([g(p, 'tgt_test_a.time.coverage') for p in P])}"
        inf_max = max([x for x in (g(p, "weighted.test_a.frac_infinite") for p in P) if x is not None] or [0.0])
        s += (f". Weighted (covariate-shift) conformal gives {wcp} (infinite weighted quantiles are counted as covered; "
              f"their share reaches {inf_max:.2f}, so the finite-interval coverage is lower than these numbers). "
              f"Target recalibration with the headline m = 30 target records restores {ra} "
              f"(k = 10: {k10}) at {wr}× the row-cal width when recalibration and test records come from the same target pool (a); "
              f"when they come from different laboratories (b) coverage is {rbs} (records spread over the pool's labs) and {rbc} "
              f"(records taken in laboratory order, i.e. from the pool's first few laboratories).")
        worst = min(P, key=lambda p: g(p, "recal_b_spread.m.coverage") if g(p, "recal_b_spread.m.coverage") is not None else 9)
        wv = g(worst, "recal_b_spread.m.coverage")
        if wv is not None and wv < 0.85:
            s += (f" Recalibration does not transfer across laboratories for {worst} (b, spread: {wv:.2f}; clustered: "
                  f"{g(worst, 'recal_b_clustered.m.coverage'):.2f}), whose target contains few laboratories.")
        L.append(s)
        ls = rng_str([g(p, "lab_specific.lab10.median") for p in P])
        lsb = rng_str([g(p, "lab_specific.lab10.share_below_0.8") for p in P])
        lr = rng_str([g(p, "lab_specific.row.median") for p in P])
        lrb = rng_str([g(p, "lab_specific.row.share_below_0.8") for p in P])
        auc = rng_str([g(p, "decomp.domain_auc") for p in P])
        ess = rng_str([g(p, "decomp.ess_frac") for p in P])
        nlab = rng_str([g(p, "lab_specific.n_labs") for p in P], 1)
        n_ev = F[P[0]]["n_evaluations"]
        need = max(1, n_ev // 2)          # a property is quoted only if half its splits are usable
        cs_t = [g(p, "decomp.covariate_share") for p in P
                if (F[p]["summary"].get("decomp.covariate_share") or {}).get("n", 0) >= need]
        cs_u = [g(p, "decomp.covariate_share_ess_only") for p in P          # same property set as cs_t
                if (F[p]["summary"].get("decomp.covariate_share") or {}).get("n", 0) >= need]
        drop = []
        for p in P:
            d = F[p]["summary"].get("decomp.covariate_share") or {}
            if d.get("n", 0) >= need:
                continue
            cnt = {k: int(round(((F[p]["summary"].get(f"decomp.excl_{k}") or {}).get("mean") or 0) * n_ev))
                   for k in ("ess", "no_excess", "reweighted_over_target")}
            parts = ([f"{cnt['ess']} for ESS/n < 0.05"] if cnt["ess"] else []) + \
                    ([f"{cnt['no_excess']} for no excess error"] if cnt["no_excess"] else []) + \
                    ([f"{cnt['reweighted_over_target']} because the reweighted residuals exceed the target residuals"]
                     if cnt["reweighted_over_target"] else [])
            drop.append(f"{p} (only {d.get('n', 0)} of {n_ev} splits usable" +
                        (": " + "; ".join(parts) + ")" if parts else ")"))
        L.append(f"  Per laboratory (≥ 20 records in the whole target fold, {nlab} laboratories per evaluation, evaluated on the records not "
                 f"used for that laboratory's own recalibration): row-cal median coverage {lr} with {lrb} of labs < 0.80; lab-specific "
                 f"recalibration on 10 of the lab's own glasses gives median {ls} with {lsb} of labs < 0.80. Decomposition: domain-classifier "
                 f"AUC {auc} (genuine covariate shift), ESS/n {ess}, but density-ratio reweighting of the source residuals explains a covariate "
                 f"share of only {rng_str([c for c in cs_u if c is not None])} of the excess mean squared error under the disclosed exclusion "
                 f"rule (ESS/n ≥ 0.05 and a positive excess error; {rng_str([c for c in cs_t if c is not None])} if splits whose reweighted "
                 f"residuals exceed the target residuals are additionally discarded — that is a selection on the outcome, so the first range is "
                 f"the one to quote)"
                 + (f"; not quoted because fewer than {need} of {n_ev} splits are usable: {'; '.join(drop)}"
                    + ("  — where the reason is that the reweighted residuals exceed the target residuals, reweighting "
                       "*over*-explains the excess error, which is not evidence of a small covariate share"
                       if any("reweighted residuals exceed" in x for x in drop) else "") if drop else "")
                 + ". The remainder is a laboratory/measurement (label) shift.")
    RR = R.get("reference_random_rows", {})
    if RR and R.get("A_labout"):
        seg, rat = [], []
        for p in P:
            sr = RR[p]["summary"]
            sa = R["A_labout"][p]["summary"]
            sl = (sr.get("identical.same_lab.rmse_model") or {}).get("mean")
            cl = (sr.get("identical.cross_lab_only.rmse_model") or {}).get("mean")
            lo = (sa.get("identical.matched.rmse_model") or {}).get("mean")
            rm_ref = (sr.get("identical.matched.rmse_model") or {}).get("mean")
            ru_ref = (sr.get("identical.unmatched.rmse_model") or {}).get("mean")
            ru = (sa.get("identical.unmatched.rmse_model") or {}).get("mean")
            if sl and cl and lo:
                u = PROP_INFO[p][1]
                nd = 1 if u == "K" else 2
                seg.append(f"{p} RMSE {sl:.{nd}f} (same lab) vs {cl:.{nd}f} (other lab) vs {lo:.{nd}f} (held-out lab) {u}, implied "
                           f"laboratory component {np.sqrt(max(lo ** 2 - sl ** 2, 0)):.{nd}f} {u}")
            if rm_ref and ru_ref and lo and ru:
                rat.append(f"{p} {lo / rm_ref:.2f} vs {ru / ru_ref:.2f}")
        if seg:
            L.append("* **Identical compositions measured elsewhere** (model error on records whose exact composition, to 1e-3 atomic "
                     "fraction, is in the training set; ungrouped random-row reference for the same-laboratory case): " + "; ".join(seg) +
                     ". Repeating a composition in another laboratory therefore costs about as much as the average prediction error itself. "
                     "It is **not**, however, the whole story: relative to the ungrouped random-row reference, holding out whole laboratories "
                     "raises the RMSE by a similar factor on records whose composition the model has seen and on records whose composition it "
                     "has not (lab-out ÷ random-row ratio, matched vs unmatched: " + "; ".join(rat) + "), so inter-laboratory disagreement and "
                     "genuine novelty contribute comparably to the shift penalty (§ 5).")
    D = R.get("D_pshift_pubgrouped", {})
    if D:
        g = lambda p, k: (D[p]["summary"].get(k) or {}).get("mean")
        L.append("* **Phosphorus design shift, publication-grouped recalibration** (20 seeds): source-calibrated coverage on R1's own "
                 "composition-grouped target-test (the uncorrected constructed shift, identical to `R1_core`) "
                 + ", ".join(f"{p} {g(p,'comp_grouped.source_cal.coverage'):.3f}" for p in P)
                 + "; composition-grouped recalibration "
                 + ", ".join(f"{p} {g(p,'comp_grouped.recal_m.coverage'):.3f}" for p in P)
                 + "; publication (Kod)-grouped, spread "
                 + ", ".join(f"{p} {g(p,'pub_kod.recal_m_spread.coverage'):.3f}" for p in P)
                 + "; clustered "
                 + ", ".join(f"{p} {g(p,'pub_kod.recal_m_clustered.coverage'):.3f}" for p in P)
                 + "; Author+Year-grouped (A1 proxy), clustered "
                 + ", ".join(f"{p} {g(p,'pub_author_year.recal_m_clustered.coverage'):.3f}" for p in P)
                 + ". The audit A1 reported microhardness 0.94 → 0.80 under publication grouping (10 seeds, submitted protocol); "
                 + (f"here the corresponding (clustered) arm gives {g('Microhardness','comp_grouped.recal_m.coverage'):.3f} → "
                    f"{g('Microhardness','pub_kod.recal_m_clustered.coverage'):.3f}: "
                    f"{a1_verdict('Microhardness', g('Microhardness','comp_grouped.recal_m.coverage'), g('Microhardness','pub_kod.recal_m_clustered.coverage'))} (§ 6)."
                    if "Microhardness" in D else ""))
    E = R.get("E_45S5")
    if E:
        d = E["description"]
        s = (f"* **45S5 Bioglass**: {d['n_records']} Tg records from {d['n_labs']} laboratories ({d['year_min']}–{d['year_max']}); "
             f"Tg {d['Tg_min_K']:.0f}–{d['Tg_max_K']:.0f} K (median {d['Tg_median_K']:.0f} K), SD {d['Tg_sd_K']:.1f} K, robust SD "
             f"{d['Tg_robust_sd_K_1.4826MAD']:.1f} K, SD of laboratory means {d['sd_of_lab_means_K']:.1f} K.")
        if "model" in E:
            S = E["model"]
            s += (f" A random forest trained without any window record (and without any record sharing their composition group, but with "
                  f"chemically similar glasses outside the ±1 mol% window still in the training set — {d['n_records_within_2molpct_outside_window']} "
                  f"further Tg records lie within ±2 mol% per oxide, so this is not an extrapolation test) predicts "
                  f"{S['row.pred_nominal']['mean']:.0f} K at the nominal composition. The row-calibrated 90 % interval is "
                  f"±{S['row.q']['mean']:.0f} K and covers {S['row.coverage_records']['mean']:.2f} of the records when each interval is centred "
                  f"on the model's prediction for that record's own composition ({S['row.coverage_records_nominal_centre']['mean']:.2f} when "
                  f"every interval is centred on the nominal composition); the laboratory-calibrated interval is ±{S['lab.q']['mean']:.0f} K and "
                  f"covers {S['lab.coverage_records']['mean']:.2f} ({S['lab.coverage_records_nominal_centre']['mean']:.2f} nominal-centred). "
                  f"For comparison 1.645 × SD = {1.645*d['Tg_sd_K']:.0f} K and 1.645 × robust SD = {1.645*d['Tg_robust_sd_K_1.4826MAD']:.0f} K: "
                  f"the i.i.d. interval is barely wider than the inter-laboratory scatter of this one glass.")
        L.append(s)
    return L


def verifier_log():
    """Independent-verification round: what each issue was and where it is now addressed.
    Nothing was rejected; the last item was raised as disclosure-only."""
    return [
        "All issues raised in the verification of the first run were adopted; the section below records where. "
        "There is no 'Verifier issues not adopted' list.\n",
        "1. *The stated covariate-share exclusion rule was not the rule in the code, and the undisclosed part selected on the outcome.* "
        "The caption in §§ 2–4 now states all three conditions, every table gives the untrimmed (ESS-filter-only) mean, median and maximum "
        "and the per-reason exclusion counts, § 0 quotes the untrimmed range, and properties whose reweighted residuals exceed the target "
        "residuals in most splits are named explicitly instead of being shown as a thin 'n = 2 of 20' cell.",
        "2. *The Kod and Author+Year columns of § 6 are not always two independent proxies.* § 6 now states, per property, in how many seeds "
        "the two proxies induce the same target partition (identical columns are then identical by construction) and, where they differ, the "
        "per-split maximum |Kod − Author+Year| coverage difference, so that coinciding means are visibly a coincidence.",
        "3. *The requested 'confirm or correct' verdict on A1's microhardness 0.94 → 0.80 was missing.* § 6 now quotes A1 § 6.3 in full, "
        "identifies the clustered ordering as the arm that corresponds to A1's design, and gives the verdict per property.",
        "4. *The lab-cal arm changes the trained model, not only the calibration set.* § 1 now defines row-cal, lab-cal and recent-cal "
        "explicitly, says that lab-cal and recent-cal refit the model, and gives the realised (not nominal) source fractions.",
        "5. *Weighted-conformal coverages were quoted in § 0 without the infinite-interval caveat.* The § 0 bullets now carry it with the "
        "maximum infinite share of that family.",
        "6. *The claim that inter-laboratory disagreement 'not extrapolation' dominates was not supported by the table under it.* § 5 now "
        "carries the random-row matched and unmatched columns and the lab-out ÷ random-row ratios, and the claim in § 0 is restated in the "
        "form the numbers support (matched and unmatched records suffer a similar shift penalty).",
        "7. *Accuracy slips in the covering summary of the first round* (ranges that ignored a 0.00 cell, a per-laboratory median quoted for "
        "the wrong family, and the phosphorus source-calibrated coverage quoted from the publication-grouped rather than the "
        "composition-grouped test). § 0 now prints R1's own composition-grouped source-calibrated coverage, and all ranges in this document "
        "are generated from the JSON rather than typed.",
        "8. *Two transparency gaps*: the IQR normaliser uses all target labels (now stated in § 8, with the reference to A1 item 11), and the "
        "45S5 model is trained without the window records but not without chemically similar glasses (now stated in § 0 and § 7, with the "
        "nearest training composition and the number of records within ±2 mol%), together with which of the two coverages is being quoted.",
        "9. *Note only (no fix required): several per-laboratory cells rest on one or two laboratories.* The laboratory counts were already "
        "printed; a sentence in § 8 now requires them to be quoted with any median taken from this document.",
    ]


def notes():
    return [
        "* Laboratory = normalised first-author key (A2 §2.1: ASCII-fold, surname + first initial, transliteration folding). First author is a proxy for laboratory; over-merging (e.g. common surnames) makes leave-lab-out more conservative.",
        "* Fold assignment: labs processed roughly largest-first with a U(0.5, 1.5) jitter on size, each to the lightest fold (A2's greedy made repeats nearly identical for the large labs; the jitter makes the 4 repeats differ in which large labs are held out together). Fold sizes are listed in the JSON (`data.<prop>.A_folds`).",
        "* Grouping: every source split and every recalibration/test split is composition-grouped (common.composition_groups, 1e-6) as the protocol requires. The laboratory-grouped source split and the laboratory/publication-grouped target halves additionally drop cal/test records whose composition occurs on the other side (counts in `sizes`). Between source and target the grouping unit is the shift variable (laboratory, period, protocol): identical compositions measured by a source laboratory and a target laboratory are kept, because they are the object of the identical-composition check; the 'novel-composition' column shows coverage on target-test records whose composition never occurs in the source.",
        "* Recalibration uses the row-cal model (trained on the composition-grouped 60 % of the source); only the calibration residuals change. Headline m = clip(|T|/3, 10, 30) = 30 in every family. k = 9 is the smallest finite k at α = 0.1; for 9 ≤ k ≤ 18 the conformal quantile is the maximum of the k residuals.",
        "* The (b) variant keeps recalibration and test records in different laboratories; 'spread' orders the pool by a composition-grouped permutation (the m records come from many pool labs), 'clustered' in laboratory order (the m records then come from the pool's first few laboratories, 3-6 on average; see `recal_b_clustered.n_labs_in_recal_m`).",
        "* Weighted conformal (Tibshirani et al. 2019) uses the same cross-fitted domain classifier as the decomposition (unlabelled target compositions only). An infinite weighted quantile is counted as covered and reported as 'inf. share'; every weighted-conformal coverage in this document, including those in § 0, is therefore an upper bound on the coverage of the finite intervals.",
        "* Covariate-share exclusion rule, in full: a split contributes to the trimmed column only if ESS/n ≥ 0.05, the target RMSE exceeds the source-cal RMSE, and the reweighted RMSE does not exceed the target RMSE. The last two conditions select on the outcome, so the untrimmed (ESS-filter-only) mean, median and maximum and the per-reason exclusion counts are printed beside it in §§ 2–4; § 0 quotes the untrimmed range.",
        "* Calibration schemes: § 1 defines row-cal, lab-cal (which also refits the model on the laboratory-grouped 60 %, so it is not a pure calibration comparison) and recent-cal, and gives the realised source fractions, which depart from 60/20/20 because laboratories are indivisible.",
        "* Normalised widths divide the mean width by the IQR of **all** labels of the evaluated pool, including the target-test labels. This follows PROTOCOL.md and R1 so that the numbers are comparable with R1_core; it is a scale constant, not a fitted quantity, but the split audit A1 (item 11) flagged the practice, so it is stated here. Coverage, width and interval score never use target-test labels except for evaluation.",
        "* Several per-laboratory cells rest on very few laboratories (the laboratory count is printed in every such table, and the ≥ 10-record threshold is reported beside the ≥ 20-record one for that reason). Any per-laboratory median quoted elsewhere must be quoted with its laboratory count; in the temporal and analysed families the target-test halves are often too small for any laboratory to reach 20 records.",
        "* The recent-cal scheme (temporal only) trains on the earliest 60 % of the source by year and calibrates on the latest 20 %; the middle 20 % is its (intermediate-period) test.",
        "* Publications: Kod = SciGlass ID // 1e8 (as instructed). The split audit A1 used Author+Year; both are reported for the P-shift sensitivity.",
        "* Reference random-row split: ungrouped, used only for the intra- versus inter-laboratory comparison and to show the optimism of ungrouped splitting.",
        "* All targets are retrospective literature data: genuine cross-laboratory, cross-time and cross-protocol shifts, but not in-silico → in-vitro/in-vivo shifts.",
    ]


if __name__ == "__main__":
    main()
