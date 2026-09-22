"""Shared, documented protocol for every analysis in the revised manuscript.

One module defines (i) the data loaders and shift definitions, (ii) the split
protocol, (iii) the conformal procedures and (iv) the uncertainty metrics, so
that every figure and table is produced under the same, auditable rules.

Split protocol (tabular design-region shift), per resampling seed s = 0..19
---------------------------------------------------------------------------
    source pool S (conventional region)  -> permuted with seed s
        train        60 %   (model fitting only)
        source-cal   20 %   (split-conformal calibration)
        source-test  20 %   (in-distribution evaluation)
    target pool T (design region)        -> permuted with seed s
        recal pool   first floor(|T|/2)  (target-domain recalibration draws)
        target-test  remaining ceil(|T|/2) (never used for fitting, calibration,
                                            recalibration or budget selection)
    excluded       records in neither S nor T (intermediate region), unused.
All index sets are disjoint by construction and asserted; with ``group=True``
(default) the permutation is done over unique compositions, so replicate
measurements of one composition can never sit on both sides of a split.

Headline recalibration size m = clip(floor(|T|/3), 10, 30) target records drawn
from the recal pool; k-sweeps draw the first k records of the same pool and are
always evaluated on the same target-test half.

Confidence intervals: every quantity is computed per split; we report the mean
over 20 splits and the 2.5-97.5 percentile interval across splits. Because the
splits resample one finite dataset they are not independent samples of the
population; the percentile interval therefore describes split-to-split
variability, not population sampling error (stated in the Methods).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BMEAT = os.path.dirname(REV)
DATA = os.path.join(REV, "data")
DEMO = DATA  # all generated/downloaded datasets live under the package's own data/
RES = os.path.join(REV, "results")
SPLITS = os.path.join(REV, "results", "splits")
for _d in (DATA, RES, SPLITS):
    os.makedirs(_d, exist_ok=True)

ALPHA = 0.10
N_SPLITS = 20
SEEDS = list(range(N_SPLITS))
LEVELS = np.array([0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95])
GLASS_PROPS = [("Tg", "glass_Tg", "K"), ("YoungModulus", "glass_E", "GPa"),
               ("Microhardness", "glass_HV", "GPa"), ("Tliquidus", "glass_Tliq", "K")]


# =============================================================== data
@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feat: list
    shift: np.ndarray                 # scalar design axis (P, Ni, aromaticity ...)
    src: np.ndarray                   # indices of the source (conventional) pool
    tgt: np.ndarray                   # indices of the target (design-region) pool
    unit: str = ""
    groups: np.ndarray | None = None  # composition id (replicates share an id)
    meta: dict = field(default_factory=dict)

    @property
    def excluded(self):
        m = np.ones(len(self.y), bool); m[self.src] = False; m[self.tgt] = False
        return np.where(m)[0]

    def summary(self):
        return {"name": self.name, "N": int(len(self.y)), "n_features": int(self.X.shape[1]),
                "n_source": int(len(self.src)), "n_target": int(len(self.tgt)),
                "n_excluded": int(len(self.excluded)),
                "n_unique_compositions": int(len(np.unique(self.groups))) if self.groups is not None else None,
                "unit": self.unit, **{k: v for k, v in self.meta.items() if np.isscalar(v)}}


def composition_groups(X, decimals=6):
    """Integer id per unique feature vector (replicate measurements share an id)."""
    keys = [hashlib.md5(np.round(r, decimals).tobytes()).hexdigest() for r in np.asarray(X, float)]
    _, inv = np.unique(np.array(keys), return_inverse=True)
    return inv


def _sciglass_frame():
    """Bioactive-system SciGlass rows (Si, Ca, Na > 0) with elements, properties
    and metadata, cached as a pickle (glasspy load is slow)."""
    import pandas as pd
    cache = os.path.join(DATA, "sciglass_bioactive.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    import glasspy.data as gd
    df = gd.SciGlass().data
    els = [c for c in df.columns if c[0] == "elements"]
    E = df[els].copy(); E.columns = [c[1] for c in els]
    base = (E.get("Si", 0) > 0) & (E.get("Ca", 0) > 0) & (E.get("Na", 0) > 0)
    sub = df[base.to_numpy()].copy()
    sub.to_pickle(cache)
    return sub


def glass_dataset(prop="Tg", label=None, unit="K", src_q=0.70, tgt_q=0.75):
    df = _sciglass_frame()
    els = [c for c in df.columns if c[0] == "elements"]
    E = df[els].copy(); E.columns = [c[1] for c in els]
    feat = [e for e in E.columns if (E[e] > 0).mean() > 0.03]
    ok = df[("property", prop)].notna().to_numpy()
    X = E.loc[ok, feat].to_numpy(float); y = df.loc[ok, ("property", prop)].to_numpy(float)
    P = E.loc[ok, "P"].to_numpy(float)
    pos = P[P > P.min()]
    src = np.where(P <= np.quantile(P, src_q))[0]
    tgt = np.where(P >= np.quantile(pos, tgt_q))[0]
    meta = {"author": df.loc[ok, ("metadata", "Author")].to_numpy(),
            "year": df.loc[ok, ("metadata", "Year")].to_numpy(),
            "chem_analysis": df.loc[ok, ("metadata", "ChemicalAnalysis")].to_numpy(),
            "shift_axis": "P atomic fraction", "src_rule": f"P <= q{src_q:.2f}(P)",
            "tgt_rule": f"P >= q{tgt_q:.2f}(P | P>0)", "source": "SciGlass via glasspy"}
    return Dataset(label or f"glass_{prop}", X, y, feat, P, src, tgt, unit,
                   composition_groups(X), meta)


def steel_dataset():
    from matminer.datasets import load_dataset
    st = load_dataset("steel_strength")
    cols = ["c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"]
    X = st[cols].to_numpy(float); y = st["yield strength"].to_numpy(float)
    ni = st["ni"].to_numpy(float); pos = ni[ni > ni.min()]
    src = np.where(ni <= np.quantile(ni, 0.60))[0]; tgt = np.where(ni >= np.quantile(pos, 0.85))[0]
    meta = {"shift_axis": "Ni content (wt%)", "src_rule": "Ni <= q0.60(Ni)",
            "tgt_rule": "Ni >= q0.85(Ni | Ni>min)", "source": "matminer steel_strength"}
    return Dataset("steel_yield", X, y, cols, ni, src, tgt, "MPa", composition_groups(X), meta)


def polymer_dataset():
    d = np.load(os.path.join(DEMO, "polymer_tg.npz"), allow_pickle=True)
    X, y, feat = d["X"].astype(float), d["y"].astype(float), [str(f) for f in d["feat"]]
    arom = X[:, feat.index("FracAromAtoms")]; pos = arom[arom > arom.min()]
    src = np.where(arom <= np.quantile(arom, 0.55))[0]; tgt = np.where(arom >= np.quantile(pos, 0.85))[0]
    meta = {"shift_axis": "fraction of aromatic atoms", "src_rule": "FracAromAtoms <= q0.55",
            "tgt_rule": "FracAromAtoms >= q0.85(FracAromAtoms | >min)",
            "source": "OsBaran polymer_tg_dataset, RDKit descriptors"}
    return Dataset("polymer_Tg", X, y, feat, arom, src, tgt, "K", composition_groups(X), meta)


def design_shift_datasets():
    out = [glass_dataset(p, lab, u) for p, lab, u in GLASS_PROPS]
    out += [steel_dataset(), polymer_dataset()]
    return out


# =============================================================== splits
def _perm_grouped(idx, groups, rng):
    """Permute indices so that replicate records of a composition stay contiguous;
    returns (ordered indices, group boundary-respecting cut function)."""
    idx = np.asarray(idx)
    if groups is None:
        return rng.permutation(idx), None
    g = groups[idx]; ug = rng.permutation(np.unique(g))
    order = np.concatenate([idx[g == u] for u in ug])
    gid = groups[order]
    return order, gid


def _cut(order, gid, frac):
    """Index position closest to frac*len that does not split a group."""
    n = len(order); pos = int(round(frac * n))
    if gid is None or pos <= 0 or pos >= n:
        return pos
    while 0 < pos < n and gid[pos] == gid[pos - 1]:
        pos += 1
    return pos


def design_split(ds: Dataset, seed: int, group=True, fr=(0.6, 0.2)):
    rng = np.random.default_rng(seed)
    groups = ds.groups if group else None
    S, gS = _perm_grouped(ds.src, groups, rng)
    a = _cut(S, gS, fr[0]); b = _cut(S, gS, fr[0] + fr[1])
    T, gT = _perm_grouped(ds.tgt, groups, rng)
    h = _cut(T, gT, 0.5)
    sp = {"train": S[:a], "cal": S[a:b], "stest": S[b:], "rpool": T[:h], "ttest": T[h:]}
    check_disjoint(sp, ds.groups if group else None)
    return sp


def check_disjoint(sp, groups=None):
    keys = list(sp)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = sp[keys[i]], sp[keys[j]]
            assert len(np.intersect1d(a, b)) == 0, f"index overlap {keys[i]} / {keys[j]}"
            if groups is not None:
                assert len(np.intersect1d(groups[a], groups[b])) == 0, \
                    f"composition overlap {keys[i]} / {keys[j]}"


def headline_m(ds: Dataset):
    return int(np.clip(len(ds.tgt) // 3, 10, 30))


def save_splits(ds: Dataset, split_fn=design_split, seeds=SEEDS, tag=None, **kw):
    """Persist every split's index sets so third parties can reproduce them."""
    out = {str(s): {k: v.tolist() for k, v in split_fn(ds, s, **kw).items()} for s in seeds}
    path = os.path.join(SPLITS, f"{tag or ds.name}_splits.json")
    json.dump({"dataset": ds.summary() if hasattr(ds, "summary") else str(ds),
               "protocol": (split_fn.__doc__ or split_fn.__name__), "splits": out},
              open(path, "w"))
    return path


# =============================================================== conformal
def conformal_q(scores, alpha=ALPHA):
    """Finite-sample split-conformal quantile; +inf if too few scores."""
    s = np.asarray(scores, float); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return np.sort(s)[k - 1]


def weighted_conformal_q(scores, w_cal, w_test, alpha=ALPHA):
    """Tibshirani et al. (2019) weighted quantile for one test weight (scalar)."""
    s = np.asarray(scores, float); order = np.argsort(s)
    s, w = s[order], np.asarray(w_cal, float)[order]
    p = np.append(w, w_test); p = p / p.sum()
    c = np.cumsum(p[:-1])
    j = np.searchsorted(c, 1 - alpha)
    return np.inf if j >= len(s) else s[j]


def beta_coverage_quantiles(n, alpha=ALPHA, qs=(0.05, 0.5, 0.95)):
    """Distribution of split-conformal test coverage given n calibration points
    (Vovk 2012; Angelopoulos & Bates 2023): Beta(n+1-l, l), l = floor((n+1)alpha)."""
    from scipy.stats import beta
    l = int(np.floor((n + 1) * alpha))
    if l < 1:
        return [np.nan] * len(qs)
    return [float(beta.ppf(q, n + 1 - l, l)) for q in qs]


# =============================================================== metrics
def coverage(lo, hi, y):
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_score(lo, hi, y, alpha=ALPHA):
    """Gneiting & Raftery (2007) interval score (lower is better)."""
    return float(np.mean((hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)))


def weighted_interval_score(y, median, intervals):
    """WIS (Bracher et al. 2021). intervals: {alpha: (lo, hi)}."""
    K = len(intervals)
    tot = 0.5 * np.abs(y - median)
    for a, (lo, hi) in intervals.items():
        tot = tot + (a / 2) * ((hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi))
    return float(np.mean(tot / (K + 0.5)))


def calibration_error(levels, covs):
    """Interval calibration error: mean and max |empirical - nominal| over levels."""
    d = np.abs(np.asarray(covs) - np.asarray(levels))
    return float(d.mean()), float(d.max())


def cluster_conditional_coverage(X, covered, n_clusters=8, seed=0, min_size=10):
    """Coverage within k-means clusters of the evaluated inputs (standardised);
    returns (worst-cluster coverage, per-cluster list)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(X)
    k = int(min(n_clusters, max(1, len(X) // min_size)))
    lab = KMeans(k, n_init=5, random_state=seed).fit_predict(Z)
    per = [(int((lab == c).sum()), float(covered[lab == c].mean())) for c in range(k) if (lab == c).sum() >= min_size]
    worst = min(c for _, c in per) if per else float("nan")
    return worst, per


def width_stratified_coverage(width, covered, n_bins=4):
    """Size-stratified coverage (Angelopoulos & Bates 2023): coverage within
    quantile bins of the interval width; returns worst-bin coverage."""
    if np.allclose(width, width[0]):
        return float(covered.mean())
    edges = np.quantile(width, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-9
    b = np.clip(np.searchsorted(edges, width, side="right") - 1, 0, n_bins - 1)
    return float(min(covered[b == i].mean() for i in range(n_bins) if (b == i).any()))


def risk_coverage(err, unc, grid=np.linspace(0.1, 1.0, 19)):
    """Selective-prediction risk-coverage curve: RMSE on the fraction of most
    confident predictions (ranked by unc); AURC = mean RMSE over the grid,
    reported relative to the no-abstention RMSE."""
    order = np.argsort(unc); e = np.asarray(err)[order]
    rm = np.array([np.sqrt(np.mean(e[:max(1, int(round(c * len(e))))] ** 2)) for c in grid])
    full = np.sqrt(np.mean(e ** 2))
    return {"grid": grid.tolist(), "rmse": rm.tolist(), "rmse_all": float(full),
            "aurc_rel": float(rm.mean() / full), "rmse_at50": float(rm[np.argmin(np.abs(grid - 0.5))]),
            "reduction_at50_pct": float(100 * (1 - rm[np.argmin(np.abs(grid - 0.5))] / full))}


def abstention_rate(width, max_width):
    """Fraction of candidates whose interval is wider than an acceptable width."""
    return float(np.mean(np.asarray(width) > max_width))


def interval_metrics(lo, hi, y, y_ref_iqr, alpha=ALPHA, X=None):
    cov = (y >= lo) & (y <= hi); w = hi - lo
    out = {"coverage": float(cov.mean()), "width": float(np.mean(w)),
           "norm_width": float(np.mean(w) / y_ref_iqr), "interval_score": interval_score(lo, hi, y, alpha),
           "wsc_width_strat": width_stratified_coverage(w, cov)}
    if X is not None and len(y) >= 40:
        out["worst_cluster_cov"] = cluster_conditional_coverage(X, cov)[0]
    return out


def summarize(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if len(a) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    return {"mean": float(a.mean()), "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": int(len(a))}


def iqr(a):
    q1, q3 = np.percentile(a, [25, 75]); return float(q3 - q1)


# =============================================================== models
MODEL_NAMES = ["RF", "ExtraTrees", "HistGB", "GP", "SVR", "kNN", "Ridge", "MLP"]


def make_model(name, seed=0):
    """Point predictors spanning tree ensembles, kernels, instance-based, linear
    and neural families. Hyperparameters fixed a priori (no tuning on target)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "RF":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=8, random_state=seed)
    if name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=8, random_state=seed)
    if name == "HistGB":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, early_stopping=False, random_state=seed)
    if name == "GP":
        return _SubsampledGP(seed=seed)
    if name == "SVR":
        from sklearn.svm import SVR
        from sklearn.compose import TransformedTargetRegressor
        return TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05)),
                                          transformer=StandardScaler())
    if name == "kNN":
        from sklearn.neighbors import KNeighborsRegressor
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance"))
    if name == "Ridge":
        from sklearn.linear_model import RidgeCV
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if name == "MLP":
        from sklearn.neural_network import MLPRegressor
        from sklearn.compose import TransformedTargetRegressor
        return TransformedTargetRegressor(make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(128, 128), alpha=1e-3, learning_rate_init=1e-3, max_iter=600,
            early_stopping=True, validation_fraction=0.1, random_state=seed)), transformer=StandardScaler())
    raise ValueError(name)


class _SubsampledGP:
    """Exact GP (RBF + white noise, ARD off) on at most 2,500 training points."""
    def __init__(self, seed=0, max_n=2500):
        self.seed, self.max_n = seed, max_n

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(y), min(self.max_n, len(y)), replace=False)
        self.sx = StandardScaler().fit(X[idx]); self.ym, self.ys = y[idx].mean(), y[idx].std() + 1e-12
        k = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
        self.gp = GaussianProcessRegressor(k, normalize_y=False, random_state=self.seed, n_restarts_optimizer=1)
        self.gp.fit(self.sx.transform(X[idx]), (y[idx] - self.ym) / self.ys)
        return self

    def predict(self, X, return_std=False):
        m, s = self.gp.predict(self.sx.transform(X), return_std=True)
        if return_std:
            return m * self.ys + self.ym, s * self.ys
        return m * self.ys + self.ym


def rf_mean_std(model, X):
    P = np.stack([t.predict(X) for t in model.estimators_], 0)
    return P.mean(0), P.std(0)


def dump(obj, name):
    path = os.path.join(RES, name)
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return path
