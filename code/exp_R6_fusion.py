"""R6 -- genuine, naturally incomplete multi-measurement fusion (Reviewer 1 comment 3;
Reviewer 2 comments 4 and 6), under the unified protocol (PROTOCOL.md, common.py).

What this is (and is not)
-------------------------
Heterogeneous, naturally incomplete characterisation measurements (density, dilatometric
expansion, refractive index) reported by different laboratories in SciGlass, fused with
composition. It is NOT imaging / spectral multimodality: no images or spectra exist in
SciGlass; that gap is a stated limitation. Missingness is real (what each laboratory chose
to measure), not simulated -- except in the explicit 'measurement not yet made' deployment
case, where auxiliaries that WERE measured are masked at prediction time.

Designs (scoping study revision/audit/A2_sciglass_explore.md, sections 4.1, 4.2, 4.5)
  B1 (primary)    target Tg; auxiliaries Density293K (rho), CTEbelowTg (CTE, x1e6 -> ppm/K),
                  RefractiveIndex (nD); 8 natural missing patterns.
  B2 (robustness) target YoungModulus; auxiliaries Density293K, CTEbelowTg, Tg.
  Near-label-leakage auxiliaries are excluded a priori (A2 4.1): viscosity iso-points and
  dilatometric/softening temperatures (TAnnealing, Tstrain, T11, T12, Tsoft,
  TdilatometricSoftening, TLittletons) and DTA/DSC crystallisation temperatures for Tg;
  ShearModulus (elastically coupled) and PoissonRatio (derived from E and G) for E. The
  script recomputes co-occurrence counts and Pearson r for the screen (JSON 'aux_screen').
  Whether the missing pattern tracks laboratory practice is COMPUTED, not asserted (JSON
  'practice_concentration'): for every laboratory / publication with >= 20 records, the share
  of its records carried by its single most frequent missing pattern, against a label-permuted
  baseline. What is missing is what was measured, reported AND captured by the SciGlass
  curators; the database is a compilation, so a missing value is not proof of a missing
  measurement (A2 section 1; stated as a limitation).
  Inputs: the 25 element atomic fractions of common.glass_dataset (present in > 3 % of the
  bioactive subset Si, Ca, Na > 0) + the auxiliaries (NaN where not reported).

Splits (20 seeds each; indices persisted in results/splits/R6_fusion_*_splits.json)
  comp  composition-grouped random 60/20/20 (train / calibration / test), grouping by
        common.composition_groups (identical feature vectors at 1e-6), seeds s = 0..19,
        rng = default_rng(s), common._perm_grouped / _cut.
  lab   laboratory-grouped (A2 4.3): test = >= 20 % of records by whole units, calibration
        = >= 25 % of the remainder (about 20 % of all) by whole units, train = rest;
        rng = default_rng(1000 + s). The unit is a connected component of the bipartite
        graph laboratory <-> composition (laboratory = A2's normalised first-author key),
        so whole laboratories are held out AND one composition never straddles a split
        (the protocol's composition grouping). Laboratories that reported an identical
        composition are thereby held out together (largest unit: about 15 % of records).
  Diagnostics per split: share of test records whose laboratory / publication (SciGlass
  Kod) also occurs in calibration or training (A1 audit, section 6.3: calibration/test
  dependence through shared publications).

Models (hyperparameters fixed a priori; common.make_model)
  hgb_comp        HistGB on composition only.
  hgb_fused       HistGB on composition + auxiliaries, native NaN handling.
  hgb_fused_drop  hgb_fused + modality-dropout augmentation (A2 4.4): every training row
                  with >= 1 observed auxiliary is duplicated with ALL auxiliaries set to NaN.
  rf_comp         RF on composition only (reference for the second family; extra).
  rf_fused_imp    RF on composition + auxiliaries, median imputation + missing-indicator
                  columns (sklearn SimpleImputer(strategy='median', add_indicator=True)).

Conformal (alpha = 0.10, |y - mu| score, finite-sample common.conformal_q)
  pooled    one quantile from all calibration records.
  mondrian  one quantile per missing-pattern stratum. Merge rule (fixed a priori, uses
            calibration pattern counts only, never labels): patterns are processed from
            most to fewest auxiliaries; a pattern whose accumulated calibration count (own
            + merged children) is < 30 is merged into the parent pattern with one fewer
            auxiliary that has the largest own calibration count (ties -> smaller bit
            code). Composition-only is the root; if the merged root itself holds < 30
            calibration records, its members fall back to the POOLED quantile. That does
            happen occasionally on the laboratory-grouped split (whole units can strip the
            root); it is logged per split as strata.root_pooled_fallback, counted in the
            JSON as root_pooled_fallback_n_splits and reported in section 2 of the .md.
            After a merge the guarantee is marginal over the merged stratum, not per pattern.
  Paired split-to-split comparison of the two schemes (JSON 'paired_pooled_vs_mondrian':
  mean difference, SD, splits better / worse / tied, Wilcoxon signed-rank p) accompanies
  every pooled -> Mondrian statement, because the worst-pattern minimum is very noisy.
  Metrics overall: RMSE, coverage, width, width / IQR(y of the whole design dataset),
  interval score, WIS (alphas 0.1..0.5), calibration error over LEVELS, width-stratified
  coverage, worst k-means cluster coverage (composition inputs), risk-coverage / AURC (for
  Mondrian; pooled width is constant so it cannot rank), worst/best pattern coverage over
  patterns with >= 20 test records, and a reference for the worst-pattern coverage that
  exact per-stratum calibration would give with the same calibration/test counts
  (Beta-binomial simulation: coverage | calibration ~ Beta(k, n+1-k)).
  Per pattern: n, RMSE, coverage, width, normalised width, interval score, stratum used.
  Width decomposition: Mondrian width of pattern p relative to composition-only glasses,
  for the fused model (total apparent effect) and for the composition-only model on the
  same strata (population effect), and fused / composition-only width within pattern
  (information effect).

'Measurement not yet made' (deployment case; fused models)
  mask_all     every test record's auxiliaries set to NaN at prediction time; calibration
               on ALL calibration records masked identically (the deployment missingness
               is uniform: nothing measured yet). Compared with two naive recipes that
               reuse natural calibration residuals: the natural composition-only stratum's
               quantile ('borrow') and the natural pooled quantile.
  by_pattern   A2's design: test records of natural pattern P (with >= 1 auxiliary)
               masked; calibration on calibration records of P's Mondrian stratum masked.

Trust-layer decision procedure (B1 only; trustlayer.TrustLayer imported, not edited)
  candidates = test records; specification Tg >= tau, tau fixed a priori as the median
  (and 75th percentile) of Tg over the whole B1 dataset; the layer is calibrated once
  with group label = Mondrian stratum. Arms (all one TrustLayer object):
    I_pooled            interval decide, groups=None (pooled quantile)
    I_mondrian          interval decide, groups = stratum
    I_mondrian_shift    + conformal novelty shift check (Mondrian by stratum); flagged
                        candidates have no target stratum -> not_certified -> measured
    S_pooled            conformal selection (Jin & Candes 2023), one BH at q = 0.10, pooled
    S_mondrian_joint    stratum-wise selection p-values, one BH over all candidates
    S_mondrian_stratum  BH inside each stratum (FDR controlled per stratum)
    (selection arms: not selected -> the same calibration's interval reject stays reject,
     everything else is measured)
  Not yet measured (candidates' auxiliaries masked):
    NYM_I_maskcal / NYM_S_maskcal  layer calibrated on identically masked calibration
                                   records (single stratum)
    NYM_I_borrow / NYM_I_pooled_nat naive: natural calibration, borrowed comp-only
                                   stratum / pooled quantile
  Per arm, overall and per natural pattern: accepted, false accepts, FDP among accepted
  (0 when nothing accepted, so the split mean estimates the FDR), power, share measured,
  Tg measurements requested (measure + not_certified), auxiliary measurements consumed,
  interval miscoverage. G1 (false accept => miscovered) is asserted per split. The realised
  selection FDP of every arm is tested against the nominal q with a one-sided one-sample t
  over the splits (JSON 'fdp_vs_q'), so 'violated' is never claimed from a point estimate.
  Minimal extension of TrustLayer (by wrapping, no subclass needed): a pre-fitted
  predictor wrapper (refit=False) and pre-computed stratum labels; strata smaller than 30
  are relabelled '__pooled__', which TrustLayer routes to its parent (all records).

Determinism: every number in the JSON is bit-exact from run to run and from process to process
(checked by running the script twice and diffing the JSON). Two sources of drift were removed in
the fix round: set-iteration order feeding the rng of the worst-pattern Beta-binomial reference,
and RandomForest's parallel tree accumulation (RF_N_JOBS = 1).

Reporting: every quantity per split; mean and 2.5-97.5 percentile across the 20 splits
(common.summarize). The splits resample one finite dataset; the interval describes
split-to-split variability, not population sampling error.
Outputs: results/R6_fusion.json, results/R6_fusion.md.
Run:  python -W ignore exp_R6_fusion.py [n_jobs]        full run (default 6 workers)
      python -W ignore exp_R6_fusion.py --quick         2 seeds, smoke test
      python -W ignore exp_R6_fusion.py --report        rebuild the .md from the JSON
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joblib import Parallel, delayed  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from common import (ALPHA, SEEDS, LEVELS, RES, Dataset, _sciglass_frame, _perm_grouped, _cut,  # noqa: E402
                    check_disjoint, composition_groups, conformal_q, interval_metrics, interval_score,
                    weighted_interval_score, calibration_error, risk_coverage, summarize, iqr, make_model,
                    save_splits, dump)
import trustlayer as TLm  # noqa: E402
from trustlayer import TrustLayer, ACCEPT, REJECT, MEASURE, NOT_CERTIFIED  # noqa: E402

TAG = "R6_fusion"
N_MIN_STRATUM = 30
MIN_PATTERN_TEST = 20            # a pattern enters the worst/best-pattern statistics at this test size
MIN_PATTERN_TEST2 = 50           # robustness: the same statistics restricted to larger patterns
PRACTICE_MIN_N = 20              # laboratories / publications entering the practice-concentration statistic
PRACTICE_PERMS = 50              # label permutations for its chance baseline
WIS_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
Q_FDR = 0.10
NOVELTY_K, NOVELTY_LEVEL = 10, 0.05
THREADS = 2                      # BLAS/OpenMP threads per worker; 6 workers -> <= 12 cores
RF_N_JOBS = 1                    # RandomForest joblib workers. 1, not THREADS: with >1 the parallel
                                 # tree accumulation is not bit-reproducible ACROSS PROCESSES (~2e-13),
                                 # which is enough to flip a width-stratified-coverage bin on an exact
                                 # tie. n_jobs=1 makes every arm bit-exact run to run (PROTOCOL.md:
                                 # "keep scripts deterministic (seeded) and re-runnable"); it costs
                                 # about 2 s per RF fit.
MODELS = ["hgb_comp", "hgb_fused", "hgb_fused_drop", "rf_comp", "rf_fused_imp"]
FUSED = ["hgb_fused", "hgb_fused_drop", "rf_fused_imp"]
COMP_REF = {"hgb_fused": "hgb_comp", "hgb_fused_drop": "hgb_comp", "rf_fused_imp": "rf_comp"}
TL_PREDICTORS = ["hgb_comp", "hgb_fused", "hgb_fused_drop", "rf_fused_imp"]
SPLIT_KINDS = ["comp", "lab"]
POOLED_LABEL = "__pooled__"

DESIGNS = {
    "B1_Tg": {"target": "Tg", "unit": "K", "aux": ["Density293K", "CTEbelowTg", "RefractiveIndex"],
              "trust_layer": True},
    "B2_E": {"target": "YoungModulus", "unit": "GPa", "aux": ["Density293K", "CTEbelowTg", "Tg"],
             "trust_layer": False},
}
ABBR = {"Density293K": "rho", "CTEbelowTg": "CTE", "RefractiveIndex": "nD", "Tg": "Tg"}
# auxiliary screen (verdicts fixed a priori from A2 section 4.1; counts and r recomputed here)
AUX_SCREEN = {
    "Tg": {"Density293K": "used", "CTEbelowTg": "used (dilatometry; the same run can also give a dilatometric Tg)",
           "RefractiveIndex": "used",
           "YoungModulus": "not used (4th modality makes patterns sparse)",
           "Microhardness": "not used (sparse)",
           "CTE433K": "not used (alternative CTE definition)", "CTE483K": "not used (alternative CTE definition)",
           "Tliquidus": "not used (another characteristic temperature)",
           "CrystallizationPeak": "excluded (same DTA/DSC thermogram as Tg)",
           "CrystallizationOnset": "excluded (same DTA/DSC thermogram as Tg)",
           "TAnnealing": "excluded (viscosity iso-point, near-label leakage)",
           "Tstrain": "excluded (viscosity iso-point, near-label leakage)",
           "T12": "excluded (Tg ~ 10^12 Pa s temperature, near-label leakage)",
           "T11": "excluded (viscosity iso-point, near-label leakage)",
           "Tsoft": "excluded (same viscosity curve)",
           "TdilatometricSoftening": "excluded (same dilatometric curve)",
           "TLittletons": "excluded (same viscosity curve)"},
    "YoungModulus": {"Density293K": "used", "CTEbelowTg": "used", "Tg": "used",
                     "ShearModulus": "excluded (elastically coupled to E)",
                     "PoissonRatio": "excluded (usually derived from E and G)",
                     "RefractiveIndex": "not used (too few co-occurring records)",
                     "Microhardness": "not used (not in the A2 design)"},
}


# ============================================================ laboratory proxy (copied from A2_pilot.py)
def fold_ascii(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace("`", "").replace("'", "")
    s = re.sub(r"[^a-z\-\. ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def author_key(s):
    """Normalised first-author key (surname + first initial), A2 section 2.1."""
    s = fold_ascii(str(s))
    toks = s.replace(".", ". ").split()
    if not toks:
        return "unknown"
    sur = toks[0].strip(".-,")
    rest = [t for t in toks[1:] if t.strip(".") not in ("iii", "ii", "jr", "sr")]
    ini = rest[0][0] if rest else ""
    sur = re.sub(r"(ii|iy|yi|y|ij)$", "i", sur)
    return sur + ("_" + ini if ini else "")


def lab_composition_components(comp_groups, lab_codes):
    """Connected components of the bipartite graph laboratory <-> composition."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for c, l in zip(comp_groups, lab_codes):
        ra, rb = find(("c", int(c))), find(("l", int(l)))
        if ra != rb:
            parent[ra] = rb
    roots = [find(("l", int(l))) for l in lab_codes]
    _, inv = np.unique(np.array([f"{a}{b}" for a, b in roots]), return_inverse=True)
    return inv


# ============================================================ data
def modal_pattern_concentration(keys, labels, min_n=20, perms=PRACTICE_PERMS, seed=0):
    """Does the missing pattern track who produced the record? For every key (laboratory or
    publication) with >= min_n records, the share of its records carried by its single most
    frequent missing pattern; compared with the same statistic after permuting the pattern
    labels over all records (what pure chance gives at these group sizes). Descriptive only,
    computed on the whole design dataset, never on a split."""
    keys = np.asarray(keys)
    s = pd.Series(keys).value_counts()
    big = list(s[s >= min_n].index)
    if not big:
        return {"n_groups": 0, "mean_modal_share": None, "share_ge_80pct": None, "permuted_baseline": None}

    def modal(lab):
        return np.array([pd.Series(lab[keys == k]).value_counts().iloc[0] / int((keys == k).sum()) for k in big])

    obs = modal(np.asarray(labels, dtype=object))
    rng = np.random.default_rng(seed)
    base = [float(modal(rng.permutation(np.asarray(labels, dtype=object))).mean()) for _ in range(perms)]
    return {"n_groups": len(big), "n_records_in_groups": int(s[s >= min_n].sum()),
            "mean_modal_share": float(obs.mean()), "share_ge_80pct": float((obs >= 0.8).mean()),
            "permuted_baseline": float(np.mean(base))}


def pattern_label(code, aux):
    lab = "+".join(ABBR[a] for i, a in enumerate(aux) if (code >> i) & 1)
    return lab or "comp-only"


def load_design(key):
    spec = DESIGNS[key]
    df = _sciglass_frame()
    els = [c for c in df.columns if c[0] == "elements"]
    E = df[els].copy()
    E.columns = [c[1] for c in els]
    feat = [e for e in E.columns if (E[e] > 0).mean() > 0.03]          # as common.glass_dataset
    P = df["property"]
    ok = P[spec["target"]].notna().to_numpy()
    Xc = E.loc[ok, feat].to_numpy(float)
    aux = spec["aux"]
    A = P.loc[ok, aux].to_numpy(float).copy()
    if "CTEbelowTg" in aux:
        A[:, aux.index("CTEbelowTg")] *= 1e6                            # 1/K -> ppm/K (cosmetic)
    y = P.loc[ok, spec["target"]].to_numpy(float)
    obs = ~np.isnan(A)
    code = (obs * (1 << np.arange(len(aux)))).sum(1).astype(int)
    labs = df.loc[ok, ("metadata", "Author")].astype(str).map(author_key).to_numpy()
    _, lab_codes = np.unique(labs, return_inverse=True)
    kod = (df.index.to_numpy()[ok] // 100000000).astype(np.int64)
    year = df.loc[ok, ("metadata", "Year")].to_numpy(float)
    groups = composition_groups(Xc)
    units = lab_composition_components(groups, lab_codes)
    Xf = np.hstack([Xc, A])
    ds = Dataset(f"{TAG}_{key}", Xf, y, feat + aux, np.zeros(len(y)), np.arange(len(y)), np.zeros(0, int),
                 spec["unit"], groups, {"target": spec["target"], "aux": ",".join(aux),
                                        "source": "SciGlass via glasspy (bioactive subset Si, Ca, Na > 0)",
                                        "lab_proxy": "normalised first-author key (A2 2.1)"})
    D = {"key": key, "ds": ds, "Xf": Xf, "nc": Xc.shape[1], "aux": aux, "y": y, "code": code,
         "labs": lab_codes, "lab_names": labs, "kod": kod, "year": year, "groups": groups, "units": units,
         "y_iqr": iqr(y), "feat": feat}
    # auxiliary screen and descriptive statistics
    scr = {}
    for a, verdict in AUX_SCREEN[spec["target"]].items():
        if a not in P.columns:
            scr[a] = {"verdict": verdict, "n_with_target": 0, "pearson_r": None}
            continue
        v = P.loc[ok, a].to_numpy(float)
        m = ~np.isnan(v)
        r = float(np.corrcoef(v[m], y[m])[0, 1]) if m.sum() >= 50 else None
        scr[a] = {"verdict": verdict, "n_with_target": int(m.sum()), "pearson_r": r}
    pats = {}
    for c in sorted(set(code.tolist()), key=lambda c: -(code == c).sum()):
        m = code == c
        pats[pattern_label(c, aux)] = {"code": int(c), "n": int(m.sum()), "y_mean": float(y[m].mean()),
                                       "y_sd": float(y[m].std(ddof=1)), "n_labs": int(len(np.unique(lab_codes[m]))),
                                       "n_publications": int(len(np.unique(kod[m]))),
                                       "median_year": float(np.nanmedian(year[m]))}
    lbl = np.array([pattern_label(int(c), aux) for c in code], dtype=object)
    # A2 section 2.1's corporate-name pattern, recomputed on this subset. It is reported only to
    # show that it is too thin to test A2 4.2's "patent/industrial series report rho+CTE"
    # illustration, which this report therefore does not use as evidence.
    astr = df.loc[ok, ("metadata", "Author")].astype(str)
    corp = astr.str.contains(r"(?:Co\.|Corp|Ltd|Inc|GmbH|Glass|Glas|Works|Company|Industr|Elec|Limited|Philips|Institut)",
                             regex=True).to_numpy()
    corp_chk = {"n_records": int(corp.sum()), "n_author_strings": int(astr[corp].nunique()),
                "n_normalised_author_keys": int(len(np.unique(lab_codes[corp]))) if corp.any() else 0}
    conc = {"min_records": PRACTICE_MIN_N, "n_permutations": PRACTICE_PERMS, "corporate_author_check": corp_chk,
            **{nm: modal_pattern_concentration(k, lbl, PRACTICE_MIN_N)
               for nm, k in (("laboratories", lab_codes), ("publications", kod))}}
    uc = pd.Series(units).value_counts()
    D["describe"] = {"n": int(len(y)), "n_features_composition": int(Xc.shape[1]), "aux": aux,
                     "practice_concentration": conc,
                     "aux_units": {a: ("ppm/K" if a == "CTEbelowTg" else "SciGlass unit") for a in aux},
                     "aux_observed": {a: int(obs[:, i].sum()) for i, a in enumerate(aux)},
                     "patterns": pats, "n_unique_compositions": int(len(np.unique(groups))),
                     "n_labs": int(len(np.unique(lab_codes))), "n_publications": int(len(np.unique(kod))),
                     "n_lab_composition_units": int(len(uc)), "largest_units": [int(v) for v in uc.iloc[:5]],
                     "largest_unit_share": float(uc.iloc[0] / len(y)),
                     "y_iqr": D["y_iqr"], "y_median": float(np.median(y)), "y_q75": float(np.percentile(y, 75)),
                     "aux_screen": scr}
    return D


# ============================================================ splits
def group_partition(groups, idx, frac, rng):
    """Split idx into (rest, held) with held >= frac of the records, whole groups only
    (A2_pilot.group_partition)."""
    g = groups[idx]
    ug = rng.permutation(np.unique(g))
    sizes = pd.Series(g).value_counts()
    target_n = frac * len(idx)
    held, acc = [], 0
    for k in ug:
        if acc >= target_n:
            break
        held.append(k)
        acc += sizes[k]
    m = np.isin(g, held)
    return idx[~m], idx[m]


def split_fusion(ds, seed, kind="comp", units=None, labs=None):
    """R6 split families. comp: composition-grouped random 60/20/20 (train/cal/test), rng =
    default_rng(seed), common._perm_grouped/_cut. lab: whole laboratory-composition units,
    test >= 20 % of records, calibration >= 25 % of the rest, train = rest, rng =
    default_rng(1000 + seed). Both are asserted composition-disjoint (and lab-disjoint for lab)."""
    n = len(ds.y)
    if kind == "comp":
        rng = np.random.default_rng(seed)
        order, gid = _perm_grouped(np.arange(n), ds.groups, rng)
        a, b = _cut(order, gid, 0.6), _cut(order, gid, 0.8)
        sp = {"train": order[:a], "cal": order[a:b], "test": order[b:]}
    elif kind == "lab":
        rng = np.random.default_rng(1000 + seed)
        rest, te = group_partition(units, np.arange(n), 0.20, rng)
        tr, cal = group_partition(units, rest, 0.25, rng)
        sp = {"train": np.sort(tr), "cal": np.sort(cal), "test": np.sort(te)}
    else:
        raise ValueError(kind)
    check_disjoint(sp, ds.groups)
    if kind == "lab":
        check_disjoint(sp, labs)
    return sp


# ============================================================ models
class ColumnModel:
    """Fit/predict an estimator on a subset of columns of the characterisation matrix."""

    def __init__(self, est, cols):
        self.est, self.cols = est, np.asarray(cols)

    def fit(self, X, y):
        self.est.fit(np.asarray(X, float)[:, self.cols], y)
        return self

    def predict(self, X):
        return self.est.predict(np.asarray(X, float)[:, self.cols])


class DropoutAugmented:
    """Modality-dropout augmentation (A2 4.4): every training row with >= 1 observed
    auxiliary is duplicated with ALL auxiliaries set to NaN."""

    def __init__(self, est, aux_cols):
        self.est, self.aux = est, np.asarray(aux_cols)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        has = ~np.isnan(X[:, self.aux]).all(1)
        Xa = X[has].copy()
        Xa[:, self.aux] = np.nan
        self.n_aug_ = int(has.sum())
        self.est.fit(np.vstack([X, Xa]), np.r_[y, y[has]])
        return self

    def predict(self, X):
        return self.est.predict(np.asarray(X, float))


class Prefit:
    """Wrap an already-fitted model for TrustLayer.fit(..., refit=False)."""

    def __init__(self, m):
        self.m = m

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.m.predict(X)


def build_models(seed, nc, na):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    comp, allc, aux = np.arange(nc), np.arange(nc + na), np.arange(nc, nc + na)

    def rf():
        m = make_model("RF", seed)
        m.set_params(n_jobs=RF_N_JOBS)
        return m

    return {"hgb_comp": ColumnModel(make_model("HistGB", seed), comp),
            "hgb_fused": ColumnModel(make_model("HistGB", seed), allc),
            "hgb_fused_drop": DropoutAugmented(make_model("HistGB", seed), aux),
            "rf_comp": ColumnModel(rf(), comp),
            "rf_fused_imp": ColumnModel(make_pipeline(SimpleImputer(strategy="median", add_indicator=True), rf()), allc)}


# ============================================================ Mondrian strata
def mondrian_strata(cal_code, all_codes, n_aux, n_min=N_MIN_STRATUM):
    """Map every pattern code to a stratum root code using calibration counts only.
    Returns (root_of, accumulated calibration count per root, merges {child: parent})."""
    own = Counter(int(c) for c in cal_code)
    pats = set(int(c) for c in all_codes)
    acc = {p: own.get(p, 0) for p in pats}
    parent = {}
    pc = lambda c: bin(c).count("1")  # noqa: E731
    for L in range(n_aux, 0, -1):
        for p in sorted(q for q in pats if pc(q) == L):
            if acc[p] < n_min:
                cands = [p & ~(1 << b) for b in range(n_aux) if (p >> b) & 1]
                best = max(cands, key=lambda c: (own.get(c, 0), -c))
                parent[p] = best
                if best not in pats:
                    pats.add(best)
                    acc[best] = own.get(best, 0)
                acc[best] += acc[p]

    def root(p):
        while p in parent:
            p = parent[p]
        return p

    root_of = {p: root(p) for p in sorted(pats)}
    size = {r: acc[r] for r in sorted(set(root_of.values()))}     # sorted: stable JSON key order
    return root_of, size, parent


def stratum_labels(codes, root_of, size, aux, candidate=True):
    """Stratum label per record. Calibration records carry their root label; a CANDIDATE
    whose stratum has < n_min calibration records (only possible for the root) gets the
    label '__pooled__', which matches no calibration record, so both q_by_stratum and
    TrustLayer route it to all calibration records (pooled)."""
    out = []
    for c in codes:
        r = root_of[int(c)]
        out.append(pattern_label(r, aux) if (size[r] >= N_MIN_STRATUM or not candidate) else POOLED_LABEL)
    return np.array(out, dtype=object)


def q_by_stratum(r_cal, s_cal, s_te, a):
    qpool = conformal_q(r_cal, a)
    q = np.full(len(s_te), qpool, float)
    for s in np.unique(s_te):
        if s == POOLED_LABEL:
            continue
        m = s_cal == s
        if m.sum() >= N_MIN_STRATUM:
            q[s_te == s] = conformal_q(r_cal[m], a)
    return q


def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else None


def worst_pattern_reference(n_tests, strata, n_cal_of_stratum, alpha=ALPHA, sims=4000, seed=0):
    """Expected minimum per-pattern coverage if every stratum were exactly exchangeable with
    its calibration records: coverage | calibration ~ Beta(k, n+1-k), k = ceil((n+1)(1-a)),
    shared by the patterns of one stratum; test counts binomial."""
    if not n_tests:
        return None
    rng = np.random.default_rng(seed)
    cov_s = {}
    for s in sorted(set(strata)):          # sorted: set order varies with PYTHONHASHSEED and
                                           # would change the order of the rng draws (run-to-run
                                           # drift of ~0.001 in this reference before the fix)
        n = n_cal_of_stratum[s]
        k = int(np.ceil((n + 1) * (1 - alpha)))
        cov_s[s] = np.ones(sims) if k > n else rng.beta(k, n + 1 - k, sims)
    mins = np.full(sims, np.inf)
    for nt, s in zip(n_tests, strata):
        mins = np.minimum(mins, rng.binomial(nt, cov_s[s]) / nt)
    return float(mins.mean())


# ============================================================ conformal evaluation
def conformal_block(scheme, y_cal, mu_cal, s_cal, y_te, mu_te, s_te, code_te, aux, Xcomp_te, y_iqr, seed):
    r_cal = np.abs(y_cal - mu_cal)
    err = y_te - mu_te
    if scheme == "pooled":
        def qf(a):
            return np.full(len(y_te), conformal_q(r_cal, a))
    else:
        def qf(a):
            return q_by_stratum(r_cal, s_cal, s_te, a)
    q = qf(ALPHA)
    lo, hi = mu_te - q, mu_te + q
    out = interval_metrics(lo, hi, y_te, y_iqr, X=Xcomp_te)
    out["rmse"] = rmse(err)
    out["n_infinite"] = int((~np.isfinite(q)).sum())
    out["wis"] = weighted_interval_score(y_te, mu_te, {a: (mu_te - qf(a), mu_te + qf(a)) for a in WIS_ALPHAS})
    curve = [float(np.mean(np.abs(err) <= qf(1 - l))) for l in LEVELS]
    out["curve"] = curve
    out["ce_mean"], out["ce_max"] = calibration_error(LEVELS, curve)
    if scheme == "mondrian":
        jit = np.random.default_rng(seed).random(len(q)) * 1e-6       # random tie-break within strata
        out["risk_coverage"] = risk_coverage(np.abs(err), q + jit)
    cov = np.abs(err) <= q
    per = {}
    for c in np.unique(code_te):
        m = code_te == c
        per[pattern_label(int(c), aux)] = {
            "n": int(m.sum()), "rmse": rmse(err[m]), "coverage": float(cov[m].mean()),
            "width": float(np.mean(2 * q[m])), "norm_width": float(np.mean(2 * q[m]) / y_iqr),
            "interval_score": interval_score(lo[m], hi[m], y_te[m]),
            "stratum": str(s_te[m][0]) if scheme == "mondrian" else "pooled"}
    out["per_pattern"] = per
    for thr, sfx in ((MIN_PATTERN_TEST, ""), (MIN_PATTERN_TEST2, "_ge%d" % MIN_PATTERN_TEST2)):
        big = [v for v in per.values() if v["n"] >= thr]
        if big:
            out["worst_pattern_cov" + sfx] = float(min(v["coverage"] for v in big))
            out["best_pattern_cov" + sfx] = float(max(v["coverage"] for v in big))
            out["pattern_cov_range" + sfx] = out["best_pattern_cov" + sfx] - out["worst_pattern_cov" + sfx]
            out["n_patterns" + sfx] = len(big)
    return out


def masked_block(name, D, sp, mu, mu_mask, s_cal, s_te, s_co):
    """'Measurement not yet made': auxiliaries masked at prediction time. s_co = candidate
    stratum label of naturally composition-only records (for the naive 'borrow' recipe)."""
    y, aux, code, y_iqr = D["y"], D["aux"], D["code"], D["y_iqr"]
    cal, te = sp["cal"], sp["test"]
    ref = COMP_REF[name]
    r_cal_nat = np.abs(y[cal] - mu[name]["cal"])
    r_cal_m = np.abs(y[cal] - mu_mask[name]["cal"])
    err_m = y[te] - mu_mask[name]["test"]
    err_nat = y[te] - mu[name]["test"]
    err_ref = y[te] - mu[ref]["test"]
    code_te = code[te]
    # natural quantities
    q_pool_nat = conformal_q(r_cal_nat)
    s_root_co = s_co
    q_borrow = q_by_stratum(r_cal_nat, s_cal, np.array([s_co], dtype=object), ALPHA)[0]
    q_mask = conformal_q(r_cal_m)
    q_mond_nat = q_by_stratum(r_cal_nat, s_cal, s_te, ALPHA)
    r_ref_cal = np.abs(y[cal] - mu[ref]["cal"])
    q_mond_ref = q_by_stratum(r_ref_cal, s_cal, s_te, ALPHA)

    def evalq(err, q, sel):
        q = np.broadcast_to(np.asarray(q, float), err.shape)[sel]
        e = err[sel]
        yt = y[te][sel]
        mu_ = yt - e
        return {"coverage": float(np.mean(np.abs(e) <= q)), "width": float(np.mean(2 * q)),
                "norm_width": float(np.mean(2 * q) / y_iqr),
                "interval_score": interval_score(mu_ - q, mu_ + q, yt)}

    allm = np.ones(len(te), bool)
    out = {"mask_all": {"rmse_masked": rmse(err_m), "rmse_natural": rmse(err_nat), "rmse_comp_model": rmse(err_ref),
                        "maskcal": evalq(err_m, q_mask, allm), "borrow_comp_only": evalq(err_m, q_borrow, allm),
                        "pooled_natural": evalq(err_m, q_pool_nat, allm), "per_pattern": {}},
           "by_pattern": {}}
    for scheme, q in (("maskcal", q_mask), ("borrow_comp_only", q_borrow)):
        worst = []
        for c in np.unique(code_te):
            m = code_te == c
            if m.sum() >= MIN_PATTERN_TEST:
                worst.append(float(np.mean(np.abs(err_m[m]) <= q)))
        out["mask_all"][scheme]["worst_pattern_cov"] = min(worst) if worst else None
    for c in np.unique(code_te):
        m = code_te == c
        out["mask_all"]["per_pattern"][pattern_label(int(c), aux)] = {
            "n": int(m.sum()), "rmse_masked": rmse(err_m[m]), "rmse_natural": rmse(err_nat[m]),
            "rmse_comp_model": rmse(err_ref[m]),
            "cov_maskcal": float(np.mean(np.abs(err_m[m]) <= q_mask)),
            "cov_borrow": float(np.mean(np.abs(err_m[m]) <= q_borrow))}
    # A2 design: per natural pattern P with >= 1 auxiliary, calibration on P's stratum masked
    for c in np.unique(code_te):
        if c == 0:
            continue
        m = code_te == c
        st = s_te[m][0]
        cm = (s_cal == st) if st != POOLED_LABEL else np.ones(len(cal), bool)
        if cm.sum() < N_MIN_STRATUM:
            cm = np.ones(len(cal), bool)
        qP = conformal_q(r_cal_m[cm])
        out["by_pattern"][pattern_label(int(c), aux)] = {
            "n_test": int(m.sum()), "stratum": str(st), "n_cal_masked": int(cm.sum()),
            "rmse_with_aux": rmse(err_nat[m]), "rmse_masked": rmse(err_m[m]), "rmse_comp_model": rmse(err_ref[m]),
            "masked_q_maskcal": evalq(err_m, qP, m), "masked_q_borrow_comp_only": evalq(err_m, q_borrow, m),
            "masked_q_pooled_natural": evalq(err_m, q_pool_nat, m),
            "with_aux_q_mondrian": evalq(err_nat, q_mond_nat, m),
            "comp_model_q_mondrian": evalq(err_ref, q_mond_ref, m)}
    out["q"] = {"maskcal": float(q_mask), "borrow_comp_only": float(q_borrow), "pooled_natural": float(q_pool_nat),
                "borrow_stratum": str(s_root_co)}
    return out


# ============================================================ trust-layer decisions
def decision_metrics(dec, y, tau, lo, hi, code_te, aux, aux_used):
    ok = y >= tau
    acc, rej = dec == ACCEPT, dec == REJECT
    mea, nc = dec == MEASURE, dec == NOT_CERTIFIED
    miss = None if lo is None else ~((y >= lo) & (y <= hi))

    def block(S):
        n = int(S.sum())
        if n == 0:
            return None
        na = int((acc & S).sum()); fa = int((acc & ~ok & S).sum())
        nok = int((ok & S).sum()); nr = int((rej & S).sum()); fr = int((rej & ok & S).sum())
        nm = int(((mea | nc) & S).sum())
        return {"n": n, "n_acceptable": nok, "n_accept": na, "false_accepts": fa, "n_reject": nr,
                "false_rejects": fr, "n_not_certified": int((nc & S).sum()),
                "fdp": fa / na if na else 0.0, "fdp_if_any": fa / na if na else None,
                "power": (na - fa) / nok if nok else None,
                "accept_rate": na / n, "measure_share": nm / n, "tg_measurements": nm,
                "aux_measurements_used": int(aux_used[S].sum()),
                "uncond_false_accept": fa / n,
                "miscoverage": float(miss[S].mean()) if miss is not None else None}

    out = {"all": block(np.ones(len(y), bool)), "per_pattern": {}}
    for c in np.unique(code_te):
        out["per_pattern"][pattern_label(int(c), aux)] = block(code_te == c)
    return out


def trust_block(D, sp, models, s_cal, s_te, Xf_mask, s_co):
    Xf, y, aux, code = D["Xf"], D["y"], D["aux"], D["code"]
    tr, cal, te = sp["train"], sp["cal"], sp["test"]
    Xc, yc, code_te = Xf[te], y[te], code[te]
    Xm = Xf_mask[te]
    n = len(te)
    z = np.zeros(n, bool)
    n_aux_obs = (~np.isnan(Xf[te][:, D["nc"]:])).sum(1)
    taus = {"median": D["describe"]["y_median"], "q75": D["describe"]["y_q75"]}
    out = {}
    for name in TL_PREDICTORS:
        tl = TrustLayer(Prefit(models[name]), alpha=ALPHA, novelty_k=NOVELTY_K, novelty_level=NOVELTY_LEVEL)
        tl.fit(Xf[tr], y[tr], refit=False)
        tl.calibrate(Xf[cal], y[cal], groups=list(s_cal))
        st = tl.state()
        q_common = conformal_q(np.abs(y[cal] - models[name].predict(Xf[cal])), ALPHA)
        assert np.isclose(st["source"]["q"], q_common), "TrustLayer quantile != common.conformal_q"
        pval, flag = tl.shift_check(Xc, groups=list(s_te))
        aux_used = n_aux_obs if name != "hgb_comp" else np.zeros(n, int)
        res = {"flag_rate": float(flag.mean()),
               "flag_rate_per_pattern": {pattern_label(int(c), aux): float(flag[code_te == c].mean())
                                         for c in np.unique(code_te)},
               "tau": {}}
        tlm = None
        if name in FUSED:
            tlm = TrustLayer(Prefit(models[name]), alpha=ALPHA, novelty_k=NOVELTY_K, novelty_level=NOVELTY_LEVEL)
            tlm.fit(Xf[tr], y[tr], refit=False)
            tlm.calibrate(Xf_mask[cal], y[cal])                         # identically masked calibration
        g_te = list(s_te)
        g_borrow = [s_co] * n
        for tn, t in taus.items():
            arms = {}
            dP, dtP = tl.decide(Xc, t, flags=z, groups=None, return_details=True)
            dM, dtM = tl.decide(Xc, t, flags=z, groups=g_te, return_details=True)
            dMS, dtMS = tl.decide(Xc, t, flags=flag, groups=g_te, return_details=True)
            selP = tl.select_fdr(Xc, t, Q_FDR, flags=z, groups=None)
            selMJ = tl.select_fdr(Xc, t, Q_FDR, flags=z, groups=g_te, scope="joint")
            selMS = tl.select_fdr(Xc, t, Q_FDR, flags=z, groups=g_te, scope="stratum")

            def sel_dec(sel, d):
                return np.where(sel, ACCEPT, np.where(d == ACCEPT, MEASURE, d))

            spec = {"I_pooled": (dP, dtP, aux_used), "I_mondrian": (dM, dtM, aux_used),
                    "I_mondrian_shift": (dMS, dtMS, aux_used),
                    "S_pooled": (sel_dec(selP, dP), dtP, aux_used),
                    "S_mondrian_joint": (sel_dec(selMJ, dM), dtM, aux_used),
                    "S_mondrian_stratum": (sel_dec(selMS, dM), dtM, aux_used)}
            if tlm is not None:
                zero_aux = np.zeros(n, int)
                dN, dtN = tlm.decide(Xm, t, flags=z, return_details=True)
                selN = tlm.select_fdr(Xm, t, Q_FDR, flags=z)
                dB, dtB = tl.decide(Xm, t, flags=z, groups=g_borrow, return_details=True)
                dPN, dtPN = tl.decide(Xm, t, flags=z, groups=None, return_details=True)
                spec.update({"NYM_I_maskcal": (dN, dtN, zero_aux), "NYM_S_maskcal": (sel_dec(selN, dN), dtN, zero_aux),
                             "NYM_I_borrow": (dB, dtB, zero_aux), "NYM_I_pooled_nat": (dPN, dtPN, zero_aux)})
            for a, (d, det, au) in spec.items():
                # miscoverage is that of the interval of the same calibration (for S_* arms it
                # is reported for reference; G1 is asserted for interval-accept arms only)
                arms[a] = decision_metrics(np.asarray(d).astype(str), yc, t, det["lo"], det["hi"], code_te, aux, au)
                if a.startswith(("I_", "NYM_I")):                       # G1: false accept => miscovered
                    for blk in [arms[a]["all"]] + list(arms[a]["per_pattern"].values()):
                        if blk is not None and blk["miscoverage"] is not None:
                            assert blk["uncond_false_accept"] <= blk["miscoverage"] + 1e-12, (a, name, tn)
            res["tau"][tn] = {"value": float(t), "acceptable_share": float((yc >= t).mean()), "arms": arms}
        out[name] = res
    return out


# ============================================================ one split
def one_split(D, kind, seed):
    t0 = time.time()
    with threadpool_limits(limits=THREADS):
        ds, Xf, y, aux, code, nc = D["ds"], D["Xf"], D["y"], D["aux"], D["code"], D["nc"]
        sp = split_fusion(ds, seed, kind, units=D["units"], labs=D["labs"])
        tr, cal, te = sp["train"], sp["cal"], sp["test"]
        Xf_mask = Xf.copy()
        Xf_mask[:, nc:] = np.nan
        models = build_models(seed, nc, len(aux))
        mu, mu_mask = {}, {}
        for name, m in models.items():
            m.fit(Xf[tr], y[tr])
            mu[name] = {"cal": m.predict(Xf[cal]), "test": m.predict(Xf[te])}
            if name in FUSED:
                mu_mask[name] = {"cal": m.predict(Xf_mask[cal]), "test": m.predict(Xf_mask[te])}
        root_of, size, merges = mondrian_strata(code[cal], np.unique(code), len(aux))
        s_cal = stratum_labels(code[cal], root_of, size, aux, candidate=False)
        s_te = stratum_labels(code[te], root_of, size, aux, candidate=True)
        s_co = stratum_labels([0], root_of, size, aux, candidate=True)[0]
        out = {"design": D["key"], "split": kind, "seed": seed,
               "sizes": {k: int(len(v)) for k, v in sp.items()},
               "n_aug_rows": int(models["hgb_fused_drop"].n_aug_),
               "pattern_counts": {part: {pattern_label(int(c), aux): int((code[idx] == c).sum())
                                         for c in np.unique(code)} for part, idx in sp.items()},
               "strata": {"merges": {pattern_label(c, aux): pattern_label(p, aux) for c, p in merges.items()},
                          "n_cal": {pattern_label(r, aux): int(v) for r, v in size.items()},
                          "root_pooled_fallback": bool(size[root_of[0]] < N_MIN_STRATUM)},
               "overlap": {"test_lab_in_cal": float(np.isin(D["labs"][te], D["labs"][cal]).mean()),
                           "test_lab_in_train": float(np.isin(D["labs"][te], D["labs"][tr]).mean()),
                           "test_pub_in_cal": float(np.isin(D["kod"][te], D["kod"][cal]).mean()),
                           "test_pub_in_train": float(np.isin(D["kod"][te], D["kod"][tr]).mean()),
                           "cal_lab_in_train": float(np.isin(D["labs"][cal], D["labs"][tr]).mean()),
                           "test_comp_in_train_or_cal": float(np.isin(D["groups"][te], D["groups"][np.r_[tr, cal]]).mean()),
                           "test_n_labs": int(len(np.unique(D["labs"][te]))),
                           "cal_n_labs": int(len(np.unique(D["labs"][cal])))},
               "models": {}}
        Xcomp_te = Xf[te][:, :nc]
        for name in MODELS:
            out["models"][name] = {sch: conformal_block(sch, y[cal], mu[name]["cal"], s_cal, y[te], mu[name]["test"],
                                                        s_te, code[te], aux, Xcomp_te, D["y_iqr"], seed)
                                   for sch in ("pooled", "mondrian")}
        # worst-pattern references (what exact per-stratum / pooled calibration would give
        # with these calibration and test counts; binomial + calibration noise only)
        out["worst_pattern_reference"] = {}
        for thr, sfx in ((MIN_PATTERN_TEST, ""), (MIN_PATTERN_TEST2, "_ge%d" % MIN_PATTERN_TEST2)):
            pats_big = [c for c in np.unique(code[te]) if (code[te] == c).sum() >= thr]
            n_tests = [int((code[te] == c).sum()) for c in pats_big]
            strata_big = [str(s_te[code[te] == c][0]) for c in pats_big]
            ncal_of = {s: int((s_cal == s).sum()) if s != POOLED_LABEL else len(cal) for s in set(strata_big)}
            out["worst_pattern_reference"]["mondrian" + sfx] = worst_pattern_reference(n_tests, strata_big, ncal_of, seed=seed)
            out["worst_pattern_reference"]["pooled" + sfx] = worst_pattern_reference(
                n_tests, ["all"] * len(n_tests), {"all": len(cal)}, seed=seed)
        # width decomposition: population vs information (Mondrian widths per pattern)
        dec = {}
        for fz, cm in COMP_REF.items():
            pf = out["models"][fz]["mondrian"]["per_pattern"]
            pc = out["models"][cm]["mondrian"]["per_pattern"]
            co = pattern_label(0, aux)
            if co not in pf:
                continue
            d = {}
            for p in pf:
                if p == co:
                    continue
                d[p] = {"width_fused": pf[p]["width"], "width_comp_model": pc[p]["width"],
                        "total_ratio_vs_comp_only_glasses": pf[p]["width"] / pf[co]["width"],
                        "population_ratio_vs_comp_only_glasses": pc[p]["width"] / pc[co]["width"],
                        "information_ratio_fused_over_comp_model": pf[p]["width"] / pc[p]["width"],
                        "log_total": float(np.log(pf[p]["width"] / pf[co]["width"])),
                        "log_population": float(np.log(pc[p]["width"] / pc[co]["width"]))}
            dec[fz] = d
        out["width_decomposition"] = dec
        out["not_yet_measured"] = {name: masked_block(name, D, sp, mu, mu_mask, s_cal, s_te, s_co) for name in FUSED}
        if DESIGNS[D["key"]]["trust_layer"]:
            out["trust_layer"] = trust_block(D, sp, models, s_cal, s_te, Xf_mask, s_co)
        out["seconds"] = time.time() - t0
    return out


# ============================================================ aggregation
def _g(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _is_num(v):
    return isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) and not isinstance(v, str)


def agg_tree(vals):
    present = [v for v in vals if v is not None]
    if not present:
        return None
    f = present[0]
    if isinstance(f, dict):
        keys = list(dict.fromkeys(k for v in present if isinstance(v, dict) for k in v))
        return {k: agg_tree([v.get(k) if isinstance(v, dict) else None for v in vals]) for k in keys}
    if _is_num(f):
        arr = [float(v) if v is not None else None for v in vals]
        s = summarize(arr)                       # mean and 2.5-97.5 percentile over splits
        nm = sum(v is None for v in arr)
        nf = sum(v is not None and not np.isfinite(v) for v in arr)
        if nm:                                   # splits where the quantity did not exist
            s["n_missing"] = nm
        if nf:                                   # infinite intervals are reported, never dropped
            s["n_nonfinite"] = nf
        return s
    if isinstance(f, str):
        c = Counter(present)
        return f if len(c) == 1 else dict(c)
    if isinstance(f, list):
        if all(isinstance(v, list) and len(v) == len(f) and all(_is_num(x) for x in v) for v in present):
            return {"mean": np.mean(np.array(present, float), 0).tolist()}
        return f
    return str(f)


def paired_stats(a, b):
    """Paired split-to-split comparison of two schemes (b - a over the same splits).
    The worst-pattern minimum has a split-to-split SD of 0.1 or more on the
    laboratory-grouped split, so no pooled -> Mondrian statement is made without this."""
    a = np.asarray([np.nan if v is None else float(v) for v in a], float)
    b = np.asarray([np.nan if v is None else float(v) for v in b], float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = b - a
    # direction-neutral counts: which statistic counts as 'better' when it rises is decided by
    # HIGHER_IS_BETTER at report time, never here (a wider interval or a wider coverage range is
    # worse, a higher worst-pattern coverage is better).
    out = {"n": int(len(d)), "mean_a": float(a.mean()) if len(a) else None,
           "mean_b": float(b.mean()) if len(b) else None,
           "mean_diff": float(d.mean()) if len(d) else None,
           "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
           "n_increase": int((d > 0).sum()), "n_decrease": int((d < 0).sum()), "n_tie": int((d == 0).sum())}
    try:
        from scipy.stats import wilcoxon
        out["wilcoxon_p"] = float(wilcoxon(b, a).pvalue) if (d != 0).any() else 1.0
    except Exception:                                                  # pragma: no cover
        out["wilcoxon_p"] = None
    return out


def one_sided_t(vals, mu0):
    """One-sided one-sample t of the per-split values against a nominal level (H1: mean > mu0).
    Used so that 'the FDR guarantee is violated' is never claimed from a point estimate."""
    a = np.asarray([float(v) for v in vals if v is not None and np.isfinite(v)], float)
    if len(a) < 2:
        return None
    sd = float(a.std(ddof=1))
    se = sd / np.sqrt(len(a))
    out = {"mean": float(a.mean()), "sd": sd, "n": int(len(a)), "mu0": float(mu0)}
    if se == 0:
        out["t"], out["p_greater"] = None, (0.0 if a.mean() > mu0 else 1.0)
        return out
    from scipy.stats import t as tdist
    out["t"] = float((a.mean() - mu0) / se)
    out["p_greater"] = float(tdist.sf(out["t"], len(a) - 1))
    return out


def pooled_fdp(rows, path):
    """Ratio of totals over splits: sum false accepts / sum accepted (stable for small strata)."""
    fa = na = 0
    for r in rows:
        v = r
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                break
        if v:
            fa += v["false_accepts"]
            na += v["n_accept"]
    return {"false_accepts_total": fa, "accepted_total": na, "fdp_ratio_of_totals": fa / na if na else None}


def summarise_rows(rows):
    S = agg_tree(rows)
    S.pop("seed", None)
    S["strata"]["root_pooled_fallback_n_splits"] = int(sum(bool(r["strata"]["root_pooled_fallback"]) for r in rows))
    S["paired_pooled_vs_mondrian"] = {
        m: {st: paired_stats([r["models"][m]["pooled"].get(st) for r in rows],
                             [r["models"][m]["mondrian"].get(st) for r in rows])
            for st in ("coverage", "width", "interval_score", "worst_pattern_cov",
                       "worst_pattern_cov_ge%d" % MIN_PATTERN_TEST2, "best_pattern_cov", "pattern_cov_range")}
        for m in MODELS}
    if "trust_layer" in rows[0]:
        S["fdp_vs_q"] = {
            name: {tn: {a: one_sided_t([_g(r, "trust_layer", name, "tau", tn, "arms", a, "all", "fdp") for r in rows],
                                       Q_FDR)
                        for a in tv["arms"]}
                   for tn, tv in res["tau"].items()}
            for name, res in rows[0]["trust_layer"].items()}
    if "trust_layer" in rows[0]:
        tot = {}
        for name, res in rows[0]["trust_layer"].items():
            tot[name] = {}
            for tn, tv in res["tau"].items():
                tot[name][tn] = {}
                for a, av in tv["arms"].items():
                    tot[name][tn][a] = {"all": pooled_fdp(rows, ["trust_layer", name, "tau", tn, "arms", a, "all"]),
                                        "per_pattern": {p: pooled_fdp(rows, ["trust_layer", name, "tau", tn, "arms", a,
                                                                             "per_pattern", p])
                                                        for p in av["per_pattern"]}}
        S["trust_layer_totals"] = tot
    S["per_split_headline"] = [
        {"seed": r["seed"], "sizes": r["sizes"],
         **{f"{m}_{sch}_cov": r["models"][m][sch]["coverage"] for m in ("hgb_comp", "hgb_fused") for sch in ("pooled", "mondrian")},
         **{f"{m}_{sch}_worst_pattern_cov": r["models"][m][sch].get("worst_pattern_cov")
            for m in ("hgb_comp", "hgb_fused") for sch in ("pooled", "mondrian")},
         **{f"{m}_{sch}_pattern_cov_range": r["models"][m][sch].get("pattern_cov_range")
            for m in ("hgb_comp", "hgb_fused") for sch in ("pooled", "mondrian")},
         **{f"{m}_rmse": r["models"][m]["pooled"]["rmse"] for m in MODELS},
         "merges": r["strata"]["merges"]} for r in rows]
    S["merge_frequency"] = dict(Counter(f"{c}->{p}" for r in rows for c, p in r["strata"]["merges"].items()))
    return S


# ============================================================ main
def _dump(obj, out_dir, name):
    if out_dir is None:
        return dump(obj, name)
    path = os.path.join(out_dir, name)
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return path


def main(n_jobs=6, seeds=SEEDS, out_dir=None):
    """out_dir=None -> results/ (full run, splits persisted); else a scratch directory
    (smoke test; splits not persisted)."""
    t0 = time.time()
    res = {"config": {"alpha": ALPHA, "seeds": list(seeds), "n_min_stratum": N_MIN_STRATUM,
                      "min_pattern_test_for_worst": MIN_PATTERN_TEST, "wis_alphas": WIS_ALPHAS, "levels": LEVELS.tolist(),
                      "q_fdr": Q_FDR, "novelty_k": NOVELTY_K, "novelty_level": NOVELTY_LEVEL,
                      "models": {"hgb": "common.make_model('HistGB'): HistGradientBoostingRegressor(max_iter=400, "
                                        "learning_rate=0.05, early_stopping=False)",
                                 "rf": "common.make_model('RF'): RandomForestRegressor(300 trees, min_samples_leaf=2), "
                                       "n_jobs=%d (1 for bit-exact cross-process reproducibility)" % RF_N_JOBS,
                                 "rf_fused_imp": "SimpleImputer(strategy='median', add_indicator=True) + RF",
                                 "hgb_fused_drop": "training rows with >=1 auxiliary duplicated with all auxiliaries NaN"},
                      "splits": {"comp": "composition-grouped random 60/20/20, rng=default_rng(seed)",
                                 "lab": "whole laboratory-composition units: test>=20%, cal>=25% of rest, "
                                        "rng=default_rng(1000+seed)"},
                      "trustlayer_selftest_n_min": TLm.min_calibration_size(ALPHA)},
           "designs": {}}
    for key in DESIGNS:
        D = load_design(key)
        print(f"[{time.time() - t0:6.0f}s] {key}: n={D['describe']['n']} patterns={ {k: v['n'] for k, v in D['describe']['patterns'].items()} }"
              f" units={D['describe']['n_lab_composition_units']} largest={D['describe']['largest_unit_share']:.3f}", flush=True)
        R = {"describe": D["describe"], "splits": {}}
        for kind in SPLIT_KINDS:
            if out_dir is None:
                path = save_splits(D["ds"], split_fn=split_fusion, seeds=seeds, tag=f"{TAG}_{key}_{kind}", kind=kind,
                                   units=D["units"], labs=D["labs"])
            rows = Parallel(n_jobs=n_jobs)(delayed(one_split)(D, kind, s) for s in seeds)
            R["splits"][kind] = summarise_rows(rows)
            if out_dir is None:
                R["splits"][kind]["splits_file"] = os.path.relpath(path, os.path.dirname(RES)).replace("\\", "/")
            R["splits"][kind]["seconds_per_split"] = summarize([r["seconds"] for r in rows])
            h = R["splits"][kind]["models"]
            print(f"[{time.time() - t0:6.0f}s] {key} {kind}: RMSE comp {h['hgb_comp']['pooled']['rmse']['mean']:.2f} "
                  f"fused {h['hgb_fused']['pooled']['rmse']['mean']:.2f} drop {h['hgb_fused_drop']['pooled']['rmse']['mean']:.2f} "
                  f"rfimp {h['rf_fused_imp']['pooled']['rmse']['mean']:.2f} | fused cov pooled "
                  f"{h['hgb_fused']['pooled']['coverage']['mean']:.3f} mond {h['hgb_fused']['mondrian']['coverage']['mean']:.3f} "
                  f"| worst pat {h['hgb_fused']['pooled']['worst_pattern_cov']['mean']:.3f} -> "
                  f"{h['hgb_fused']['mondrian']['worst_pattern_cov']['mean']:.3f}", flush=True)
        res["designs"][key] = R
        _dump(res, out_dir, f"{TAG}.json")
    res["runtime_min"] = (time.time() - t0) / 60
    _dump(res, out_dir, f"{TAG}.json")
    write_report(res, out_dir)
    print(f"done in {res['runtime_min']:.1f} min", flush=True)


# ============================================================ report
def _m(s, fmt="{:.3f}", ci=True):
    if s is None or not isinstance(s, dict) or s.get("mean") is None:
        return "n/a"
    t = fmt.format(s["mean"])
    if ci and s.get("lo") is not None:
        t += " [" + fmt.format(s["lo"]) + ", " + fmt.format(s["hi"]) + "]"
    return t


SPLIT_NAME = {"comp": "composition-grouped random", "lab": "laboratory-grouped"}
MODEL_NAME = {"hgb_comp": "HGB composition-only", "hgb_fused": "HGB fused (native NaN)",
              "hgb_fused_drop": "HGB fused + modality dropout", "rf_comp": "RF composition-only",
              "rf_fused_imp": "RF fused (median imputation + indicators)"}
ARM_NAME = {"I_pooled": "interval, pooled calibration", "I_mondrian": "interval, Mondrian by pattern",
            "I_mondrian_shift": "interval, Mondrian + shift check",
            "S_pooled": "conformal selection, pooled (BH q=0.10)",
            "S_mondrian_joint": "conformal selection, Mondrian p-values, one BH",
            "S_mondrian_stratum": "conformal selection, BH inside each pattern",
            "NYM_I_maskcal": "not yet measured: interval, masked calibration",
            "NYM_S_maskcal": "not yet measured: selection, masked calibration",
            "NYM_I_borrow": "not yet measured: naive, borrowed comp-only quantile",
            "NYM_I_pooled_nat": "not yet measured: naive, pooled natural quantile"}
SEL_ARMS = ["S_pooled", "S_mondrian_joint", "S_mondrian_stratum"]
# Which direction of a paired difference counts as an improvement. None = neither (coverage
# should sit at 1 - alpha, not be large), so the split counts are labelled higher / lower.
HIGHER_IS_BETTER = {"coverage": None, "best_pattern_cov": None, "width": False,
                    "interval_score": False, "pattern_cov_range": False,
                    "worst_pattern_cov": True, "worst_pattern_cov_ge%d" % MIN_PATTERN_TEST2: True}


def sel_evidence(S, predictor="hgb_fused"):
    """All (threshold x scope) conformal-selection arms of one split family, ordered by the
    strength of the evidence that their realised FDP exceeds the nominal q."""
    rows = []
    for tn in ("median", "q75"):
        for a in SEL_ARMS:
            h = _g(S, "fdp_vs_q", predictor, tn, a) or {}
            t = _g(S, "trust_layer_totals", predictor, tn, a, "all") or {}
            rows.append({"tau": tn, "arm": a, "p": h.get("p_greater"), "mean": h.get("mean"),
                         "ratio": t.get("fdp_ratio_of_totals")})
    rows.sort(key=lambda r: (1e9 if r["p"] is None else r["p"]))
    return rows


def sel_sentence(S, cfg, predictor="hgb_fused", n_strongest=2):
    """Data-driven verdict on the (threshold x scope) selection arms of one split family: the
    realised FDP range, and which arms - if any - a one-sided t over the splits puts above the
    nominal q. Returns (verdict, sentence); the verdict is derived from the test, never asserted,
    so the prose can never contradict the numbers next to it."""
    ev = sel_evidence(S, predictor)
    rs = [r["ratio"] for r in ev if r["ratio"] is not None]
    sig = [r for r in ev if r["p"] is not None and r["p"] < 0.05]
    txt = (f"realised FDP {min(rs):.3f}-{max(rs):.3f} as a ratio of totals over the {len(ev)} "
           f"threshold x scope combinations; ")
    if sig:
        txt += (f"**{len(sig)} of {len(ev)} {'is' if len(sig) == 1 else 'are'} above q = {cfg['q_fdr']:.2f}** by a "
                "one-sided t over the splits"
                + (", the strongest being " if n_strongest else "")
                + "; ".join(f"*{ARM_NAME[r['arm']]}, tau = {r['tau']}*: {fdp_txt(S, predictor, r['tau'], r['arm'])}"
                            for r in sig[:n_strongest]))
    else:
        txt += f"**none of the {len(ev)} is above q = {cfg['q_fdr']:.2f}** by a one-sided t over the splits"
    return ("violated" if sig else "upheld"), txt


def fdp_txt(S, predictor, tn, arm, q=Q_FDR):
    """Realised selection FDP with its dispersion AND a one-sided t against the nominal q,
    so that 'holds' / 'violated' is never read off a point estimate."""
    v = _g(S, "trust_layer", predictor, "tau", tn, "arms", arm, "all", "fdp")
    t = _g(S, "trust_layer_totals", predictor, tn, arm, "all")
    h = _g(S, "fdp_vs_q", predictor, tn, arm)
    txt = _m(v, "{:.3f}") + " per split"
    if t and t.get("fdp_ratio_of_totals") is not None:
        txt += f", {t['fdp_ratio_of_totals']:.3f} as a ratio of totals"
    if h and h.get("p_greater") is not None:
        txt += (", one-sided t vs q = %.2f " % q) + ("p < 0.001" if h["p_greater"] < 0.001
                                                     else "p = %.3f" % h["p_greater"])
    return txt


def build_report(res):
    L = []
    w = L.append
    cfg = res["config"]
    n_splits = len(cfg["seeds"])
    w("# R6 - genuine, naturally incomplete multi-measurement fusion\n")
    w("Answers Reviewer 1 comment 3 (\"Figure 6 divides one composition vector into two feature groups\") and "
      "Reviewer 2 comment 6 (\"overly idealized ... difficult to consider representative of genuine "
      "biomedical-materials multimodal missingness\"), and integrates the four trust-layer components into one "
      "decision procedure for Reviewer 2 comment 4.\n")
    w("What is fused: **heterogeneous, naturally incomplete characterisation measurements (density, dilatometric "
      "expansion, refractive index) reported by different laboratories** in SciGlass. This is *not* imaging, "
      "spectral or mechanical multimodality: SciGlass contains no images or spectra, and that gap remains a stated "
      "limitation. The missingness is real rather than simulated - it is what was measured, reported **and captured "
      "in SciGlass** - except in the explicitly labelled 'measurement not yet made' deployment case, where "
      "measurements that WERE captured are masked at prediction time. SciGlass is a curated compilation (A2 section "
      "1), so an absent density is not proof that the density was never measured: it may have been reported in the "
      "source paper and not captured by the curators. What the experiment needs is only that the missingness is "
      "*real and structured*, which section 1 demonstrates by computation, not that it is a pure record of "
      "laboratory intent; the curation layer is listed as a limitation in section 8.\n")
    w(f"Protocol: `revision/code/PROTOCOL.md` and `common.py`; alpha = {cfg['alpha']}, finite-sample conformal "
      f"quantile, {n_splits} splits per split family, every quantity computed per split and reported as "
      "**mean [2.5, 97.5 percentile over splits]**. The splits resample one finite dataset, so the interval "
      "describes split-to-split variability, not population sampling error. Script: "
      "`revision/code/exp_R6_fusion.py`; results: `revision/results/R6_fusion.json`. No figures.\n")

    # ---------------------------------------------------------------- bottom line
    def PP(key, kind, model, sch):
        return res["designs"][key]["splits"][kind]["models"][model][sch]["per_pattern"]

    def spread(key, kind, model, sch):
        it = [(p, v["coverage"]["mean"]) for p, v in PP(key, kind, model, sch).items()
              if v["n"]["mean"] >= MIN_PATTERN_TEST]
        lo, hi = min(it, key=lambda t: t[1]), max(it, key=lambda t: t[1])
        return f"{lo[1]:.3f} ({lo[0]}) to {hi[1]:.3f} ({hi[0]})"

    def MS(key, kind, *path):
        return _g(res["designs"][key]["splits"][kind], *path)

    def PAIR(key, kind, model, stat):
        return _g(res["designs"][key]["splits"][kind], "paired_pooled_vs_mondrian", model, stat)

    def pair_txt(key, kind, model, stat, fmt="{:+.3f}"):
        """Paired split-to-split difference Mondrian - pooled. The split counts are labelled
        better/worse only for a statistic with a defined direction (HIGHER_IS_BETTER); for
        coverage, which should be close to 1 - alpha rather than large, they are labelled
        higher/lower."""
        p = PAIR(key, kind, model, stat)
        if not p or p.get("mean_diff") is None:
            return "n/a"
        wp = "n/a" if p.get("wilcoxon_p") is None else (
            "p < 0.001" if p["wilcoxon_p"] < 0.001 else "p = %.3f" % p["wilcoxon_p"])
        hib = HIGHER_IS_BETTER.get(stat)
        up, dn = ("higher", "lower") if hib is None else (("better", "worse") if hib else ("worse", "better"))
        return (f"paired difference {fmt.format(p['mean_diff'])}, SD {p['sd_diff']:.3f}, "
                f"{p['n_increase']} of {p['n']} splits {up} / {p['n_decrease']} {dn}, Wilcoxon {wp}")

    def wpc(key, kind, model, sch):
        return _m(MS(key, kind, "models", model, sch, "worst_pattern_cov"))

    def rng_dir(key, kind, model="hgb_fused"):
        p = PAIR(key, kind, model, "pattern_cov_range")
        if not p or p.get("mean_diff") is None:
            return "n/a"
        return "narrows" if p["mean_diff"] < 0 else ("widens" if p["mean_diff"] > 0 else "is unchanged")

    def _dirs(kind):
        return sorted({rng_dir(k, kind) for k in res["designs"]})

    rangedir_sentence = (
        ("On both random splits this statistic %s; on both laboratory-grouped splits it %s, i.e. the "
         "pattern-conditional quantiles make the per-split spread of coverage worse, not better, once the "
         "laboratories are held out." % (_dirs("comp")[0], _dirs("lab")[0]))
        if len(_dirs("comp")) == 1 and len(_dirs("lab")) == 1 and _dirs("comp") != _dirs("lab")
        else "The direction of this statistic is not uniform across the four design x split families; the paired "
             "tests above give each one.")

    w("\n## 0. Bottom line\n")
    w("1. **The missingness is genuine and strongly informative.** Tg (B1) co-occurs with density, dilatometric "
      f"expansion and refractive index in {len(res['designs']['B1_Tg']['describe']['patterns'])} natural patterns of "
      f"{min(v['n'] for v in res['designs']['B1_Tg']['describe']['patterns'].values())}-"
      f"{max(v['n'] for v in res['designs']['B1_Tg']['describe']['patterns'].values())} records each; which "
      "measurements exist is a proxy for glass family and for the reporting practice of the source laboratory "
      "(MNAR), not a random mask. Computed, not asserted: a B1 laboratory with at least "
      f"{res['designs']['B1_Tg']['describe']['practice_concentration']['min_records']} records carries "
      f"{100 * res['designs']['B1_Tg']['describe']['practice_concentration']['laboratories']['mean_modal_share']:.0f} % "
      "of them in a single missing pattern, against "
      f"{100 * res['designs']['B1_Tg']['describe']['practice_concentration']['laboratories']['permuted_baseline']:.0f} % "
      "when the patterns are permuted (section 1).\n")
    w("2. **Fusing real auxiliary measurements buys very little accuracy.** RMSE, composition-only -> fused HGB: "
      f"B1 {MS('B1_Tg', 'comp', 'models', 'hgb_comp', 'pooled', 'rmse')['mean']:.2f} -> "
      f"{MS('B1_Tg', 'comp', 'models', 'hgb_fused', 'pooled', 'rmse')['mean']:.2f} K on the composition-grouped "
      f"random split and {MS('B1_Tg', 'lab', 'models', 'hgb_comp', 'pooled', 'rmse')['mean']:.2f} -> "
      f"{MS('B1_Tg', 'lab', 'models', 'hgb_fused', 'pooled', 'rmse')['mean']:.2f} K across held-out laboratories; "
      f"B2 {MS('B2_E', 'comp', 'models', 'hgb_comp', 'pooled', 'rmse')['mean']:.2f} -> "
      f"{MS('B2_E', 'comp', 'models', 'hgb_fused', 'pooled', 'rmse')['mean']:.2f} GPa and "
      f"{MS('B2_E', 'lab', 'models', 'hgb_comp', 'pooled', 'rmse')['mean']:.2f} -> "
      f"{MS('B2_E', 'lab', 'models', 'hgb_fused', 'pooled', 'rmse')['mean']:.2f} GPa. The RF-with-imputation family "
      "behaves the same way. They buy almost nothing in decisions either: in the end-to-end procedure of section 7 "
      f"({MS('B1_Tg', 'comp', 'sizes', 'test')['mean']:.0f} candidates, random split), "
      f"{_g(B1c0 := res['designs']['B1_Tg']['splits']['comp'], 'trust_layer', 'hgb_fused', 'tau', 'median', 'arms', 'I_mondrian', 'all', 'aux_measurements_used')['mean']:.0f} "
      "already-existing auxiliary values save "
      f"{_g(B1c0, 'trust_layer', 'hgb_comp', 'tau', 'median', 'arms', 'I_mondrian', 'all', 'tg_measurements')['mean'] - _g(B1c0, 'trust_layer', 'hgb_fused', 'tau', 'median', 'arms', 'I_mondrian', 'all', 'tg_measurements')['mean']:.0f} "
      f"of {_g(B1c0, 'trust_layer', 'hgb_comp', 'tau', 'median', 'arms', 'I_mondrian', 'all', 'tg_measurements')['mean']:.0f} "
      "Tg measurements. The reportable effect of the auxiliaries is not accuracy, and not measurement savings; it is "
      "what an honest uncertainty statement has to look like when records are characterised to different depths.\n")
    w("3. **Pooled conformal hides pattern-conditional miscoverage; Mondrian-by-pattern narrows the spread of "
      "per-pattern coverage on i.i.d.-style splits, and does not repair it across held-out laboratories.** Three "
      "distinct statistics are involved and all three are given on both split families, so that they are never "
      f"mixed. Over patterns with >= {MIN_PATTERN_TEST} test records: (a) the range of the MEAN per-pattern "
      "coverage (average each pattern over the splits first, then take max - min: the stable statistic, but it has "
      "no per-split value and therefore no paired test); (b) the mean over splits of the per-split RANGE (max - min "
      "within a split, so it carries the binomial noise of the small patterns, but it can be tested pairwise); "
      "(c) the mean over splits of the per-split WORST-PATTERN MINIMUM, the noisiest of the three. (b) and (c) are "
      "always quoted with their percentile interval and their paired split-to-split difference. (a) and (b) need "
      "not move in the same direction - (b) is a max - min of noisy per-split estimates, (a) averages each pattern "
      "first - which is exactly why they are named apart and never substituted for one another.\n")
    w("   (a) Range of the mean per-pattern coverage, fused model, pooled -> Mondrian: B1 random split "
      f"{spread('B1_Tg', 'comp', 'hgb_fused', 'pooled')} -> {spread('B1_Tg', 'comp', 'hgb_fused', 'mondrian')}; "
      f"B1 laboratory-grouped {spread('B1_Tg', 'lab', 'hgb_fused', 'pooled')} -> "
      f"{spread('B1_Tg', 'lab', 'hgb_fused', 'mondrian')}; B2 random split "
      f"{spread('B2_E', 'comp', 'hgb_fused', 'pooled')} -> {spread('B2_E', 'comp', 'hgb_fused', 'mondrian')}; "
      f"B2 laboratory-grouped {spread('B2_E', 'lab', 'hgb_fused', 'pooled')} -> "
      f"{spread('B2_E', 'lab', 'hgb_fused', 'mondrian')}.\n")
    w("   (b) Mean per-split range, pooled -> Mondrian: "
      + "; ".join(
          f"{k.split('_')[0]} {'random' if kd == 'comp' else 'laboratory'} "
          f"{MS(k, kd, 'models', 'hgb_fused', 'pooled', 'pattern_cov_range')['mean']:.3f} -> "
          f"{MS(k, kd, 'models', 'hgb_fused', 'mondrian', 'pattern_cov_range')['mean']:.3f} "
          f"({pair_txt(k, kd, 'hgb_fused', 'pattern_cov_range')})"
          for k in res["designs"] for kd in res["designs"][k]["splits"])
      + ". " + rangedir_sentence + "\n")
    w("   (c) Worst-pattern minimum, pooled -> Mondrian: "
      f"B1 random {wpc('B1_Tg', 'comp', 'hgb_fused', 'pooled')} -> {wpc('B1_Tg', 'comp', 'hgb_fused', 'mondrian')} "
      f"({pair_txt('B1_Tg', 'comp', 'hgb_fused', 'worst_pattern_cov')}); "
      f"B1 laboratory {wpc('B1_Tg', 'lab', 'hgb_fused', 'pooled')} -> {wpc('B1_Tg', 'lab', 'hgb_fused', 'mondrian')} "
      f"({pair_txt('B1_Tg', 'lab', 'hgb_fused', 'worst_pattern_cov')}); "
      f"B2 random {wpc('B2_E', 'comp', 'hgb_fused', 'pooled')} -> {wpc('B2_E', 'comp', 'hgb_fused', 'mondrian')} "
      f"({pair_txt('B2_E', 'comp', 'hgb_fused', 'worst_pattern_cov')}); "
      f"B2 laboratory {wpc('B2_E', 'lab', 'hgb_fused', 'pooled')} -> {wpc('B2_E', 'lab', 'hgb_fused', 'mondrian')} "
      f"({pair_txt('B2_E', 'lab', 'hgb_fused', 'worst_pattern_cov')}). The sampling-noise reference for this "
      f"statistic is {MS('B1_Tg', 'lab', 'worst_pattern_reference', 'mondrian')['mean']:.3f} (B1 laboratory split).\n")
    moved = [(k, kd) for k in res["designs"] for kd in res["designs"][k]["splits"]
             if (PAIR(k, kd, "hgb_fused", "worst_pattern_cov") or {}).get("wilcoxon_p") is not None
             and PAIR(k, kd, "hgb_fused", "worst_pattern_cov")["wilcoxon_p"] < 0.05]
    moved_rng = [(k, kd) for k in res["designs"] for kd in res["designs"][k]["splits"]
                 if (PAIR(k, kd, "hgb_fused", "pattern_cov_range") or {}).get("wilcoxon_p") is not None
                 and PAIR(k, kd, "hgb_fused", "pattern_cov_range")["wilcoxon_p"] < 0.05]
    n_fam = sum(len(R["splits"]) for R in res["designs"].values())
    w(f"   Read together: the worst-pattern minimum moves significantly (paired Wilcoxon p < 0.05) in {len(moved)} "
      f"of the {n_fam} design x split families"
      + (" (" + ", ".join(f"{k}, {SPLIT_NAME[kd]} split" for k, kd in moved) + ")" if moved else "")
      + "; everywhere else it is inside split noise, in either direction. The per-split range moves significantly in "
      + (", ".join(f"{k}, {SPLIT_NAME[kd]} split ({rng_dir(k, kd)})" for k, kd in moved_rng)
         if moved_rng else f"none of the {n_fam}")
      + ". **On the laboratory-grouped families the evidence supports *'Mondrian does not repair per-pattern "
        "coverage across laboratories'*.** It does not support a claim that Mondrian measurably lowers the "
        "worst-pattern minimum there (both paired tests are far from significant); the only significant "
        "laboratory-split movement of any of the three statistics is the widening of B2's per-split range, which "
        "points the same way as the absence of a repair.\n")
    w("   The mechanism is nevertheless visible per pattern. Mondrian narrows the interval of a well-characterised "
      "stratum (B1, rho+CTE: "
      f"{PP('B1_Tg', 'lab', 'hgb_fused', 'mondrian')['rho+CTE']['width']['mean']:.0f} K versus "
      f"{PP('B1_Tg', 'lab', 'hgb_fused', 'pooled')['rho+CTE']['width']['mean']:.0f} K pooled), and under a "
      "laboratory shift that narrower interval then under-covers "
      f"({PP('B1_Tg', 'lab', 'hgb_fused', 'mondrian')['rho+CTE']['coverage']['mean']:.3f} versus "
      f"{PP('B1_Tg', 'lab', 'hgb_fused', 'pooled')['rho+CTE']['coverage']['mean']:.3f}). Pattern-conditional "
      "calibration is not a substitute for handling the cross-laboratory shift.\n")
    w("4. **The width differences between patterns are population differences, not information gain.** For B1's "
      "best-characterised pattern (rho+CTE) the fused intervals are "
      f"{_g(res['designs']['B1_Tg']['splits']['comp'], 'width_decomposition', 'hgb_fused', 'rho+CTE', 'total_ratio_vs_comp_only_glasses')['mean']:.2f}x "
      "the width of composition-only glasses, but a model that never sees an auxiliary value already gives "
      f"{_g(res['designs']['B1_Tg']['splits']['comp'], 'width_decomposition', 'hgb_fused', 'rho+CTE', 'population_ratio_vs_comp_only_glasses')['mean']:.2f}x "
      "on the same glasses; the measured values themselves contribute only "
      f"{_g(res['designs']['B1_Tg']['splits']['comp'], 'width_decomposition', 'hgb_fused', 'rho+CTE', 'information_ratio_fused_over_comp_model')['mean']:.2f}x. "
      "Across held-out laboratories the population term accounts for the width difference entirely (information "
      f"ratio {_m(_g(res['designs']['B1_Tg']['splits']['lab'], 'width_decomposition', 'hgb_fused', 'rho+CTE', 'information_ratio_fused_over_comp_model'), '{:.2f}')}), "
      "so this item is the one place where the laboratory-grouped split strengthens rather than weakens the "
      "statement. The paper must not say that the auxiliary measurements 'halve' the uncertainty (section 5).\n")
    w("5. **'Not yet measured' is a different regime from 'naturally missing', and it is fixable.** Masking every "
      "candidate's auxiliaries at prediction time costs the native-NaN model "
      f"{MS('B1_Tg', 'comp', 'not_yet_measured', 'hgb_fused', 'mask_all', 'rmse_masked')['mean']:.1f} K versus "
      f"{MS('B1_Tg', 'comp', 'models', 'hgb_comp', 'pooled', 'rmse')['mean']:.1f} K for a composition-only model "
      "on the same glasses; modality-dropout training removes the penalty "
      f"({MS('B1_Tg', 'comp', 'not_yet_measured', 'hgb_fused_drop', 'mask_all', 'rmse_masked')['mean']:.1f} K). "
      "Every calibration number in this item is for the **modality-dropout model**, on B1's random split. "
      "Calibrating on identically masked calibration records gives "
      f"{MS('B1_Tg', 'comp', 'not_yet_measured', 'hgb_fused_drop', 'mask_all', 'maskcal', 'coverage')['mean']:.3f} "
      f"coverage at {MS('B1_Tg', 'comp', 'not_yet_measured', 'hgb_fused_drop', 'mask_all', 'maskcal', 'width')['mean']:.0f} K; "
      "borrowing the quantile of naturally composition-only glasses gives "
      f"{MS('B1_Tg', 'comp', 'not_yet_measured', 'hgb_fused_drop', 'mask_all', 'borrow_comp_only', 'coverage')['mean']:.3f} "
      f"at {MS('B1_Tg', 'comp', 'not_yet_measured', 'hgb_fused_drop', 'mask_all', 'borrow_comp_only', 'width')['mean']:.0f} K "
      "(valid but needlessly wide); reusing the pooled natural quantile gives "
      f"{MS('B1_Tg', 'comp', 'not_yet_measured', 'hgb_fused_drop', 'mask_all', 'pooled_natural', 'coverage')['mean']:.3f} "
      f"at {MS('B1_Tg', 'comp', 'not_yet_measured', 'hgb_fused_drop', 'mask_all', 'pooled_natural', 'width')['mean']:.0f} K "
      "for the same model, and "
      f"{MS('B1_Tg', 'comp', 'not_yet_measured', 'hgb_fused', 'mask_all', 'pooled_natural', 'coverage')['mean']:.3f} "
      "for the native-NaN model, whose accuracy the masking does degrade - the naive recipe is the one that turns a "
      "model-quality problem into an invalid interval. The calibration set must reproduce the deployment-time "
      "missingness, not the historical one (full table in section 6).\n")
    vc = sel_sentence(res["designs"]["B1_Tg"]["splits"]["comp"], cfg)
    vl = sel_sentence(res["designs"]["B1_Tg"]["splits"]["lab"], cfg)
    w("6. **End to end, the layer's guarantee is marginal, and it degrades exactly where the data stop being "
      "exchangeable.** With the specification Tg >= median, interval-based acceptance keeps the overall false-accept "
      f"proportion at {_g(res['designs']['B1_Tg']['splits']['comp'], 'trust_layer_totals', 'hgb_fused', 'median', 'I_mondrian', 'all')['fdp_ratio_of_totals']:.3f} "
      "(random split), but per pattern it reaches "
      f"{max(v['fdp_ratio_of_totals'] for v in _g(res['designs']['B1_Tg']['splits']['comp'], 'trust_layer_totals', 'hgb_fused', 'median', 'I_mondrian', 'per_pattern').values() if v['accepted_total'] >= 20):.3f}; "
      f"conformal selection was tested against its nominal q = {cfg['q_fdr']:.2f} on both split families with a "
      "one-sided t over the splits (JSON `fdp_vs_q`); the verdicts below are that test, not point estimates. "
      f"Composition-grouped random split - guarantee {vc[0]}: {vc[1]}. Laboratory-grouped split - guarantee "
      f"{vl[0]}: {vl[1]}. Per pattern the realised FDP reaches "
      f"{max(v['fdp_ratio_of_totals'] for v in _g(res['designs']['B1_Tg']['splits']['lab'], 'trust_layer_totals', 'hgb_fused', 'median', 'S_mondrian_stratum', 'per_pattern').values() if v['accepted_total'] >= 20):.3f} "
      "in one pattern (laboratory split). Section 7 and section 8 give every arm.\n")

    # ---------------------------------------------------------------- data
    w("\n## 1. Designs, measurements and natural missing patterns\n")
    for key, R in res["designs"].items():
        d = R["describe"]
        w(f"\n**{key}** - target {DESIGNS[key]['target']} ({d['n']} records, {d['n_unique_compositions']} unique "
          f"compositions, {d['n_labs']} laboratories, {d['n_publications']} publications), inputs = "
          f"{d['n_features_composition']} element atomic fractions + auxiliaries "
          + ", ".join(f"{a} ({d['aux_observed'][a]} records)" for a in d["aux"]) + ". "
          f"IQR({DESIGNS[key]['target']}) = {d['y_iqr']:.3g} {DESIGNS[key]['unit']}.\n")
        w("| pattern (what was measured) | n | share | " + DESIGNS[key]['target'] + " mean +/- SD | labs | publications | median year |")
        w("|---|---|---|---|---|---|---|")
        for p, v in d["patterns"].items():
            w(f"| {p} | {v['n']} | {100 * v['n'] / d['n']:.1f} % | {v['y_mean']:.4g} +/- {v['y_sd']:.3g} | "
              f"{v['n_labs']} | {v['n_publications']} | {v['median_year']:.0f} |")
        w("")
        w("Auxiliary screen (counts and Pearson r recomputed here; verdicts fixed a priori, A2 section 4.1):\n")
        w("| candidate auxiliary | co-occurring records | r with target | verdict |")
        w("|---|---|---|---|")
        for a, v in d["aux_screen"].items():
            r = "n/a" if v["pearson_r"] is None else f"{v['pearson_r']:+.2f}"
            w(f"| {a} | {v['n_with_target']} | {r} | {v['verdict']} |")
        w("")
    w("The patterns are strongly non-random (MNAR): pattern-specific target means differ by up to "
      f"{max(abs(a['y_mean'] - b['y_mean']) for a in res['designs']['B1_Tg']['describe']['patterns'].values() for b in res['designs']['B1_Tg']['describe']['patterns'].values()):.0f} K for B1. "
      "That the pattern also tracks who produced the record is **computed here, not asserted**: for every laboratory "
      f"or publication with at least {res['designs']['B1_Tg']['describe']['practice_concentration']['min_records']} "
      "records, the share of its records carried by its single most frequent missing pattern, against the same "
      "statistic after permuting the pattern labels over all records "
      f"({res['designs']['B1_Tg']['describe']['practice_concentration']['n_permutations']} permutations).\n")
    w("| design | unit | groups | records in them | mean share of the modal pattern | groups >= 80 % one pattern | "
      "permuted baseline |")
    w("|---|---|---|---|---|---|---|")
    for key, R in res["designs"].items():
        pc = R["describe"]["practice_concentration"]
        for nm in ("laboratories", "publications"):
            v = pc[nm]
            w(f"| {key} | {nm} (>= {pc['min_records']} records) | {v['n_groups']} | {v['n_records_in_groups']} | "
              f"{100 * v['mean_modal_share']:.1f} % | {100 * v['share_ge_80pct']:.1f} % | "
              f"{100 * v['permuted_baseline']:.1f} % |")
    w("\nA laboratory therefore reports one characterisation set and stays with it, and a single publication almost "
      "always reports exactly one - which is why the missing pattern behaves like a grouping variable and why "
      "pattern-conditional calibration matters below. (A2 section 4.2 illustrates the same point with patent and "
      "industrial series reporting density and expansion and optical papers reporting the refractive index. That "
      "attribution is an illustration from the A2 scoping study; it is **not** recomputed here. A2's corporate-name "
      "string pattern, re-applied to B1, matches "
      f"{res['designs']['B1_Tg']['describe']['practice_concentration']['corporate_author_check']['n_records']} records "
      f"resolving to {(_nk := res['designs']['B1_Tg']['describe']['practice_concentration']['corporate_author_check']['n_normalised_author_keys'])} "
      f"normalised author {'key' if _nk == 1 else 'keys'}, which is far too thin to establish the attribution, so "
      "this report does not use it as evidence.)\n")

    # ---------------------------------------------------------------- splits
    w("\n## 2. Splits and the dependence they do (not) remove\n")
    w(f"| design | split family | train / cal / test | test records whose laboratory also occurs in cal | "
      f"... in train | test records whose publication occurs in cal | test laboratories |")
    w("|---|---|---|---|---|---|---|")
    for key, R in res["designs"].items():
        for kind, S in R["splits"].items():
            o, sz = S["overlap"], S["sizes"]
            w(f"| {key} | {SPLIT_NAME[kind]} | {sz['train']['mean']:.0f} / {sz['cal']['mean']:.0f} / "
              f"{sz['test']['mean']:.0f} | {_m(o['test_lab_in_cal'], '{:.2f}')} | {_m(o['test_lab_in_train'], '{:.2f}')} | "
              f"{_m(o['test_pub_in_cal'], '{:.2f}')} | {_m(o['test_n_labs'], '{:.0f}')} |")
    w("\nIn the composition-grouped random split a composition never straddles a split, but most test records still "
      "share a laboratory and a publication with calibration records - the dependence the A1 audit (section 6.3) "
      "found between recalibration and test records. The laboratory-grouped split removes it completely (the "
      "grouping unit is a connected component of the laboratory-composition graph, so laboratories that published "
      "an identical composition are held out together; largest unit "
      f"{100 * res['designs']['B1_Tg']['describe']['largest_unit_share']:.0f} % of B1 records). **Which split a "
      "number comes from is stated everywhere, and every table reports both.** The laboratory-grouped family is the "
      "deployment-realistic one and carries the negative conclusions (sections 3, 7, 8); the random split is the "
      "optimistic bound and is where the positive calibration effect of section 3 and the trust-layer arms of "
      "section 7 are legible at all. Section 5's width decomposition is quoted from the random split in section 0 "
      "because that is the conservative direction for its claim - the laboratory-grouped split makes the "
      "'information' contribution of the measurements smaller still, not larger.\n")
    w("Mondrian strata: a pattern with fewer than %d calibration records is merged into its parent pattern with one "
      "fewer measurement (rule fixed a priori, calibration counts only). Merges actually applied, over all splits:\n"
      % cfg["n_min_stratum"])
    w("| design | split family | merges (child -> parent: number of splits out of %d) |" % n_splits)
    w("|---|---|---|")
    for key, R in res["designs"].items():
        for kind, S in R["splits"].items():
            mf = S.get("merge_frequency") or {}
            w(f"| {key} | {SPLIT_NAME[kind]} | " + (", ".join(f"{k}: {v}" for k, v in sorted(mf.items())) or "none") + " |")
    w("\nAfter a merge the coverage statement is marginal over the merged stratum, not conditional on the pattern; "
      "the per-pattern tables name the stratum each pattern used.\n")
    fb = {(key, kind): S["strata"].get("root_pooled_fallback_n_splits", 0)
          for key, R in res["designs"].items() for kind, S in R["splits"].items()}
    any_fb = {k: v for k, v in fb.items() if v}
    if any_fb:
        w("One further case has to be recorded: if the merged composition-only ROOT itself ends up with fewer than "
          f"{cfg['n_min_stratum']} calibration records, its members (composition-only plus everything merged into it) "
          "fall back to the pooled quantile, so that split's 'Mondrian' arm is partly pooled. This happened in "
          + ", ".join(f"{v} of {n_splits} {key} {SPLIT_NAME[kind]} splits" for (key, kind), v in sorted(any_fb.items()))
          + " (whole laboratory-composition units can strip the root); everywhere else the root kept its own "
            "quantile. The occurrence is in the JSON as `strata.root_pooled_fallback`.\n")
    else:
        w(f"The merged composition-only root never fell below {cfg['n_min_stratum']} calibration records in any "
          "split, so no stratum ever fell back to the pooled quantile (`strata.root_pooled_fallback` is 0 "
          "throughout).\n")

    # ---------------------------------------------------------------- accuracy + calibration
    w("\n## 3. Accuracy and calibration, pooled versus Mondrian-by-pattern\n")
    for key, R in res["designs"].items():
        unit = DESIGNS[key]["unit"]
        for kind, S in R["splits"].items():
            w(f"\n**{key}, {SPLIT_NAME[kind]} split** ({n_splits} splits)\n")
            w(f"| model | RMSE ({unit}) | pooled cov. | pooled width ({unit}) | Mondrian cov. | Mondrian width ({unit}) | "
              "pooled interval score | Mondrian interval score | mean per-SPLIT per-pattern cov. range, pooled -> "
              "Mondrian |")
            w("|---|---|---|---|---|---|---|---|---|")
            for m in MODELS:
                P, M = S["models"][m]["pooled"], S["models"][m]["mondrian"]
                w(f"| {MODEL_NAME[m]} | {_m(P['rmse'], '{:.2f}')} | {_m(P['coverage'])} | {_m(P['width'], '{:.1f}')} | "
                  f"{_m(M['coverage'])} | {_m(M['width'], '{:.1f}')} | {_m(P['interval_score'], '{:.1f}', False)} | "
                  f"{_m(M['interval_score'], '{:.1f}', False)} | {_m(P['pattern_cov_range'], '{:.3f}', False)} -> "
                  f"{_m(M['pattern_cov_range'], '{:.3f}', False)} |")
            w("")
            w("The last column is the max - min of per-pattern coverage **within a split**, averaged over splits "
              "(so it carries the binomial noise of the small patterns). The range of the per-pattern coverage "
              "MEANS, which averages first and is therefore smaller, is in section 0 item 3(a) and in the "
              "per-pattern tables of section 4; the two can move in opposite directions and are never quoted "
              "interchangeably. Paired test of the per-split range (Mondrian minus pooled), fused HGB: "
              + pair_txt(key, kind, "hgb_fused", "pattern_cov_range") + ".\n")
            w("Worst-pattern coverage for the same models - the noisiest statistic in this report, so it is given "
              "with its percentile interval and with the PAIRED split-to-split difference (Mondrian minus pooled "
              f"over the same {n_splits} splits):\n")
            w("| model | worst-pattern cov. pooled | worst-pattern cov. Mondrian | paired difference |")
            w("|---|---|---|---|")
            for m in MODELS:
                P, M = S["models"][m]["pooled"], S["models"][m]["mondrian"]
                w(f"| {MODEL_NAME[m]} | {_m(P['worst_pattern_cov'])} | {_m(M['worst_pattern_cov'])} | "
                  f"{pair_txt(key, kind, m, 'worst_pattern_cov')} |")
            ref = S["worst_pattern_reference"]
            w(f"\nReference for those two tables: with these calibration and test counts, exact calibration would give "
              f"a worst-pattern coverage of {_m(ref['pooled'], '{:.3f}', False)} (pooled) and "
              f"{_m(ref['mondrian'], '{:.3f}', False)} (Mondrian) by sampling noise alone; restricted to patterns "
              f"with >= {MIN_PATTERN_TEST2} test records the observed values are "
              f"{_m(S['models']['hgb_fused']['pooled']['worst_pattern_cov_ge50'], '{:.3f}', False)} -> "
              f"{_m(S['models']['hgb_fused']['mondrian']['worst_pattern_cov_ge50'], '{:.3f}', False)} for the fused "
              f"model against a reference of {_m(ref['mondrian_ge50'], '{:.3f}', False)} "
              f"({pair_txt(key, kind, 'hgb_fused', 'worst_pattern_cov_ge%d' % MIN_PATTERN_TEST2)}). A worst-pattern "
              "difference whose paired Wilcoxon p is well above 0.05 is split noise and must not be read as a "
              "measured improvement or a measured degradation; the coverage-range column above is the stable "
              "statistic.\n")
    # full metric set for the primary model
    w("\nFull metric set (Reviewer 2 comment 12) for the fused HGB model:\n")
    w("| design | split | calibration | coverage | width | width / IQR | interval score | WIS | calib. error (mean) | "
      "worst width-bin cov. | worst k-means cluster cov. | rel. AURC | RMSE reduction on 50 % most confident |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, R in res["designs"].items():
        for kind, S in R["splits"].items():
            for sch in ("pooled", "mondrian"):
                B = S["models"]["hgb_fused"][sch]
                rc = B.get("risk_coverage") or {}
                w(f"| {key} | {kind} | {sch} | {_m(B['coverage'])} | {_m(B['width'], '{:.1f}', False)} | "
                  f"{_m(B['norm_width'], '{:.2f}', False)} | {_m(B['interval_score'], '{:.1f}', False)} | "
                  f"{_m(B['wis'], '{:.1f}', False)} | {_m(B['ce_mean'], '{:.3f}', False)} | "
                  f"{_m(B['wsc_width_strat'], '{:.3f}', False)} | {_m(B.get('worst_cluster_cov'), '{:.3f}', False)} | "
                  f"{_m(rc.get('aurc_rel'), '{:.3f}', False) if rc else 'n/a (constant width)'} | "
                  f"{_m(rc.get('reduction_at50_pct'), '{:.1f}', False) + ' %' if rc else 'n/a (constant width)'} |")
    w("\nThe pooled scheme gives every candidate the same width, so it cannot rank candidates at all (no "
      "risk-coverage curve); the Mondrian scheme ranks them only at the resolution of the missing pattern.\n")

    # ---------------------------------------------------------------- per pattern
    w("\n## 4. Per-pattern results (the reviewers' question)\n")
    for key, R in res["designs"].items():
        unit = DESIGNS[key]["unit"]
        for kind, S in R["splits"].items():
            F, C = S["models"]["hgb_fused"], S["models"]["hgb_comp"]
            w(f"\n**{key}, {SPLIT_NAME[kind]} split** - fused HGB unless stated; the composition-only model's "
              "Mondrian width is in parentheses.\n")
            w(f"| pattern | n test | stratum used | RMSE comp-only -> fused ({unit}) | pooled cov. | Mondrian cov. | "
              f"Mondrian width ({unit}) | pooled width ({unit}) | pooled interval score | Mondrian interval score |")
            w("|---|---|---|---|---|---|---|---|---|---|")
            for p in F["mondrian"]["per_pattern"]:
                fm, fp = F["mondrian"]["per_pattern"][p], F["pooled"]["per_pattern"][p]
                cm = C["mondrian"]["per_pattern"][p]
                w(f"| {p} | {_m(fm['n'], '{:.0f}', False)} | {fm['stratum'] if isinstance(fm['stratum'], str) else 'varies'} | "
                  f"{_m(cm['rmse'], '{:.1f}', False)} -> {_m(fm['rmse'], '{:.1f}', False)} | {_m(fp['coverage'])} | "
                  f"{_m(fm['coverage'])} | {_m(fm['width'], '{:.1f}', False)} ({_m(cm['width'], '{:.1f}', False)}) | "
                  f"{_m(fp['width'], '{:.1f}', False)} | {_m(fp['interval_score'], '{:.1f}', False)} | "
                  f"{_m(fm['interval_score'], '{:.1f}', False)} |")
            miss = {p: v["n"]["n_missing"] for p, v in F["mondrian"]["per_pattern"].items()
                    if isinstance(v.get("n"), dict) and v["n"].get("n_missing")}
            if miss:
                w("\nA pattern that has no test record in a given split contributes nothing to that split, and its "
                  f"row is the mean over the splits in which it does appear: "
                  + ", ".join(f"{p} absent in {v} of {n_splits} splits" for p, v in sorted(miss.items()))
                  + " (`n_missing` in the JSON).\n")
    w("")

    # ---------------------------------------------------------------- width decomposition
    w("\n## 5. Width differences between patterns are mostly population differences\n")
    w("Ratios of Mondrian widths, relative to the composition-only glasses of the same split. "
      "*Total* = fused model, pattern p versus composition-only glasses (what a reader would see); "
      "*population* = the same ratio for the composition-only MODEL, which uses none of the auxiliary values and "
      "therefore measures only how homogeneous that sub-population is; *information* = fused / composition-only "
      "model within the pattern, the part actually contributed by the measured values. The last column is the share "
      "of the log width difference explained by the population term.\n")
    for key, R in res["designs"].items():
        for kind, S in R["splits"].items():
            dec = _g(S, "width_decomposition", "hgb_fused")
            if not dec:
                continue
            w(f"\n**{key}, {SPLIT_NAME[kind]} split**\n")
            w("| pattern | total ratio | population ratio | information ratio | population share of log difference |")
            w("|---|---|---|---|---|")
            for p, v in dec.items():
                lt, lp = v["log_total"]["mean"], v["log_population"]["mean"]
                share = f"{100 * lp / lt:.0f} %" if abs(lt) > 0.05 else "n/a (no width difference)"
                w(f"| {p} | {_m(v['total_ratio_vs_comp_only_glasses'], '{:.2f}')} | "
                  f"{_m(v['population_ratio_vs_comp_only_glasses'], '{:.2f}')} | "
                  f"{_m(v['information_ratio_fused_over_comp_model'], '{:.2f}')} | {share} |")
    w("\nThe laboratory-grouped split is the stronger version of this result, not the weaker one: for B1's "
      "rho+CTE pattern the information ratio is "
      f"{_m(_g(res['designs']['B1_Tg']['splits']['lab'], 'width_decomposition', 'hgb_fused', 'rho+CTE', 'information_ratio_fused_over_comp_model'), '{:.2f}')} "
      "across held-out laboratories against "
      f"{_m(_g(res['designs']['B1_Tg']['splits']['comp'], 'width_decomposition', 'hgb_fused', 'rho+CTE', 'information_ratio_fused_over_comp_model'), '{:.2f}')} "
      "on the random split - i.e. once the laboratory is held out the measured values contribute nothing at all to "
      "the width.\n")
    w("\n**The paper must not claim that the auxiliary measurements 'halve' the uncertainty.** Richly characterised "
      "glasses are simply more homogeneous families; a composition-only model produces nearly the same narrow "
      "intervals for them. The honest statement is that pattern-conditional calibration makes the interval width "
      "track what is known about a glass, and that the measurements themselves contribute the small 'information' "
      "column.\n")

    # ---------------------------------------------------------------- not yet measured
    w("\n## 6. 'The measurement has not been made yet' is not the same as 'naturally missing'\n")
    w("All auxiliaries of the test records are masked at prediction time. `masked calibration` calibrates on "
      "calibration records masked identically; the two naive recipes reuse natural calibration residuals.\n")
    for key, R in res["designs"].items():
        unit = DESIGNS[key]["unit"]
        for kind, S in R["splits"].items():
            w(f"\n**{key}, {SPLIT_NAME[kind]} split** - every test record masked\n")
            w(f"| fused model | RMSE natural | RMSE masked | RMSE of the composition-only model | "
              f"masked cal.: cov. / width | naive borrowed comp-only quantile: cov. / width | "
              f"naive pooled natural quantile: cov. / width |")
            w("|---|---|---|---|---|---|---|")
            for m in FUSED:
                A = S["not_yet_measured"][m]["mask_all"]
                w(f"| {MODEL_NAME[m]} | {_m(A['rmse_natural'], '{:.2f}', False)} | {_m(A['rmse_masked'], '{:.2f}')} | "
                  f"{_m(A['rmse_comp_model'], '{:.2f}', False)} | {_m(A['maskcal']['coverage'])} / "
                  f"{_m(A['maskcal']['width'], '{:.1f}', False)} | {_m(A['borrow_comp_only']['coverage'])} / "
                  f"{_m(A['borrow_comp_only']['width'], '{:.1f}', False)} | {_m(A['pooled_natural']['coverage'])} / "
                  f"{_m(A['pooled_natural']['width'], '{:.1f}', False)} |")
    w("\nPer pattern (A2's design: only the records of one natural pattern are masked, calibration on the records of "
      "that pattern's stratum masked the same way), for both designs and both split families:\n")
    for key, R in res["designs"].items():
        unit = DESIGNS[key]["unit"]
        for kind, S in R["splits"].items():
            bp = _g(S, "not_yet_measured", "hgb_fused", "by_pattern")
            bd = _g(S, "not_yet_measured", "hgb_fused_drop", "by_pattern")
            if not bp:
                continue
            w(f"\n**{key}, {SPLIT_NAME[kind]} split**\n")
            w(f"| pattern | n test | RMSE with the measurements | RMSE masked, no augmentation -> with modality dropout | "
              f"RMSE of the composition-only model | masked cal. cov. / width (dropout model) | "
              f"borrowed comp-only quantile cov. / width (dropout model) |")
            w("|---|---|---|---|---|---|---|")
            for p in bp:
                a, b = bp[p], bd[p]
                w(f"| {p} | {_m(a['n_test'], '{:.0f}', False)} | {_m(a['rmse_with_aux'], '{:.1f}', False)} | "
                  f"{_m(a['rmse_masked'], '{:.1f}', False)} -> {_m(b['rmse_masked'], '{:.1f}', False)} | "
                  f"{_m(a['rmse_comp_model'], '{:.1f}', False)} | {_m(b['masked_q_maskcal']['coverage'])} / "
                  f"{_m(b['masked_q_maskcal']['width'], '{:.1f}', False)} | "
                  f"{_m(b['masked_q_borrow_comp_only']['coverage'])} / "
                  f"{_m(b['masked_q_borrow_comp_only']['width'], '{:.1f}', False)} |")
    w("")

    # ---------------------------------------------------------------- trust layer
    w("\n## 7. The trust layer as one decision procedure (Reviewer 2 comment 4)\n")
    key = "B1_Tg"
    R = res["designs"][key]
    w(f"Candidates = the test records of each split; specification Tg >= tau with tau fixed a priori as the median "
      f"({R['describe']['y_median']:.1f} K) and the 75th percentile ({R['describe']['y_q75']:.1f} K) of Tg over the "
      f"whole B1 dataset; `trustlayer.TrustLayer` (imported unchanged) with the missing pattern as the Mondrian "
      f"group label, alpha = {cfg['alpha']}, conformal selection at q = {cfg['q_fdr']}, novelty level "
      f"{cfg['novelty_level']}. FDP = false accepts / accepted (a false accept is an accepted candidate whose true "
      f"Tg is below tau); the table gives the ratio of totals over all splits. 'Measurements' counts candidates sent "
      f"to a Tg measurement (decision 'measure' or 'not certified'); 'auxiliary measurements' counts the density / "
      f"expansion / refractive-index values the arm consumed.\n")
    for kind, S in R["splits"].items():
        for tn in ("median", "q75"):
            w(f"\n**{SPLIT_NAME[kind]} split, tau = {tn}** (fused HGB predictor; "
              f"{_m(_g(S, 'trust_layer', 'hgb_fused', 'tau', tn, 'acceptable_share'), '{:.2f}', False)} of candidates "
              f"actually satisfy the specification)\n")
            w(f"Counts are **per split** (mean over {n_splits} splits); the two FDP columns are **ratios of totals "
              f"over all {n_splits} splits**, and the n in the worst-pattern column is that pattern's accepted total "
              "over all splits, not a per-split count.\n")
            w("| arm | accepted (per split) | FDP among accepted (all splits) | power (per split) | measured "
              "(per split) | auxiliary measurements (per split) | worst-pattern FDP (all splits, n = accepted total) "
              "| interval miscoverage (per split) |")
            w("|---|---|---|---|---|---|---|---|")
            for a in ARM_NAME:
                A = _g(S, "trust_layer", "hgb_fused", "tau", tn, "arms", a)
                T = _g(S, "trust_layer_totals", "hgb_fused", tn, a)
                if A is None:
                    continue
                pp = {p: v["fdp_ratio_of_totals"] for p, v in T["per_pattern"].items()
                      if v["fdp_ratio_of_totals"] is not None and v["accepted_total"] >= 20}
                wp = max(pp, key=pp.get) if pp else None
                worst = (f"{pp[wp]:.3f} ({wp}, n={T['per_pattern'][wp]['accepted_total']})" if pp else "n/a")
                w(f"| {ARM_NAME[a]} | {_m(A['all']['n_accept'], '{:.0f}', False)} | "
                  f"{T['all']['fdp_ratio_of_totals']:.3f} | {_m(A['all']['power'], '{:.3f}', False)} | "
                  f"{_m(A['all']['tg_measurements'], '{:.0f}', False)} | "
                  f"{_m(A['all']['aux_measurements_used'], '{:.0f}', False)} | {worst} | "
                  f"{_m(A['all']['miscoverage'], '{:.3f}', False)} |")
    w("\nPer-pattern false-accept proportions (ratio of totals over splits, patterns with at least 20 accepted "
      "candidates in total), fused HGB, tau = median:\n")
    for kind, S in R["splits"].items():
        pats = list(res["designs"][key]["describe"]["patterns"])
        w(f"\n**{SPLIT_NAME[kind]} split**\n")
        w("| arm | " + " | ".join(pats) + " |")
        w("|---" * (len(pats) + 1) + "|")
        for a in ["I_pooled", "I_mondrian", "S_pooled", "S_mondrian_joint", "S_mondrian_stratum"]:
            T = _g(S, "trust_layer_totals", "hgb_fused", "median", a)
            cells = []
            for p in pats:
                v = (T or {}).get("per_pattern", {}).get(p)
                cells.append("n/a" if not v or v["accepted_total"] < 20 else
                             f"{v['fdp_ratio_of_totals']:.3f} (n={v['accepted_total']})")
            w(f"| {ARM_NAME[a]} | " + " | ".join(cells) + " |")
    def _ratio(kind, tn, a):
        v = _g(R["splits"][kind], "trust_layer_totals", "hgb_fused", tn, a, "all") or {}
        return v.get("fdp_ratio_of_totals")

    stratum_worse = all((_ratio("lab", tn, "S_mondrian_stratum") or 0) > (_ratio("lab", tn, "S_pooled") or 0)
                        for tn in ("median", "q75"))
    w("\nReading of section 7. (i) Interval-based acceptance is very conservative overall - it accepts only when the "
      "whole interval clears tau - so its false-accept proportion stays far below alpha, but that is a marginal "
      "statement: per pattern it reaches the values in the table below, which is exactly the coverage-versus-FDR gap "
      "documented in `trustlayer.py` (G1). (ii) Conformal selection accepts far more candidates at a controlled "
      f"error rate; against q = {cfg['q_fdr']:.2f} its guarantee is **{sel_sentence(R['splits']['comp'], cfg)[0]}** "
      f"on the composition-grouped split and **{sel_sentence(R['splits']['lab'], cfg)[0]}** on the "
      "laboratory-grouped one (the table below gives every arm). (iii) Where it fails, it fails because a candidate "
      "from an unseen laboratory is not exchangeable with the calibration records of its stratum. "
      + ("Running BH inside each pattern makes it worse, not better - the realised FDP of the per-stratum scope is "
         "above the pooled scope's at both thresholds on the laboratory-grouped split - because the smaller strata "
         "carry the laboratory offset undiluted.\n" if stratum_worse else
         "On this run the per-stratum scope is not uniformly worse than the pooled one across the thresholds, so "
         "that ordering is not claimed here; the table below gives both.\n"))
    w("\nBecause a realised FDP is itself a noisy quantity, every selection arm is tested against its nominal q with "
      f"a one-sided one-sample t over the {n_splits} splits (JSON `fdp_vs_q`). 'Upheld' and 'violated' anywhere in "
      "this report mean this test, not a point estimate:\n")
    w("| split family | tau | arm | realised FDP (mean [2.5, 97.5] over splits) | ratio of totals | one-sided t vs "
      f"q = {cfg['q_fdr']:.2f} |")
    w("|---|---|---|---|---|---|")
    for kind, S in R["splits"].items():
        for tn in ("median", "q75"):
            for a in SEL_ARMS:
                v = _g(S, "trust_layer", "hgb_fused", "tau", tn, "arms", a, "all", "fdp")
                t = _g(S, "trust_layer_totals", "hgb_fused", tn, a, "all")
                h = _g(S, "fdp_vs_q", "hgb_fused", tn, a) or {}
                p = h.get("p_greater")
                ptxt = "n/a" if p is None else ("p < 0.001" if p < 0.001 else "p = %.3f" % p)
                if p is not None:
                    ptxt += " (%s)" % ("above q" if p < 0.05 else "not distinguishable from q")
                w(f"| {SPLIT_NAME[kind]} | {tn} | {ARM_NAME[a]} | {_m(v, '{:.3f}')} | "
                  f"{t['fdp_ratio_of_totals']:.3f} | {ptxt} |")
    w("(iv) The auxiliary measurements barely pay for themselves in decisions either.\n")
    for kind, S in R["splits"].items():
        A = lambda p, a: _g(S, "trust_layer", p, "tau", "median", "arms", a, "all")  # noqa: E731
        saved = A('hgb_comp', 'I_mondrian')['tg_measurements']['mean'] - A('hgb_fused', 'I_mondrian')['tg_measurements']['mean']
        sv = (f"saves {saved:.1f} Tg measurements" if saved >= 1.5 else
              f"saves {saved:.1f} Tg measurements on average, i.e. effectively none" if saved > 0 else
              f"actually requests {-saved:.1f} MORE Tg measurements")
        w(f"On the "
          f"{SPLIT_NAME[kind]} split (tau = median, Mondrian intervals), the composition-only model sends "
          f"{A('hgb_comp', 'I_mondrian')['tg_measurements']['mean']:.0f} of "
          f"{S['sizes']['test']['mean']:.0f} candidates to a Tg measurement; the fused model sends "
          f"{A('hgb_fused', 'I_mondrian')['tg_measurements']['mean']:.0f} while consuming "
          f"{A('hgb_fused', 'I_mondrian')['aux_measurements_used']['mean']:.0f} existing auxiliary values, i.e. it "
          f"{sv} in exchange, and its power moves from {A('hgb_comp', 'I_mondrian')['power']['mean']:.3f} to "
          f"{A('hgb_fused', 'I_mondrian')['power']['mean']:.3f}. The modality-dropout model deciding with nothing "
          f"measured yet (masked calibration) needs "
          f"{A('hgb_fused_drop', 'NYM_I_maskcal')['tg_measurements']['mean']:.0f} measurements at power "
          f"{A('hgb_fused_drop', 'NYM_I_maskcal')['power']['mean']:.3f}, against "
          f"{A('hgb_fused_drop', 'I_mondrian')['tg_measurements']['mean']:.0f} and "
          f"{A('hgb_fused_drop', 'I_mondrian')['power']['mean']:.3f} with the auxiliaries in hand.\n")
    w("\nShift check inside the same procedure: the conformal novelty flag fires on "
      + ", ".join(f"{_m(_g(R['splits'][k], 'trust_layer', 'hgb_fused', 'flag_rate'), '{:.3f}', False)} of candidates "
                  f"({SPLIT_NAME[k]})" for k in R["splits"])
      + ". Composition-and-auxiliary novelty is a covariate-space test: it does not detect a laboratory offset, "
        "which is a label shift (A2 section 2.5), so it cannot rescue the laboratory-grouped coverage on its own.\n")
    w("\nPredictor families other than the fused HGB behave the same way; the JSON holds every arm for "
      + ", ".join(MODEL_NAME[m] for m in TL_PREDICTORS) + ".\n")

    # ---------------------------------------------------------------- claims
    B1c, B1l = res["designs"]["B1_Tg"]["splits"]["comp"], res["designs"]["B1_Tg"]["splits"]["lab"]
    w("\n## 8. What the manuscript may and may not claim from this experiment\n")
    w("**May be claimed.**\n")
    w("* The fusion experiment uses *heterogeneous, naturally incomplete characterisation measurements (density, "
      "dilatometric expansion, refractive index) reported by different laboratories*, with "
      f"{len(res['designs']['B1_Tg']['describe']['patterns'])} missing patterns that occur in the literature, "
      "not a composition vector split into two halves and not a simulated mask (answers R1-3 and R2-6 on the "
      "'idealized' character of the old Figure 6).\n")
    w("* Missingness is informative (MNAR): the pattern predicts the property and tracks the reporting practice of "
      "the source laboratory (section 1, computed against a permutation baseline), so a single pooled conformal "
      "quantile over-covers well-characterised glasses and under-covers composition-only ones; Mondrian-by-pattern "
      "calibration narrows the spread of per-pattern coverage on i.i.d.-style splits and makes the width track what "
      "was measured. Say 'measured, reported and captured in SciGlass', not 'chosen to measure'.\n")
    w("* The deployment case matters: a model with native NaN handling has learned 'missing means a different glass "
      "family', so masking a measurement that simply has not been made yet degrades it; modality-dropout training "
      "plus calibration on identically masked calibration records restores both accuracy and valid, tight "
      "intervals.\n")
    w("* The four components run as one procedure on one object (uncertainty, shift check, decision/acquisition, "
      "missing-modality strata), and the experiment reports what that procedure actually delivers, including where "
      "the guarantee does not hold.\n")
    w("\n**Must not be claimed.**\n")
    w("* Not imaging, spectral or histological multimodality. SciGlass has no images or spectra. State it as a "
      "limitation, and do not generalise from three scalar characterisation measurements to multimodal biomedical "
      "data.\n")
    w("* Not an accuracy story: the fused model gains "
      f"{100 * (1 - MS('B1_Tg', 'comp', 'models', 'hgb_fused', 'pooled', 'rmse')['mean'] / MS('B1_Tg', 'comp', 'models', 'hgb_comp', 'pooled', 'rmse')['mean']):.0f} % "
      "RMSE on B1's random split and nothing across laboratories.\n")
    w("* The auxiliaries do not 'halve' or otherwise materially shrink the uncertainty of a given glass. Almost all "
      "of the between-pattern width difference is a population difference (section 5).\n")
    def rng_verdict(key, kind, model="hgb_fused"):
        """Verdict on the per-pattern coverage RANGE, derived from its paired test."""
        p = PAIR(key, kind, model, "pattern_cov_range")
        if not p or p.get("wilcoxon_p") is None:
            return "n/a"
        if p["wilcoxon_p"] >= 0.05:
            return "shows no significant change"
        return "narrows significantly" if p["mean_diff"] < 0 else "widens significantly"

    w("* Pattern-conditional calibration is not a general fix. Per-SPLIT per-pattern coverage range (the statistic "
      f"that can be tested pairwise): on the composition-grouped random split it {rng_verdict('B1_Tg', 'comp')} for "
      f"B1 ({pair_txt('B1_Tg', 'comp', 'hgb_fused', 'pattern_cov_range')}) and {rng_verdict('B2_E', 'comp')} for B2 "
      f"({pair_txt('B2_E', 'comp', 'hgb_fused', 'pattern_cov_range')}); across held-out laboratories it "
      f"{rng_verdict('B1_Tg', 'lab')} for B1 ({pair_txt('B1_Tg', 'lab', 'hgb_fused', 'pattern_cov_range')}) and "
      f"{rng_verdict('B2_E', 'lab')} for B2 ({pair_txt('B2_E', 'lab', 'hgb_fused', 'pattern_cov_range')}). The "
      "range of the per-pattern coverage MEANS on the laboratory split is B1 "
      f"{spread('B1_Tg', 'lab', 'hgb_fused', 'pooled')} pooled versus "
      f"{spread('B1_Tg', 'lab', 'hgb_fused', 'mondrian')} Mondrian, B2 "
      f"{spread('B2_E', 'lab', 'hgb_fused', 'pooled')} versus {spread('B2_E', 'lab', 'hgb_fused', 'mondrian')} - a "
      "different statistic from the per-split range (it averages each pattern first), and on B1 it moves the other "
      "way, which is why neither is quoted without its name. "
      "The worst-pattern minimum does not improve either, and its laboratory-split movement is not individually "
      f"significant (B1 {pair_txt('B1_Tg', 'lab', 'hgb_fused', 'worst_pattern_cov')}; B2 "
      f"{pair_txt('B2_E', 'lab', 'hgb_fused', 'worst_pattern_cov')}). **The defensible claim is that across "
      "laboratories Mondrian does not bring the worst pattern back to the nominal level - not that it measurably "
      "lowers that minimum** (the one significant laboratory-split movement of any of these statistics is the "
      "widening of B2's per-split range, section 0 item 3(b)). Whatever it does to the range there, the worst "
      "pattern stays at "
      f"{MS('B1_Tg', 'lab', 'models', 'hgb_fused', 'mondrian', 'worst_pattern_cov')['mean']:.3f} (B1) and "
      f"{MS('B2_E', 'lab', 'models', 'hgb_fused', 'mondrian', 'worst_pattern_cov')['mean']:.3f} (B2) against a "
      f"nominal {1 - cfg['alpha']:.2f}.\n")
    ev_l = sel_evidence(B1l)
    sig_l = [r for r in ev_l if r["p"] is not None and r["p"] < 0.05]
    weak_l = [r for r in ev_l if r["p"] is not None and r["p"] >= 0.05 and (r["mean"] or 0) > cfg["q_fdr"]]
    vc8, vl8 = sel_sentence(B1c, cfg, n_strongest=1), sel_sentence(B1l, cfg, n_strongest=0)
    w(f"* The FDR guarantee of the selection arm is {vl8[0]} once whole laboratories are held out, and {vc8[0]} "
      f"when they are not; the verdict is a one-sided t of the realised FDP against q over the {n_splits} splits, "
      f"not a point estimate. Composition-grouped random split: {vc8[1]}. Laboratory-grouped split: {vl8[1]}.\n")
    if sig_l:
        w("  **Quote the arms with the strongest evidence, not the mildest one**, in that order: "
          + "; ".join(f"*{ARM_NAME[r['arm']]}, tau = {r['tau']}*: {fdp_txt(B1l, 'hgb_fused', r['tau'], r['arm'])}"
                      for r in sig_l) + ".\n")
    if weak_l:
        w(f"  Arms whose realised FDP exceeds q but which are **not individually distinguishable from q** at "
          f"{n_splits} splits must be qualified as such wherever they are quoted: "
          + "; ".join(f"{ARM_NAME[r['arm']]}, tau = {r['tau']}: {fdp_txt(B1l, 'hgb_fused', r['tau'], r['arm'])}"
                      for r in weak_l) + ".\n")
    w("  Cross-laboratory shift is a label/measurement shift (A2 section 2.5) and needs the target-recalibration "
      "machinery of the other experiments, not a Mondrian stratum.\n")
    w("* Nothing here supports the retrospective '65-90 % fewer measurements' claim of the old Figure 11; that is a "
      "separate, retrospective replay experiment.\n")
    w("\n**Suggested wording.** \"We fuse composition with heterogeneous, naturally incomplete characterisation "
      "measurements - density, dilatometric expansion and refractive index - reported by different laboratories, "
      f"which co-occur with the glass-transition temperature in {len(res['designs']['B1_Tg']['describe']['patterns'])} "
      "natural missingness patterns. Fusion changes accuracy only marginally; what changes is the reliability of the "
      "uncertainty statement. A single pooled conformal quantile over-covers richly characterised glasses "
      f"({PP('B1_Tg', 'comp', 'hgb_fused', 'pooled')['rho+CTE']['coverage']['mean']:.2f}) and under-covers glasses "
      f"described by composition alone ({PP('B1_Tg', 'comp', 'hgb_fused', 'pooled')['comp-only']['coverage']['mean']:.2f}), "
      "while calibrating separately within each missingness pattern brings every pattern close to the nominal level "
      f"({spread('B1_Tg', 'comp', 'hgb_fused', 'mondrian')}). The narrower intervals of well-characterised glasses "
      "are, however, mostly a property of those sub-populations rather than information contributed by the "
      "measurements themselves, and when whole laboratories are held out the pattern-conditional calibration no "
      "longer equalises coverage at all.\"\n")
    w("\n**Limitations of this experiment.** (0) **SciGlass is a curated compilation, so the missingness is "
      "'measured, reported and captured', not 'measured and reported'.** A missing density may mean the source "
      "paper reported one that the curators did not transcribe; A2 section 1 documents the compilation character, "
      "and A2 section 3 finds that 27-32 % of cross-laboratory duplicate pairs report an identical Tg, consistent "
      "with re-reported or compiled values. Nothing in this experiment requires the missingness to be a pure record "
      "of laboratory intent - only that it is real, structured and correlated with the property, which section 1 "
      "establishes by computation - but the paper must not describe the pattern as what a laboratory *chose* to "
      "measure. (i) The laboratory is a normalised first-author key, a proxy. (ii) The "
      "dilatometric expansion coefficient can come from the same instrument run as a dilatometric Tg, so that "
      "auxiliary is not fully instrument-independent; the viscosity iso-points and crystallisation temperatures that "
      "are near-label leakage were excluded a priori (section 1). (iii) Patterns are unbalanced, so worst-pattern "
      "statistics carry binomial noise; the reference columns quantify it. (iv) All splits resample one finite "
      "retrospective database; the percentile intervals describe split-to-split variability only. (v) The "
      "specification thresholds are dataset quantiles fixed a priori, not a clinical requirement, and none of the "
      "conformal statements is a safety statement (trustlayer.py, G1-G3).\n")

    # ---------------------------------------------------------------- fix round
    w("\n## 9. Fix round: what an independent verification changed\n")
    w("The numerical pipeline was independently re-implemented and reproduced exactly (same split indices, same "
      "per-split values); nothing in the computation changed in this round. What changed is what the report claims "
      "from it:\n")
    w("1. **Cross-model number substitution repaired (severe).** Bottom line 5 quoted two masked-calibration "
      "numbers for the modality-dropout model and the third for the native-NaN model, and called it 'invalid'. It "
      "now names the model for every number and gives both models' pooled-natural-quantile coverage.\n")
    w("2. **Worst-pattern coverage is no longer printed without dispersion.** Section 3 now shows the per-pattern "
      "coverage RANGE in the main table and the worst-pattern minimum in a separate table with percentile intervals "
      "and the paired split-to-split difference (mean, SD, splits better / worse, Wilcoxon). The bottom line was "
      "rewritten accordingly: section 0 now names the design x split families in which the minimum actually moves, "
      "and the laboratory-split changes, being inside split noise, support the ABSENCE of a repair rather than a "
      "measured degradation.\n")
    w("3. **One named statistic per sentence.** Bottom line 3 previously quoted the coverage range for the random "
      "split and the worst-pattern minimum for the laboratory split. Three statistics are now defined explicitly - "
      "the range of the per-pattern coverage means, the mean per-split range, and the per-split worst-pattern "
      "minimum - and all three are given for all four design x split families. The second and third disagree in "
      "direction on B1's laboratory split, which is precisely why the original mixing was a defect.\n")
    w("4. **The root-fallback statement was wrong and is corrected.** The script's docstring claimed the merged "
      "composition-only root never fell below the minimum calibration size. Section 2 above now reports the actual "
      "occurrences from `strata.root_pooled_fallback`.\n")
    w("5. **The missingness mechanism is no longer over-attributed.** SciGlass is a curated compilation, so the "
      "wording is 'measured, reported and captured in SciGlass'; database curation is limitation (0).\n")
    w("6. **An illustrative claim is no longer presented as a finding.** The 'patent and industrial series report "
      "density and expansion; optical papers report the refractive index' sentence was not computed anywhere. It is "
      "replaced by a computed laboratory/publication concentration statistic with a permutation baseline "
      "(section 1), and A2's attribution is explicitly marked as an illustration.\n")
    w("7. **The FDR violation is demonstrated with the decisive cases.** Every selection arm now carries a "
      "one-sided t against the nominal q (section 7 table); section 8 orders the laboratory-split evidence by the "
      "strength of that test, quotes the decisive arms first, and explicitly qualifies any arm whose realised FDP "
      "exceeds q but is not individually distinguishable from it. Every 'upheld' / 'violated' verdict in this "
      "report is now emitted by the test rather than written into the prose.\n")
    w("8. **Cosmetic fixes:** the split-provenance promise in section 2 now matches what sections 0 and 5 actually "
      "quote; the leftover caption fragment before the per-pattern masking tables is gone; the measurement-saving "
      "sentence no longer reads 'saves 1 Tg measurements'; per-pattern tables footnote the patterns that are absent "
      "from some splits' test sets; and the section 7 arm tables state which columns are per-split means and which "
      "are totals over all splits.\n")
    w("9. **Two sources of run-to-run drift found while re-running, and removed.** The Beta-binomial reference for "
      "the worst-pattern coverage drew its random numbers in `set` iteration order, which varies with the process "
      "hash seed and moved that reference by about 0.001 between runs; and RandomForest's parallel tree "
      "accumulation is not bit-reproducible across processes (about 2e-13), which is enough to flip a "
      "width-stratified-coverage bin on an exact tie. Both are fixed (sorted iteration, RF `n_jobs = 1`), and the "
      "JSON is now bit-exact when the script is run twice. No reported digit changed.\n")
    w("\n**Verifier issues not adopted: none.** Every issue raised was accepted and fixed. Two were corrections of "
      "fact (items 1 and 4), three were over-claims relative to the dispersion or the provenance of the evidence "
      "(items 2, 5 and 7), one replaced an uncomputed assertion with a computation (item 6), and the rest were "
      "presentational. The verifier also listed inaccuracies confined to the implementer's summary message rather "
      "than to these outputs; those are corrected in the summary returned with this run.\n")
    return "\n".join(L) + "\n"


def write_report(res, out_dir=None):
    txt = build_report(res)
    with open(os.path.join(out_dir or RES, f"{TAG}.md"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    if "--report" in sys.argv:
        src = os.environ.get("R6_OUT_DIR") or RES
        write_report(json.load(open(os.path.join(src, f"{TAG}.json"))), os.environ.get("R6_OUT_DIR"))
    elif "--quick" in sys.argv:
        main(n_jobs=4, seeds=SEEDS[:2], out_dir=os.environ["R6_OUT_DIR"])
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        main(int(args[0]) if args else 6)
