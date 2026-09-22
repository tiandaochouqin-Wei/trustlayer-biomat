"""R9 (legacy Figure 10): conformal risk control (CRC) of the false-negative rate for
glass-forming-ability (GFA) classification, re-run under the unified protocol.

Data: demo/glass_gfa_magpie.npz (matbench_glass, 5,680 compositions, 132 Magpie
descriptors, y = 1 glass-former). Row alignment with matminer's matbench_glass is
verified here (labels and a sample of re-featurised rows), which also gives the
composition string of each row for the chemical-system sensitivity split.

Split (per seed 0..29): grouped 60 / 20 / 20 train / calibration / test. Groups =
identical Magpie vectors (common.composition_groups); all 5,680 rows are unique, so
this equals a random split. Sensitivity: grouped by chemical system (sorted element
set, 398 systems), so a test composition's chemical system is never seen in training
or calibration.

Model: HistGradientBoostingClassifier(max_iter = 300, learning_rate = 0.08), the
submitted hyperparameters, fixed a priori; fitted on train only.

CRC (Angelopoulos et al., ICLR 2024), loss on a true glass-former = 1{p < t} (a miss,
bounded by B = 1). Decision: flag glass-former iff p >= t. The threshold is the largest
t with (n R_hat(t) + B) / (n + 1) <= alpha, n = number of calibration positives,
chosen on the calibration split only; guarantee E[FNR_test] <= alpha (marginal over
calibration draws), with E[FNR] >= alpha - 2B/(n+1) under continuity.
Reported per target alpha in {0.02, ..., 0.20}: realised test FNR (mean, 2.5-97.5 %
across splits, SD, min-max and every per-split value), share of test splits whose
realised FNR exceeds alpha, share of test candidates flagged as glass-formers, FPR,
precision; the uncorrected empirical threshold (largest t with R_hat(t) <= alpha) and
the naive 0.5 threshold as comparators.

One yardstick for all arms: fnr_excess_over_alpha = mean realised FNR - alpha, and
fnr_excess_in_split_se = that excess divided by SD/sqrt(n_splits). Because the splits
resample one finite dataset, this ratio is descriptive, not a valid test; it is reported
so the primary and the chemical-system arms are judged on the same scale rather than one
being called "compatible with the guarantee" and the other "a violation".

Legacy reproduction: the exact submitted split code (rng.permutation(len(y)), int cuts,
30 seeds) to reproduce the submitted numbers.

Output: results/R9_legacy_classif.json (merged into results/R9_legacy.json).
"""
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

from common import (DEMO, RES, _cut, _perm_grouped, check_disjoint, composition_groups, dump,
                    save_splits, summarize)

ALPHAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
SEEDS30 = list(range(30))
TAG = "R9_legacy_classif"


class GFA:
    """Minimal dataset container accepted by common.save_splits."""
    def __init__(self):
        d = np.load(os.path.join(DEMO, "glass_gfa_magpie.npz"))
        self.X, self.y = d["X"].astype(float), d["y"].astype(int)
        self.groups = composition_groups(self.X)
        self.name = "gfa"
        self.chemsys, self.alignment = self._chemsys()

    def _chemsys(self):
        from matminer.datasets import load_dataset
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition
        df = load_dataset("matbench_glass")
        lab_ok = bool(len(df) == len(self.y) and np.all(df["gfa"].astype(int).to_numpy() == self.y))
        ep = ElementProperty.from_preset("magpie")
        idx = np.random.default_rng(0).choice(len(df), 25, replace=False)
        feat_ok = bool(all(np.allclose(ep.featurize(Composition(df["composition"].iloc[i])), self.X[i]) for i in idx))
        assert lab_ok and feat_ok, "matbench_glass rows do not align with the cached Magpie matrix"
        cs = np.array(["-".join(sorted(e.symbol for e in Composition(c).elements)) for c in df["composition"]])
        _, inv = np.unique(cs, return_inverse=True)
        return inv, {"labels_identical": lab_ok, "features_match_25_random_rows": feat_ok,
                     "n_chemical_systems": int(inv.max() + 1)}

    def summary(self):
        return {"name": "matbench_glass (Magpie)", "N": int(len(self.y)), "n_features": int(self.X.shape[1]),
                "positive_rate": float(self.y.mean()), "n_unique_feature_rows": int(len(np.unique(self.groups))),
                **self.alignment}


def split_602020(ds, seed, by="composition"):
    """Grouped 60/20/20 split; by = 'composition' (identical Magpie vector) or 'chemsys'."""
    g = ds.groups if by == "composition" else ds.chemsys
    rng = np.random.default_rng([seed, 0 if by == "composition" else 1])
    order, gid = _perm_grouped(np.arange(len(ds.y)), g, rng)
    a = _cut(order, gid, 0.6); b = _cut(order, gid, 0.8)
    sp = {"train": order[:a], "cal": order[a:b], "test": order[b:]}
    check_disjoint(sp, g)
    return sp


def split_legacy(ds, seed):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(ds.y))
    ntr, ncal = int(0.6 * len(ds.y)), int(0.2 * len(ds.y))
    return {"train": idx[:ntr], "cal": idx[ntr:ntr + ncal], "test": idx[ntr + ncal:]}


def crc_threshold(p_pos_cal, alpha, B=1.0):
    """Largest t with (n * R_hat(t) + B) / (n + 1) <= alpha, R_hat(t) = mean(p < t)."""
    pp = np.sort(p_pos_cal); n = len(pp)
    kmax = int(np.floor(alpha * (n + 1) - B + 1e-12))       # max number of calibration misses allowed
    if kmax < 0:
        return 0.0
    if kmax >= n:
        return 1.0 + 1e-12
    return float(pp[kmax])                                    # exactly <= kmax points strictly below


def crc_threshold_loop(pp, alpha):
    """Submitted implementation (used to verify crc_threshold)."""
    n = len(pp); ts = np.unique(np.concatenate([[0.0], np.sort(pp), [1.0]])); best = 0.0
    for t in ts:
        if (n * np.mean(pp < t) + 1.0) / (n + 1.0) <= alpha:
            best = t
    return best


def emp_threshold(p_pos_cal, alpha):
    """Uncorrected plug-in: largest t with R_hat(t) <= alpha."""
    pp = np.sort(p_pos_cal); n = len(pp); kmax = int(np.floor(alpha * n + 1e-12))
    return float(pp[kmax]) if kmax < n else 1.0 + 1e-12


def decision_metrics(p, y, t):
    flag = p >= t; pos = y == 1
    tp = int((flag & pos).sum())
    return {"fnr": float(np.mean(~flag[pos])), "flagged": float(flag.mean()),
            "fpr": float(np.mean(flag[~pos])), "precision": float(tp / max(flag.sum(), 1))}


def one_split(ds, seed, split_fn, **kw):
    sp = split_fn(ds, seed, **kw)
    X, y = ds.X, ds.y
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed)
    clf.fit(X[sp["train"]], y[sp["train"]])
    p_cal = clf.predict_proba(X[sp["cal"]])[:, 1]; p_te = clf.predict_proba(X[sp["test"]])[:, 1]
    pc_pos = p_cal[y[sp["cal"]] == 1]
    out = {"seed": seed, "sizes": {k: int(len(v)) for k, v in sp.items()}, "n_pos_cal": int(len(pc_pos)),
           "test_accuracy_0.5": float(np.mean((p_te >= 0.5) == (y[sp["test"]] == 1))),
           "naive_0.5": decision_metrics(p_te, y[sp["test"]], 0.5), "crc": {}, "empirical": {}}
    for a in ALPHAS:
        t = crc_threshold(pc_pos, a)
        if seed == 0:
            assert abs(t - crc_threshold_loop(pc_pos, a)) < 1e-12, "vectorised CRC differs from submitted loop"
        out["crc"][a] = {"t": t, **decision_metrics(p_te, y[sp["test"]], t)}
        out["empirical"][a] = {"t": emp_threshold(pc_pos, a), **decision_metrics(p_te, y[sp["test"]], emp_threshold(pc_pos, a))}
    return out


def summarise_arm(rows):
    S = {"sizes_seed0": rows[0]["sizes"], "n_pos_cal": summarize([r["n_pos_cal"] for r in rows]),
         "test_accuracy_0.5": summarize([r["test_accuracy_0.5"] for r in rows]),
         "naive_0.5": {k: summarize([r["naive_0.5"][k] for r in rows]) for k in rows[0]["naive_0.5"]}}
    for arm in ("crc", "empirical"):
        S[arm] = {}
        for a in ALPHAS:
            v = {k: summarize([r[arm][a][k] for r in rows]) for k in rows[0][arm][a]}
            f = np.array([r[arm][a]["fnr"] for r in rows], float)
            v["share_splits_fnr_above_alpha"] = float(np.mean(f > a))
            v["crc_lower_bound_alpha_minus_2_over_n_plus_1"] = float(a - 2 / (np.mean([r["n_pos_cal"] for r in rows]) + 1))
            # one yardstick for every arm: excess of the mean realised FNR over the target,
            # in units of the split-to-split standard error (splits resample one dataset,
            # so this is a descriptive ratio, not a valid hypothesis test).
            se = f.std(ddof=1) / np.sqrt(len(f))
            v["fnr_excess_over_alpha"] = float(f.mean() - a)
            v["fnr_excess_in_split_se"] = float((f.mean() - a) / se) if se > 0 else None
            v["fnr_min"] = float(f.min()); v["fnr_max"] = float(f.max())
            S[arm][a] = v
    S["per_split_fnr_at_0.10"] = [r["crc"][0.10]["fnr"] for r in rows]
    S["per_split_fnr_crc"] = {str(a): [r["crc"][a]["fnr"] for r in rows] for a in ALPHAS}
    return S


def merge_parts():
    parts = {}
    for k in ("wcp", "local", "classif", "esol"):
        p = os.path.join(RES, f"R9_legacy_{k}.json")
        if os.path.exists(p):
            parts[k] = json.load(open(p))
    dump(parts, "R9_legacy.json")


def main(n_jobs=6):
    t0 = time.time()
    ds = GFA()
    out = {"dataset": ds.summary(), "alphas": ALPHAS, "seeds": SEEDS30,
           "model": "HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08), fixed a priori",
           "crc": "largest t with (n R_hat + 1)/(n + 1) <= alpha on calibration positives (B = 1)"}
    arms = {"grouped_composition": (split_602020, {"by": "composition"}),
            "grouped_chemical_system": (split_602020, {"by": "chemsys"}),
            "legacy_submitted_split": (split_legacy, {})}
    for name, (fn, kw) in arms.items():
        if fn is split_602020:
            save_splits(ds, split_fn=fn, seeds=SEEDS30, tag=f"gfa_{kw['by']}", **kw)
        rows = Parallel(n_jobs=n_jobs)(delayed(one_split)(ds, s, fn, **kw) for s in SEEDS30)
        S = summarise_arm(rows)
        out[name] = S
        print(f"[{name}] acc {S['test_accuracy_0.5']['mean']:.3f} naive FNR {S['naive_0.5']['fnr']['mean']:.4f} "
              f"flag {S['naive_0.5']['flagged']['mean']:.3f} | n_pos_cal {S['n_pos_cal']['mean']:.0f} | {time.time() - t0:.0f}s")
        for a in ALPHAS:
            c = S["crc"][a]
            print(f"   a={a:.2f} CRC FNR {c['fnr']['mean']:.4f} sd {c['fnr']['sd']:.4f} [{c['fnr']['lo']:.3f},{c['fnr']['hi']:.3f}] "
                  f"range [{c['fnr_min']:.3f},{c['fnr_max']:.3f}] excess {c['fnr_excess_over_alpha']:+.4f} "
                  f"({c['fnr_excess_in_split_se']:.2f} SE) P(>a) {c['share_splits_fnr_above_alpha']:.2f} "
                  f"flagged {c['flagged']['mean']:.3f} fpr {c['fpr']['mean']:.3f} "
                  f"| emp FNR {S['empirical'][a]['fnr']['mean']:.4f}", flush=True)
    out["runtime_min"] = (time.time() - t0) / 60
    dump(out, f"{TAG}.json")
    merge_parts()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_parts()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
