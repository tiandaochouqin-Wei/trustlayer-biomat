"""R9 (legacy Figure 4 / Figure 8a,b): ESOL/Delaney aqueous solubility (logS) calibration
and triage, re-run under the unified protocol.

Data: demo/delaney.csv (1,128 compounds). Features: the six numeric descriptors used in
the submission (minimum degree, molecular weight, H-bond donors, rings, rotatable bonds,
polar surface area). The column "ESOL predicted log solubility" is NOT used (it is the
output of a regression fitted to these same measurements and would leak the label).

Groups (primary, "identity"): connected components of records that share a canonical
RDKit SMILES (11 duplicate molecules) or an identical six-descriptor vector (119 sets,
328 records: DIFFERENT molecules that these six descriptors cannot tell apart, with a
within-group spread of the measured logS that is reported in the JSON). Sensitivity arms
that decompose this:
  "smiles"   - group only the 11 true duplicate molecules (the only genuine duplication
               in the file); this isolates how much of the change is duplicate leakage
               and how much is descriptor degeneracy;
  "none"     - plain random split, as submitted, at the same 60/20/20 sizes (this
               isolates the effect of the smaller training split from the effect of
               grouping when comparing with the submitted 70/30 triage panel);
  "scaffold" - Bemis-Murcko scaffold groups, a harder, shifted split; acyclic molecules
               share the empty scaffold; because two scaffolds hold 28 % and 23 % of the
               molecules, scaffold groups are assigned to partitions by a randomised
               size-balancing rule instead of contiguous cuts.
Note on wording: the identity grouping is NOT mainly a duplicate-leakage correction. A
reader who opens delaney.csv finds only 11 duplicate molecules. It is a degeneracy of
the six-descriptor representation: distinct molecules with identical descriptors are
indistinguishable to the model, so leaving them on both sides of the split measures
memorisation of a descriptor cell rather than generalisation.

Per split seed 0..19: grouped 60 / 20 / 20 train / calibration / test; RF
(make_model("RF")) on train only.
  * Calibration curve at levels {0.5, ..., 0.95} on test: split conformal (absolute
    residual) vs the RF tree-spread heuristic turned into Gaussian intervals; interval
    calibration error; full metric set at 90 % (coverage, width, IQR-normalised width,
    interval score, WIS, width-stratified and worst-cluster coverage) for split
    conformal, the heuristic and normalised split conformal (score |r| / tree SD).
  * Triage risk-coverage: RMSE of the most-confident fraction ranked by the normalised
    conformal half-width (identical ranking to the tree SD, since the calibrated scale
    is a common factor), vs a random ranking and an oracle ranking (by |error|);
    reduction at 50 % and relative AURC.
  * Abstention: share of test molecules whose 90 % normalised-conformal interval is
    wider than an acceptable full width W in {1.5, 2.0, 3.0} logS units, with the
    coverage and RMSE of the accepted molecules.

Legacy reproduction of the submitted numbers: calibration panel on a 55/20/25 random
split with 15 seeds; triage panel on a separate 70/30 random split (seeds 2000+s) with
no calibration set and ranking by raw tree SD.

Output: results/R9_legacy_esol.json (merged into results/R9_legacy.json).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm

from common import (ALPHA, DEMO, LEVELS, RES, SEEDS, Dataset, _cut, _perm_grouped, calibration_error,
                    check_disjoint, composition_groups, conformal_q, dump, interval_metrics, iqr,
                    make_model, rf_mean_std, risk_coverage, save_splits, summarize, weighted_interval_score)

FEATS = ["Minimum Degree", "Molecular Weight", "Number of H-Bond Donors", "Number of Rings",
         "Number of Rotatable Bonds", "Polar Surface Area"]
TARGET = "measured log solubility in mols per litre"
WIS_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
ABST_W = [1.5, 2.0, 3.0]
TAG = "R9_legacy_esol"


def load():
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    df = pd.read_csv(os.path.join(DEMO, "delaney.csv"))
    X = df[FEATS].to_numpy(float); y = df[TARGET].to_numpy(float)
    can = [Chem.MolToSmiles(Chem.MolFromSmiles(s.strip())) for s in df["smiles"]]
    _, sid = np.unique(can, return_inverse=True)
    fid = composition_groups(X)
    n = len(y); ns, nf = sid.max() + 1, fid.max() + 1
    rows = np.r_[np.arange(n), np.arange(n)]; cols = np.r_[n + sid, n + ns + fid]
    A = coo_matrix((np.ones(2 * n), (rows, cols)), shape=(n + ns + nf,) * 2)
    _, comp = connected_components(A, directed=False)
    _, ident = np.unique(comp[:n], return_inverse=True)
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in can]
    _, scid = np.unique(scaf, return_inverse=True)
    # how much of the identity grouping is true duplication and how much is descriptor
    # degeneracy (distinct molecules sharing all six descriptors)?
    can_a = np.asarray(can)
    multi_f = np.bincount(fid)[fid] > 1                      # record sits in a shared-descriptor cell
    deg = np.zeros(fid.max() + 1, bool)
    for u in np.unique(fid[multi_f]):
        deg[u] = len(np.unique(can_a[fid == u])) > 1         # cell holds >1 distinct molecule
    sd_in_cell = [float(np.std(y[fid == u], ddof=1)) for u in np.where(deg)[0]]
    meta = {"n_duplicate_canonical_smiles": int(n - ns), "n_identity_groups": int(ident.max() + 1),
            "n_records_in_multi_record_identity_groups": int(np.sum(np.bincount(ident)[ident] > 1)),
            "n_descriptor_degenerate_groups": int(deg.sum()),
            "n_records_in_descriptor_degenerate_groups": int(np.isin(fid, np.where(deg)[0]).sum()),
            "mean_within_group_logS_sd_degenerate": float(np.mean(sd_in_cell)),
            "max_within_group_logS_sd_degenerate": float(np.max(sd_in_cell)),
            "n_scaffold_groups": int(scid.max() + 1), "largest_scaffold_group": int(np.bincount(scid).max()),
            "source": "ESOL/Delaney (MoleculeNet delaney-processed.csv)"}
    ds = Dataset("esol_logS", X, y, FEATS, np.zeros(n), np.arange(n), np.array([], int), "logS", ident, meta)
    ds.scaffold = scid
    ds.smiles_id = sid
    return ds


def _balanced_group_split(g, rng, fr=(0.6, 0.2, 0.2)):
    """Random grouped split for very unequal group sizes (scaffolds: the acyclic and the
    benzene scaffold hold 28 % and 23 % of the molecules): groups in random order, each
    assigned to the partition with the smallest (filled + size) / target ratio."""
    ug = rng.permutation(np.unique(g)); target = np.array(fr) * len(g); filled = np.zeros(3)
    parts = [[], [], []]
    for u in ug:
        members = np.where(g == u)[0]
        j = int(np.argmin((filled + len(members)) / target))
        parts[j].append(members); filled[j] += len(members)
    return [np.concatenate(p) if p else np.array([], int) for p in parts]


def esol_split(ds, seed, by="identity"):
    """Grouped 60/20/20 (by = identity | smiles | scaffold | none)."""
    g = {"identity": ds.groups, "scaffold": ds.scaffold, "none": None, "smiles": ds.smiles_id}[by]
    rng = np.random.default_rng([seed, {"identity": 0, "scaffold": 1, "none": 2, "smiles": 3}[by]])
    if by == "scaffold":
        tr, ca, te = _balanced_group_split(g, rng)
        sp = {"train": tr, "cal": ca, "test": te}
    else:
        order, gid = _perm_grouped(np.arange(len(ds.y)), g, rng)
        a = _cut(order, gid, 0.6); b = _cut(order, gid, 0.8)
        sp = {"train": order[:a], "cal": order[a:b], "test": order[b:]}
    check_disjoint(sp, g)
    return sp


def fit_rf(X, y, seed):
    m = make_model("RF", seed); m.set_params(n_jobs=1)
    return m.fit(X, y)


def one_split(ds, seed, by):
    sp = esol_split(ds, seed, by)
    X, y = ds.X, ds.y
    m = fit_rf(X[sp["train"]], y[sp["train"]], seed)
    mc, sc = rf_mean_std(m, X[sp["cal"]]); mt, st = rf_mean_std(m, X[sp["test"]])
    yc, yt = y[sp["cal"]], y[sp["test"]]
    r_cal = np.abs(yc - mc); err = np.abs(yt - mt); iq = iqr(y[sp["train"]])
    out = {"seed": seed, "sizes": {k: int(len(v)) for k, v in sp.items()}}
    cc, ch, cn = [], [], []
    sd_c = sc + 1e-6; sd_t = st + 1e-6
    for l in LEVELS:
        cc.append(float(np.mean(err <= conformal_q(r_cal, 1 - l))))
        ch.append(float(np.mean(err <= norm.ppf(0.5 + l / 2) * st)))
        cn.append(float(np.mean(err <= conformal_q(r_cal / sd_c, 1 - l) * sd_t)))
    out["curve"] = {"conformal": cc, "heuristic": ch, "normalised_conformal": cn}
    out["ce"] = {"conformal": calibration_error(LEVELS, cc), "heuristic": calibration_error(LEVELS, ch),
                 "normalised_conformal": calibration_error(LEVELS, cn)}
    q = conformal_q(r_cal, ALPHA); z = norm.ppf(1 - ALPHA / 2); qn = conformal_q(r_cal / sd_c, ALPHA)
    Xt = X[sp["test"]]
    arms = {"conformal": (mt - q, mt + q), "heuristic": (mt - z * st, mt + z * st),
            "normalised_conformal": (mt - qn * sd_t, mt + qn * sd_t)}
    ints = {"conformal": {a: (mt - conformal_q(r_cal, a), mt + conformal_q(r_cal, a)) for a in WIS_ALPHAS},
            "heuristic": {a: (mt - norm.ppf(1 - a / 2) * st, mt + norm.ppf(1 - a / 2) * st) for a in WIS_ALPHAS},
            "normalised_conformal": {a: (mt - conformal_q(r_cal / sd_c, a) * sd_t, mt + conformal_q(r_cal / sd_c, a) * sd_t)
                                     for a in WIS_ALPHAS}}
    out["at90"] = {}
    for k, (lo, hi) in arms.items():
        mm = interval_metrics(lo, hi, yt, iq, X=Xt)
        mm["wis"] = weighted_interval_score(yt, mt, ints[k])
        out["at90"][k] = mm
    rng = np.random.default_rng([seed, 99])
    out["triage"] = {"conformal_ranked": risk_coverage(err, qn * sd_t),
                     "random_ranked": risk_coverage(err, rng.permutation(len(err)).astype(float)),
                     "oracle_ranked": risk_coverage(err, err)}
    wn = 2 * qn * sd_t; covn = err <= qn * sd_t
    out["abstention"] = {}
    for W in ABST_W:
        acc = wn <= W
        out["abstention"][W] = {"abstain_rate": float(1 - acc.mean()),
                                "coverage_accepted": float(covn[acc].mean()) if acc.any() else None,
                                "rmse_accepted": float(np.sqrt(np.mean(err[acc] ** 2))) if acc.any() else None}
    out["rmse_test"] = float(np.sqrt(np.mean(err ** 2)))
    return out


def legacy_panel_a(ds, seed):
    N = len(ds.y); idx = np.random.default_rng(seed).permutation(N)
    n_tr, n_cal = int(0.55 * N), int(0.20 * N)
    tr, cal, te = idx[:n_tr], idx[n_tr:n_tr + n_cal], idx[n_tr + n_cal:]
    m = fit_rf(ds.X[tr], ds.y[tr], seed)
    mt, st = rf_mean_std(m, ds.X[te]); mc, _ = rf_mean_std(m, ds.X[cal])
    rc = np.abs(ds.y[cal] - mc); n = len(rc)
    q = np.quantile(rc, min(np.ceil((n + 1) * 0.9) / n, 1.0), method="higher"); z = norm.ppf(0.95)
    e = np.abs(ds.y[te] - mt)
    return {"cov90_conformal": float(np.mean(e <= q)), "cov90_heuristic": float(np.mean(e <= z * st)),
            "width90_conformal": float(2 * q), "width90_heuristic": float(np.mean(2 * z * st)),
            "n_test_sharing_identity_group_with_train": int(np.isin(ds.groups[te], ds.groups[tr]).sum()),
            "n_test_sharing_smiles_group_with_train": int(np.isin(ds.smiles_id[te], ds.smiles_id[tr]).sum()),
            "n_train": int(len(tr)), "n_test": int(len(te))}


def legacy_panel_b(ds, seed):
    N = len(ds.y); idx = np.random.default_rng(2000 + seed).permutation(N)
    tr, te = idx[:int(0.7 * N)], idx[int(0.7 * N):]
    m = fit_rf(ds.X[tr], ds.y[tr], seed)
    mt, st = rf_mean_std(m, ds.X[te])
    out = risk_coverage(np.abs(ds.y[te] - mt), st)
    out["n_train"] = int(len(tr)); out["n_test"] = int(len(te))
    out["n_test_sharing_identity_group_with_train"] = int(np.isin(ds.groups[te], ds.groups[tr]).sum())
    out["n_test_sharing_smiles_group_with_train"] = int(np.isin(ds.smiles_id[te], ds.smiles_id[tr]).sum())
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


def merge_parts():
    parts = {}
    for k in ("wcp", "local", "classif", "esol"):
        p = os.path.join(RES, f"R9_legacy_{k}.json")
        if os.path.exists(p):
            parts[k] = json.load(open(p))
    dump(parts, "R9_legacy.json")


def main(n_jobs=6):
    t0 = time.time()
    ds = load()
    out = {"dataset": {**ds.summary(), **ds.meta}, "features": FEATS, "levels": LEVELS.tolist(),
           "note": "splits resample one dataset; percentile intervals describe split-to-split variability"}
    for by in ("identity", "smiles", "none", "scaffold"):
        save_splits(ds, split_fn=esol_split, tag=f"esol_{by}", by=by)
        rows = Parallel(n_jobs=n_jobs)(delayed(one_split)(ds, s, by) for s in SEEDS)
        S = {"sizes_seed0": rows[0]["sizes"], "sizes_test": summarize([r["sizes"]["test"] for r in rows])}
        S["curve_mean"] = {k: np.mean([r["curve"][k] for r in rows], 0).tolist() for k in rows[0]["curve"]}
        S["curve_at90"] = {k: summarize([r["curve"][k][list(LEVELS).index(0.90)] for r in rows]) for k in rows[0]["curve"]}
        S["ce_mean"] = {k: summarize([r["ce"][k][0] for r in rows]) for k in rows[0]["ce"]}
        S["at90"] = {a: {k: agg(rows, ["at90", a, k]) for k in rows[0]["at90"][a]} for a in rows[0]["at90"]}
        S["triage"] = {a: {"reduction_at50_pct": agg(rows, ["triage", a, "reduction_at50_pct"]),
                           "aurc_rel": agg(rows, ["triage", a, "aurc_rel"]),
                           "rmse_at50": agg(rows, ["triage", a, "rmse_at50"]),
                           "rmse_all": agg(rows, ["triage", a, "rmse_all"]),
                           "curve_mean": np.mean([r["triage"][a]["rmse"] for r in rows], 0).tolist(),
                           "grid": rows[0]["triage"][a]["grid"]} for a in rows[0]["triage"]}
        S["abstention"] = {W: {k: agg(rows, ["abstention", W, k]) for k in rows[0]["abstention"][W]} for W in ABST_W}
        S["rmse_test"] = summarize([r["rmse_test"] for r in rows])
        out[f"split_{by}"] = S
        print(f"[{by}] n_test {S['sizes_test']['mean']:.0f} | cov90 conf {S['curve_at90']['conformal']['mean']:.3f} "
              f"[{S['curve_at90']['conformal']['lo']:.3f},{S['curve_at90']['conformal']['hi']:.3f}] heur {S['curve_at90']['heuristic']['mean']:.3f} "
              f"[{S['curve_at90']['heuristic']['lo']:.3f},{S['curve_at90']['heuristic']['hi']:.3f}] | CE conf {S['ce_mean']['conformal']['mean']:.3f} "
              f"heur {S['ce_mean']['heuristic']['mean']:.3f} | width conf {S['at90']['conformal']['width']['mean']:.2f} heur {S['at90']['heuristic']['width']['mean']:.2f} "
              f"| triage {S['triage']['conformal_ranked']['reduction_at50_pct']['mean']:.1f}% [{S['triage']['conformal_ranked']['reduction_at50_pct']['lo']:.1f},"
              f"{S['triage']['conformal_ranked']['reduction_at50_pct']['hi']:.1f}] rmse50 {S['triage']['conformal_ranked']['rmse_at50']['mean']:.3f} "
              f"all {S['rmse_test']['mean']:.3f} | {time.time() - t0:.0f}s", flush=True)
    la = Parallel(n_jobs=n_jobs)(delayed(legacy_panel_a)(ds, s) for s in range(15))
    lb = Parallel(n_jobs=n_jobs)(delayed(legacy_panel_b)(ds, s) for s in range(20))
    rm = np.array([r["rmse"] for r in lb]).mean(0); full = float(np.mean([r["rmse_all"] for r in lb]))
    i50 = int(np.argmin(np.abs(np.array(lb[0]["grid"]) - 0.5)))
    out["legacy"] = {"panel_a_55_20_25_15seeds": {k: summarize([r[k] for r in la]) for k in la[0]},
                     "panel_b_70_30_seeds2000": {
                         "rmse_at50_of_mean_curve": float(rm[i50]), "rmse_all_mean": full,
                         "reduction_at50_pct_submitted_formula": float(100 * (1 - rm[i50] / full)),
                         "reduction_at50_pct_per_split": summarize([r["reduction_at50_pct"] for r in lb]),
                         "n_train": lb[0]["n_train"], "n_test": lb[0]["n_test"],
                         "n_test_sharing_identity_group_with_train":
                             summarize([r["n_test_sharing_identity_group_with_train"] for r in lb]),
                         "n_test_sharing_smiles_group_with_train":
                             summarize([r["n_test_sharing_smiles_group_with_train"] for r in lb])}}
    out["attribution_of_the_triage_change"] = {
        "note": "decomposes the submitted 70/30 ungrouped triage number into (a) training-split size and "
                "(b) grouping; and the grouping step into true duplicates vs descriptor degeneracy",
        "legacy_70_30_ungrouped": {"reduction_at50_pct": float(100 * (1 - rm[i50] / full)), "rmse_all": full,
                                   "n_train": lb[0]["n_train"]},
        "protocol_60_20_20_ungrouped": {"reduction_at50_pct": out["split_none"]["triage"]["conformal_ranked"]["reduction_at50_pct"]["mean"],
                                        "rmse_all": out["split_none"]["rmse_test"]["mean"],
                                        "n_train": out["split_none"]["sizes_seed0"]["train"]},
        "protocol_60_20_20_grouped_by_smiles_only": {"reduction_at50_pct": out["split_smiles"]["triage"]["conformal_ranked"]["reduction_at50_pct"]["mean"],
                                                     "rmse_all": out["split_smiles"]["rmse_test"]["mean"]},
        "protocol_60_20_20_grouped_by_identity": {"reduction_at50_pct": out["split_identity"]["triage"]["conformal_ranked"]["reduction_at50_pct"]["mean"],
                                                  "rmse_all": out["split_identity"]["rmse_test"]["mean"]},
        "protocol_60_20_20_scaffold": {"reduction_at50_pct": out["split_scaffold"]["triage"]["conformal_ranked"]["reduction_at50_pct"]["mean"],
                                       "rmse_all": out["split_scaffold"]["rmse_test"]["mean"]}}
    print("legacy", json.dumps({k: {kk: (vv["mean"] if isinstance(vv, dict) and "mean" in vv else vv) for kk, vv in v.items()}
                                for k, v in out["legacy"].items()}))
    out["runtime_min"] = (time.time() - t0) / 60
    dump(out, f"{TAG}.json")
    merge_parts()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_parts()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
