"""R9 (legacy Figure 12): local (conditional) coverage of source-calibrated split
conformal across bioactive-glass design space, re-run under the unified protocol.

Data and shift: common.glass_dataset("Tg") (SciGlass, Si/Ca/Na > 0; source P <= q0.70(P),
which is P = 0; target P >= q0.75(P | P > 0)). The submitted analysis used a different
target threshold (q0.80) and a 70/30 source split; both are replaced by the protocol.

Per split seed 0..19 (design_split):
  * RF (make_model("RF")) on source-train; split-conformal radius q from source-cal
    (alpha = 0.10).
  * Evaluation records = every glass record NOT used for training or calibration:
    source-test (stest), target (recal pool + target-test; nothing is recalibrated
    here, so both target halves are untouched evaluation records) and intermediate
    records (0 < P < target threshold; excluded from source and target elsewhere).
    Source-calibration records are NOT evaluated (the submitted analysis evaluated them,
    which inflates coverage because q is their own empirical quantile).
  * Coverage indicator c_i = 1{|y_i - yhat_i| <= q}.

Embedding: centred log-ratio (CLR) of the 25 retained element fractions after replacing
zeros by a pseudo-count eps = 1e-4 (atomic fraction) and re-closing each row to sum 1;
PCA (2 components) fitted once on all 7,623 records (7,025 unique compositions; inputs
only, label-free). Sensitivity: eps in {1e-5, 1e-4, 1e-3}.

Local coverage: Nadaraya-Watson (Gaussian kernel) smoother of c_i over the PC plane,
evaluated (i) on a 90 x 90 grid spanning the 1-99th percentiles of each PC (cells with
kernel mass < 3 are left empty) and (ii) at every target / source-test record location.
At a record location the smoother is reported both in-sample (the record's own indicator
enters with the Gaussian self-weight 1) and leave-one-out (self-weight removed), because
the two differ by up to 0.04 at the smallest bandwidth and the kNN estimator below
already excludes the point itself; the LOO form is the comparable one.
Bandwidth sensitivity: h = f x h_Scott, f in {0.25, 0.5, 1, 2}, h_Scott = n^(-1/6) x mean
PC standard deviation of the evaluation records, plus the submitted absolute h = 0.55.
Projection-free check: k-nearest-neighbour local coverage in the full CLR space
(k in {25, 50, 100}) at the target records, and worst k-means cluster coverage
(common.cluster_conditional_coverage, k = 8, CLR space).

Grid statistics are reported over the SUPPORTED cells only (kernel mass >= 3); the share
of supported cells is reported alongside, and the share over all cells (unsupported cells
counted as "not below nominal") is kept separately as grid_frac_all_cells_below_0.9.

Legacy reproduction of the submitted numbers (bioglass_tg.npz, q0.80 target, one 70/30
split, evaluation including source-calibration rows, h = 0.55).

Output: results/R9_legacy_local.json (merged into results/R9_legacy.json).
"""
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

from common import (ALPHA, DEMO, RES, SEEDS, cluster_conditional_coverage, conformal_q, design_split,
                    dump, glass_dataset, make_model, summarize)

EPS_MAIN = 1e-4
EPS_GRID = [1e-5, 1e-4, 1e-3]
H_FACTORS = [0.25, 0.5, 1.0, 2.0]
H_ABS_LEGACY = 0.55
KNN = [25, 50, 100]
NG = 90
MIN_MASS = 3.0
TAG = "R9_legacy_local"


def clr(X, eps):
    c = np.asarray(X, float).copy()
    c[c <= 0] = eps
    c = c / c.sum(1, keepdims=True)
    lc = np.log(c)
    return lc - lc.mean(1, keepdims=True)


def pca2(C):
    mu = C.mean(0)
    U, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    return (C - mu) @ Vt[:2].T, (S ** 2 / (S ** 2).sum())[:2]


def nw_at(P, Q, c, h):
    """Nadaraya-Watson estimate at query points Q from (P, c); returns (estimate, mass)."""
    est = np.empty(len(Q)); mass = np.empty(len(Q))
    for i0 in range(0, len(Q), 512):
        q = Q[i0:i0 + 512]
        w = np.exp(-((q[:, None, :] - P[None, :, :]) ** 2).sum(-1) / (2 * h * h))
        m = w.sum(1); mass[i0:i0 + 512] = m
        est[i0:i0 + 512] = (w * c[None, :]).sum(1) / np.maximum(m, 1e-300)
    return est, mass


def nw_self(P, c, h):
    """NW smoother evaluated at the sample points; returns (in-sample, leave-one-out).
    The Gaussian kernel gives every point self-weight exactly 1, so the LOO estimate is
    (sum_j w_ij c_j - c_i) / (sum_j w_ij - 1); it is NaN for isolated points."""
    n = len(P); est = np.empty(n); loo = np.empty(n)
    for i0 in range(0, n, 512):
        q = P[i0:i0 + 512]
        w = np.exp(-((q[:, None, :] - P[None, :, :]) ** 2).sum(-1) / (2 * h * h))
        m = w.sum(1); s = (w * c[None, :]).sum(1)
        est[i0:i0 + 512] = s / np.maximum(m, 1e-300)
        mm = m - 1.0
        loo[i0:i0 + 512] = np.where(mm > 1e-9, (s - c[i0:i0 + 512]) / np.maximum(mm, 1e-300), np.nan)
    return est, loo


def _mean_fin(a):
    a = np.asarray(a, float); f = np.isfinite(a)
    return float(a[f].mean()) if f.any() else None


def grid_field(P, c, h, gx, gy):
    G = np.array([(a, b) for b in gy for a in gx])      # row-major over gy
    est, mass = nw_at(P, G, c, h)
    est[mass < MIN_MASS] = np.nan
    return est.reshape(len(gy), len(gx)), mass.reshape(len(gy), len(gx))


def fit_rf(X, y, seed):
    m = make_model("RF", seed); m.set_params(n_jobs=1)
    return m.fit(X, y)


def one_split(ds, seed, emb, gx, gy):
    sp = design_split(ds, seed)
    X, y = ds.X, ds.y
    m = fit_rf(X[sp["train"]], y[sp["train"]], seed)
    q = conformal_q(np.abs(y[sp["cal"]] - m.predict(X[sp["cal"]])), ALPHA)
    roles = {"source_test": sp["stest"], "target": np.r_[sp["rpool"], sp["ttest"]], "intermediate": ds.excluded}
    ev = np.concatenate(list(roles.values()))
    role_of = np.concatenate([[k] * len(v) for k, v in roles.items()])
    cov = (np.abs(y[ev] - m.predict(X[ev])) <= q).astype(float)
    out = {"seed": seed, "q": float(q), "n_eval": int(len(ev)),
           "n_eval_by_role": {k: int(len(v)) for k, v in roles.items()},
           "marginal": float(cov.mean()),
           "by_role": {k: float(cov[role_of == k].mean()) for k in roles}}
    tmask, smask = role_of == "target", role_of == "source_test"
    out["eps"] = {}
    fields = {}
    for eps, (PC, C) in emb.items():
        P = PC[ev]
        hS = len(ev) ** (-1 / 6) * P.std(0).mean()
        e = {"h_scott": float(hS), "nw": {}, "knn_target": {}}
        hs = {f"scott_x{f:g}": f * hS for f in H_FACTORS}
        if eps == EPS_MAIN:
            hs["abs_0.55"] = H_ABS_LEGACY
        for hn, h in hs.items():
            est, loo = nw_self(P, cov, h)
            lt = loo[tmask]; ls = loo[smask]
            r = {"h": float(h), "local_at_target": float(est[tmask].mean()),
                 "local_at_source_test": float(est[smask].mean()),
                 "local_at_target_loo": _mean_fin(lt), "local_at_source_test_loo": _mean_fin(ls),
                 "n_loo_undefined_target": int(np.sum(~np.isfinite(lt))),
                 "frac_target_local_below_0.9": float(np.mean(est[tmask] < 1 - ALPHA)),
                 "frac_target_local_loo_below_0.9": _mean_fin(np.where(np.isfinite(lt),
                                                                       (lt < 1 - ALPHA).astype(float), np.nan))}
            if eps == EPS_MAIN:
                F, _ = grid_field(P, cov, h, gx, gy)
                sup = np.isfinite(F)
                r["grid_min"] = float(np.nanmin(F))
                r["grid_frac_supported"] = float(sup.mean())
                r["grid_frac_supported_below_0.9"] = float(np.mean(F[sup] < 1 - ALPHA)) if sup.any() else None
                r["grid_frac_all_cells_below_0.9"] = float(np.mean(np.where(sup, F, np.inf) < 1 - ALPHA))
                if hn in ("scott_x1", "abs_0.55"):
                    fields[hn] = F
            e["nw"][hn] = r
        Cev = C[ev]
        nn = NearestNeighbors(n_neighbors=max(KNN) + 1).fit(Cev)
        _, idx = nn.kneighbors(Cev[tmask])
        for k in KNN:
            nb = idx[:, 1:k + 1]                                  # exclude the point itself
            e["knn_target"][k] = float(cov[nb].mean())
        if eps == EPS_MAIN:
            e["worst_cluster_cov_clr"] = cluster_conditional_coverage(Cev, cov.astype(bool), seed=seed)[0]
        out["eps"][str(eps)] = e
    return out, fields


def legacy():
    """Original demo/exp_rev_local_coverage.py numbers (and the effect of each change)."""
    d = np.load(os.path.join(DEMO, "bioglass_tg.npz"), allow_pickle=True)
    X, y, P = d["X"].astype(float), d["y"], d["P"]
    src_mask = P <= np.quantile(P, 0.70); tgt_mask = P >= np.quantile(P[P > 0], 0.80)
    PC, evr = pca2(clr(X, 1e-4))
    rng = np.random.default_rng(0); si = rng.permutation(np.where(src_mask)[0])
    ntr = int(0.7 * len(si)); tr, cal = si[:ntr], si[ntr:]
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=4, random_state=0).fit(X[tr], y[tr])
    rc = np.abs(y[cal] - m.predict(X[cal]))
    q = np.quantile(rc, min(np.ceil((len(cal) + 1) * (1 - ALPHA)) / len(cal), 1.0), method="higher")
    cov = (np.abs(y - m.predict(X)) <= q).astype(float)
    ev = np.ones(len(y), bool); ev[tr] = False
    ev_idx = np.where(ev)[0]
    rr = np.random.default_rng(1)
    if len(ev_idx) > 3500:
        ev_idx = rr.choice(ev_idx, 3500, replace=False)
    te = tgt_mask & ev
    est, _ = nw_at(PC[ev_idx], PC[te][:400], cov[ev_idx], 0.55)
    ev_nocal = ev.copy(); ev_nocal[cal] = False
    idx2 = np.where(ev_nocal)[0]
    est2, _ = nw_at(PC[idx2], PC[te], cov[idx2], 0.55)
    # composition of the legacy evaluation set: the 70/30 source split leaves NO held-out
    # source rows, so deleting the calibration rows deletes every source record.
    interm = (~src_mask) & (~tgt_mask)
    mix = {"source_cal": int(np.sum(ev & src_mask)), "source_other": int(np.sum(ev & src_mask & ~np.isin(np.arange(len(y)), cal))),
           "target_q080": int(np.sum(ev & tgt_mask)), "intermediate": int(np.sum(ev & interm))}
    return {"n_eval": int(ev.sum()), "n_cal_rows_in_eval": int(len(cal)), "n_target_q080": int(tgt_mask.sum()),
            "eval_mixture": mix, "eval_mixture_excl_cal": {**mix, "source_cal": 0},
            "coverage_by_role_incl_cal": {"source_cal": float(cov[cal].mean()),
                                          "target": float(cov[ev & tgt_mask].mean()),
                                          "intermediate": float(cov[ev & interm].mean())},
            "evr": evr.tolist(), "marginal_incl_cal_rows": float(cov[ev].mean()),
            "local_at_target_h0.55": float(est.mean()),
            "raw_target_coverage": float(cov[te].mean()),
            "marginal_excl_cal_rows": float(cov[ev_nocal].mean()),
            "local_at_target_h0.55_excl_cal_rows": float(est2.mean()),
            "source_cal_rows_coverage": float(cov[cal].mean())}


def merge_parts():
    parts = {}
    for k in ("wcp", "local", "classif", "esol"):
        p = os.path.join(RES, f"R9_legacy_{k}.json")
        if os.path.exists(p):
            parts[k] = json.load(open(p))
    dump(parts, "R9_legacy.json")


def main(n_jobs=6):
    t0 = time.time()
    ds = glass_dataset("Tg", "glass_Tg", "K")
    emb = {}
    evr_all = {}
    for eps in EPS_GRID:
        C = clr(ds.X, eps); PC, evr = pca2(C)
        emb[eps] = (PC, C); evr_all[str(eps)] = evr.tolist()
    PCm = emb[EPS_MAIN][0]
    lo1, hi1 = np.percentile(PCm[:, 0], [1, 99]); lo2, hi2 = np.percentile(PCm[:, 1], [1, 99])
    gx, gy = np.linspace(lo1, hi1, NG), np.linspace(lo2, hi2, NG)
    res = Parallel(n_jobs=n_jobs)(delayed(one_split)(ds, s, emb, gx, gy) for s in SEEDS)
    rows = [r for r, _ in res]
    out = {"dataset": ds.summary(), "protocol": {
        "pseudo_count_main": EPS_MAIN, "pseudo_count_grid": EPS_GRID,
        "pca_fit": "all 7,623 records (7,025 unique compositions), inputs only, no labels",
        "at_point_estimator": "Nadaraya-Watson reported in-sample and leave-one-out "
                              "(local_at_*_loo); the kNN estimator always excludes the point itself",
        "grid_statistics": "grid_frac_supported_below_0.9 is over supported cells only "
                           "(kernel mass >= 3; grid_frac_supported gives their share); "
                           "grid_frac_all_cells_below_0.9 is over all cells",
        "evaluated_records": "source-test + all target records (recal pool and target-test halves) + intermediate "
                             "(0 < P < target threshold); source-train and source-cal excluded",
        "kernel": "Gaussian Nadaraya-Watson", "h_factors_of_scott": H_FACTORS, "h_abs_legacy": H_ABS_LEGACY,
        "grid": f"{NG}x{NG}, 1-99th percentiles of PC1/PC2, cells with kernel mass < {MIN_MASS} empty",
        "knn_k": KNN}, "pca_explained_variance": evr_all, "n_eval_by_role_seed0": rows[0]["n_eval_by_role"]}
    out["marginal"] = summarize([r["marginal"] for r in rows])
    out["by_role"] = {k: summarize([r["by_role"][k] for r in rows]) for k in rows[0]["by_role"]}
    out["eps"] = {}
    for e in rows[0]["eps"]:
        E = {"h_scott": summarize([r["eps"][e]["h_scott"] for r in rows]), "nw": {}, "knn_target": {}}
        for hn in rows[0]["eps"][e]["nw"]:
            E["nw"][hn] = {k: summarize([r["eps"][e]["nw"][hn][k] for r in rows]) for k in rows[0]["eps"][e]["nw"][hn]}
        for k in KNN:
            E["knn_target"][k] = summarize([r["eps"][e]["knn_target"][k] for r in rows])
        if "worst_cluster_cov_clr" in rows[0]["eps"][e]:
            E["worst_cluster_cov_clr"] = summarize([r["eps"][e]["worst_cluster_cov_clr"] for r in rows])
        out["eps"][e] = E
    # seed-averaged local-coverage fields for plotting (cells supported in >= half of the splits)
    out["field"] = {"gx": gx.round(4).tolist(), "gy": gy.round(4).tolist(), "layout": "field[iy][ix]"}
    for hn in ("scott_x1", "abs_0.55"):
        Fs = np.stack([f[hn] for _, f in res])
        ok = np.isfinite(Fs).mean(0) >= 0.5
        M = np.where(ok, np.nanmean(np.where(np.isfinite(Fs), Fs, np.nan), 0), np.nan)
        out["field"][hn] = np.where(np.isfinite(M), np.round(M, 4), None).tolist()
    role = np.full(len(ds.y), "intermediate", dtype=object); role[ds.src] = "source"; role[ds.tgt] = "target"
    out["points"] = {"pc1": np.round(PCm[:, 0], 4).tolist(), "pc2": np.round(PCm[:, 1], 4).tolist(),
                     "role": role.tolist()}
    out["per_split"] = [{"seed": r["seed"], "marginal": r["marginal"], **r["by_role"],
                         "local_target_scott": r["eps"][str(EPS_MAIN)]["nw"]["scott_x1"]["local_at_target"],
                         "local_target_scott_loo": r["eps"][str(EPS_MAIN)]["nw"]["scott_x1"]["local_at_target_loo"],
                         "local_target_h055": r["eps"][str(EPS_MAIN)]["nw"]["abs_0.55"]["local_at_target"],
                         "local_target_h055_loo": r["eps"][str(EPS_MAIN)]["nw"]["abs_0.55"]["local_at_target_loo"]}
                        for r in rows]
    out["legacy"] = legacy()
    out["runtime_min"] = (time.time() - t0) / 60
    em = out["eps"][str(EPS_MAIN)]["nw"]
    print(f"marginal {out['marginal']['mean']:.3f} [{out['marginal']['lo']:.3f},{out['marginal']['hi']:.3f}] | by role "
          + " ".join(f"{k} {v['mean']:.3f}" for k, v in out["by_role"].items()))
    for hn, v in em.items():
        print(f"  h {hn:10s} ({v['h']['mean']:.2f}): local@target {v['local_at_target']['mean']:.3f} "
              f"[{v['local_at_target']['lo']:.3f},{v['local_at_target']['hi']:.3f}] (LOO {v['local_at_target_loo']['mean']:.3f}) "
              f"local@stest {v['local_at_source_test']['mean']:.3f} (LOO {v['local_at_source_test_loo']['mean']:.3f})"
              + (f" | grid supported {v['grid_frac_supported']['mean']:.2f}, of which below 0.9: "
                 f"{v['grid_frac_supported_below_0.9']['mean']:.3f} (all cells {v['grid_frac_all_cells_below_0.9']['mean']:.3f})"
                 if "grid_frac_supported" in v else ""))
    for e, E in out["eps"].items():
        print(f"  eps {e}: knn " + " ".join(f"k{k} {E['knn_target'][k]['mean']:.3f}" for k in KNN)
              + f" | scott NW@target {E['nw']['scott_x1']['local_at_target']['mean']:.3f}")
    print("legacy", out["legacy"], f"{(time.time() - t0) / 60:.1f} min")
    dump(out, f"{TAG}.json")
    merge_parts()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_parts()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
