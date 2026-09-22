"""R8 budget: the cost of target-domain recalibration and an honest retrospective replay.

Answers Reviewer 2 comment 7 (target data are the scarcest, most expensive data),
Reviewer 2 comment 9 and Reviewer 1 comment 3 (the retrospective acquisition replay
of the submitted Figure 11b and its "65% and 90%" measurement reduction), under the
unified protocol (common.py, PROTOCOL.md). Reference model = RF, as in R1_core.

Per design-shift dataset and per split s = 0..19 (common.design_split):
 (a) k-sweep. Recalibrate on the first k records of the recalibration pool and
     evaluate on the target-test half (alpha = 0.10 and 0.20). Distribution over
     splits (5/50/95th percentiles; mean and 2.5-97.5 % interval) of realised
     coverage and normalised width, against exchangeable theory: the Beta law of
     the conditional coverage (infinite test set, common.beta_coverage_quantiles)
     and the exact Beta-Binomial law for the actual target-test size.
     Supplement: B_SUB random k-subsets of the recalibration pool per split
     (still evaluated on the same target-test half) for smoother tail estimates.
 (b) Smallest k with 5th-percentile coverage >= 0.80 and mean >= 0.88. This is a
     retrospective, descriptive quantity (it reads target-test labels) and is never
     used to select anything. Its protocol-compliant counterpart selects k inside
     the recalibration pool only (B_SEL grouped inner splits of the pool: the first k
     records of a grouped permutation calibrate, the remaining whole compositions are
     held out; R_SEL inner-RNG replicates for the Monte Carlo spread) and then
     evaluates the chosen k once on target-test. The reliable reference for small
     target regions is the exact Beta-Binomial law at every integer k.
 (c) The submitted manuscript's Chebyshev budget rule m = a(1-a)/(d e^2) - 1 versus
     the exact Beta rule (smallest n with P_Beta(coverage < 1-a-e) <= d), plus the
     finite-test Beta-Binomial version and the empirical failure rates.
 (d) alpha sensitivity (0.10 vs 0.20): minimum k for a finite interval and the
     coverage / width trade-off.
 (e) Retrospective replay. The target pool is the candidate set; candidates are
     "measured" one at a time in (i) random order (= the protocol's seeded grouped
     permutation, so the first k measured are the first k of the recalibration
     pool) and (ii) decreasing model-agnostic novelty sigma(x) (mean distance to the
     10 nearest standardised training inputs). After each measurement we
     recalibrate on all measured target records and evaluate coverage and
     normalised width on the not-yet-measured remainder (>= MIN_REM records).
     Stopping points defined from remainder labels are retrospective oracles;
     label-free, pre-registered stopping rules are reported alongside.
Outputs results/R8_budget.json and results/R8_budget.md.
Usage: python -W ignore exp_R8_budget.py [n_jobs]      (full run)
       python -W ignore exp_R8_budget.py --md-only      (rewrite the md from the json)
"""
import json
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import beta as beta_d, betabinom, binom, spearmanr

from common import (ALPHA, LEVELS, SEEDS, RES, design_shift_datasets, design_split, headline_m,
                    conformal_q, make_model, interval_score, weighted_interval_score,
                    calibration_error, summarize, iqr, dump, beta_coverage_quantiles)

TAG = "R8_budget"
KGRID = [0, 4, 5, 9, 10, 15, 20, 30, 40, 60, 80, 120, 200, 300]
ALPHAS = [0.10, 0.20]
EPS_GRID = [0.05, 0.08, 0.10]
DELTA_GRID = [0.05, 0.10]
B_SUB = 200            # random k-subsets of the recalibration pool per split ((a)/(c) supplement only)
B_SEL = 2000           # grouped inner splits of the recalibration pool for the protocol-compliant (b) selection
R_SEL = 5              # independent inner-RNG replicates of that selection (Monte Carlo spread)
MIN_REM = 10           # replay: smallest not-yet-measured remainder on which coverage is evaluated
STOP_MIN_REM, STOP_MIN_REM_FRAC = 20, 0.20   # default stopping horizon: remainder >= max(20, 20% of |T|)
                                             # (below that one miss moves coverage by > 0.05). This is an
                                             # analyst convention; HZ_GRID reports the sensitivity to it.
HZ_GRID = [10, 20, 30, 50]                   # replay horizon sensitivity: minimum remainder sizes
STAY_MIN_WINDOW = 10   # a 'stays from n' stop must be verified over >= 10 further measurements up to the horizon
                       # (otherwise a stop at the very end of the horizon holds trivially over 1-2 points)
NOV_K = 10             # novelty: mean distance to the 10 nearest standardised training inputs
RF_JOBS = 2            # tree-model threads per worker (shared machine)
WIS_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
CRIT_P5, CRIT_MEAN = 0.80, 0.88          # (b) criterion
C1_COV, C1_FRAC = 0.90, 0.80             # (e) criterion from the task
MANUSCRIPT_EPS, MANUSCRIPT_DELTA = 0.08, 0.10
TOL = 1e-9             # tolerance for coverage comparisons (coverages are ratios of integers)


def akey(a):
    return f"alpha_{a:.2f}"


# ================================================================= theory helpers
def j_rank(n, alpha=ALPHA):
    """1-based order statistic used by common.conformal_q; > n means an infinite interval."""
    return int(np.ceil((n + 1) * (1 - alpha)))


def _fr(x):
    """Exact rational from the decimal literal (avoids float ceil errors, e.g. 249.00000000000003 -> 250)."""
    return Fraction(str(x))


def chebyshev_m(eps, delta, alpha=ALPHA):
    """Submitted manuscript / demo/exp6: m = alpha(1-alpha)/(delta eps^2) - 1, rounded up (exact arithmetic)."""
    a, d, e = _fr(alpha), _fr(delta), _fr(eps)
    return max(1, math.ceil(a * (1 - a) / (d * e * e) - 1))


def cantelli_m(eps, delta, alpha=ALPHA):
    """One-sided Chebyshev (Cantelli) with the same variance bound: a(1-a)(1-d)/(d e^2) - 1 (exact arithmetic)."""
    a, d, e = _fr(alpha), _fr(delta), _fr(eps)
    return max(1, math.ceil(a * (1 - a) * (1 - d) / (d * e * e) - 1))


def beta_fail_vec(ns, eps, alpha=ALPHA):
    """P_Beta(conditional coverage < 1 - alpha - eps) for each calibration size in ns;
    1.0 (counted as failing: uninformative) where the interval is infinite."""
    ns = np.asarray(ns, int)
    j = np.ceil((ns + 1) * (1 - alpha)).astype(int)
    out = np.ones(len(ns))
    ok = j <= ns
    out[ok] = beta_d.cdf(1 - alpha - eps, j[ok], ns[ok] + 1 - j[ok])
    return out


def beta_fail(n, eps, alpha=ALPHA):
    return float(beta_fail_vec([n], eps, alpha)[0])


def _first_stable(ps, delta):
    """ps[i] belongs to n = i+1. 'first' = smallest n with ps <= delta; 'stable' = smallest n
    from which ps <= delta for every larger n in range (saw-tooth because j is discrete)."""
    ok = ps <= delta
    first = int(np.argmax(ok) + 1) if ok.any() else None
    bad = np.where(~ok)[0]
    stable = int(bad.max() + 2) if len(bad) else 1
    return first, (stable if stable <= len(ps) else None)


def exact_beta_rule(eps, delta, alpha=ALPHA, n_max=5000):
    """Exact Beta rule: smallest n with P_Beta(coverage < 1-alpha-eps) <= delta."""
    return _first_stable(beta_fail_vec(np.arange(1, n_max + 1), eps, alpha), delta)


def runs(ks):
    """Compress a sorted list of integers into [start, end] runs."""
    out = []
    for k in ks:
        if out and k == out[-1][1] + 1:
            out[-1][1] = int(k)
        else:
            out.append([int(k), int(k)])
    return out


def runs_str(rr, max_runs=6):
    if not rr:
        return "none"
    s = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in rr[:max_runs])
    return s + (", ..." if len(rr) > max_runs else "")


def bb_rule(eps, delta, n_test, alpha=ALPHA, k_max=3000, k_pool=None, k_list_max=200):
    """Finite-test version: exact Beta-Binomial P(realised coverage on n_test exchangeable test
    points < 1-alpha-eps) as a function of the calibration size k.

    As k -> infinity this probability tends to the Binomial(n_test, 1-alpha) tail ('limit'). The limit
    is NOT a floor: at conservative saw-tooth sizes (the conformal order statistic is the sample maximum
    or close to it) the probability dips below it. Where limit > delta no k meets delta *stably*, but
    delta can still be met at such dips. Returns first / stable k, the limit, the minimum over k and its
    argmin, the runs of k <= k_list_max meeting delta, and the same restricted to k <= k_pool."""
    thr = int(np.ceil((1 - alpha - eps) * n_test - 1e-9)) - 1     # X <= thr  <=>  X/n < 1-alpha-eps
    limit = float(binom.cdf(thr, n_test, 1 - alpha)) if thr >= 0 else 0.0
    ks = np.arange(1, k_max + 1)
    j = np.ceil((ks + 1) * (1 - alpha)).astype(int)
    ps = np.ones(k_max)
    ok = j <= ks
    ps[ok] = betabinom.cdf(thr, n_test, j[ok], ks[ok] + 1 - j[ok]) if thr >= 0 else 0.0
    first, stable = _first_stable(ps, delta)
    imin = int(np.argmin(ps[:k_list_max]))     # over k <= k_list_max: the saw-tooth dips (beyond, ps -> limit)
    meet = (ps <= delta + 1e-12)
    out = {"first": first, "stable": stable, "limit_P_fail_k_to_inf": limit,
           "stably_reachable": stable is not None, f"min_P_fail_k_upto_{k_list_max}": float(ps[imin]),
           f"argmin_k_upto_{k_list_max}": imin + 1,
           f"k_runs_meeting_delta_upto_{k_list_max}": runs(ks[:k_list_max][meet[:k_list_max]].tolist())}
    if k_pool is not None and k_pool >= 1:
        kp = int(min(k_pool, k_max))
        ip = int(np.argmin(ps[:kp]))
        out.update({"k_pool": kp, "met_within_pool": bool(meet[:kp].any()),
                    "k_runs_meeting_delta_within_pool": runs(ks[:kp][meet[:kp]].tolist()),
                    "min_P_fail_within_pool": float(ps[ip]), "argmin_k_within_pool": ip + 1})
    return out


def bb_mixture(pairs, alpha, qs=(0.05, 0.5, 0.95), eps_list=EPS_GRID):
    """Exact exchangeable law of realised test coverage, mixed over splits.
    pairs = [(k, n_test)] per split. Returns quantiles, P(cov >= 1-alpha), P(cov < 1-alpha-eps)."""
    vals, probs = [], []
    for k, n in pairs:
        j = j_rank(k, alpha)
        if k < 1 or j > k:
            return None
        x = np.arange(n + 1)
        vals.append(x / n); probs.append(betabinom.pmf(x, n, j, k + 1 - j) / len(pairs))
    v = np.concatenate(vals); p = np.concatenate(probs)
    o = np.argsort(v, kind="stable"); v, p = v[o], p[o]; c = np.cumsum(p)
    quant = [float(v[min(len(v) - 1, np.searchsorted(c, q - 1e-12))]) for q in qs]
    return {"q5_50_95": quant,
            "P_ge_nominal": float(p[v >= 1 - alpha - TOL].sum()),
            "P_fail": {f"{e:.2f}": float(p[v < 1 - alpha - e - TOL].sum()) for e in eps_list}}


def theory_tables():
    out = {}
    for a in ALPHAS:
        rows = []
        for e in EPS_GRID:
            for d in DELTA_GRID:
                mc = chebyshev_m(e, d, a)
                first, stable = exact_beta_rule(e, d, a)
                rows.append({"eps": e, "delta": d, "chebyshev_m": mc, "cantelli_m": cantelli_m(e, d, a),
                             "exact_beta_first": first, "exact_beta_stable": stable,
                             "chebyshev_over_exact": (mc / stable) if stable else None,
                             "P_beta_fail_at_chebyshev_m": beta_fail(mc, e, a),
                             "P_beta_fail_at_exact_first": beta_fail(first, e, a) if first else None})
        out[akey(a)] = rows
    # sanity: Beta variance bound used by the manuscript, Var <= a(1-a)/(n+1)
    ratios = []
    for n in range(9, 2001):
        j = j_rank(n); aa, bb = j, n + 1 - j
        var = aa * bb / ((aa + bb) ** 2 * (aa + bb + 1))
        ratios.append(var * (n + 1) / (ALPHA * (1 - ALPHA)))
    out["variance_bound_check"] = {"max_Var_over_bound_n9_2000": float(max(ratios)),
                                   "holds": bool(max(ratios) <= 1 + 1e-12)}
    # consistency of our order-statistic convention with common.beta_coverage_quantiles
    mism = [n for n in range(9, 1001) if abs(beta_coverage_quantiles(n)[1] -
                                             float(beta_d.ppf(0.5, j_rank(n), n + 1 - j_rank(n)))) > 1e-9]
    out["common_beta_convention_mismatch_n"] = mism
    # the original demo scripts' quantile (demo/exp4, demo/exp6)
    out["original_cq_check"] = original_cq_check()
    return out


def original_cq_check():
    """demo/exp4_campaign_replay.py and demo/exp6_recalibration_budget.py use
    np.quantile(r, min(ceil((n+1)(1-a))/n, 1), method='higher'). Compare the order
    statistic it returns with the finite-sample conformal one (common.conformal_q)."""
    rows = []
    for n in [3, 5, 8, 9, 10, 13, 15, 20, 22, 23, 26, 30, 40, 52, 56, 116, 140]:
        r = np.arange(1.0, n + 1)
        q_orig = np.quantile(r, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0), method="higher")
        q_prot = conformal_q(r, ALPHA)
        rows.append({"n": n, "orig_rank": int(q_orig), "conformal_rank": (int(q_prot) if np.isfinite(q_prot) else "inf"),
                     "orig_expected_cov": float(int(q_orig) / (n + 1)),
                     "conformal_expected_cov": (float(int(q_prot) / (n + 1)) if np.isfinite(q_prot) else 1.0)})
    return rows


RULE_N = {"headline_m": None,
          "exact_beta_eps0.08_delta0.10": exact_beta_rule(0.08, 0.10)[1],
          "exact_beta_eps0.05_delta0.10": exact_beta_rule(0.05, 0.10)[1],
          "chebyshev_eps0.08_delta0.10": chebyshev_m(0.08, 0.10)}


# ================================================================= per-split work
def pool_select(r_pool, g_pool, ks, rng, B=B_SEL, alpha=ALPHA):
    """Protocol-compliant (b): choose k from recalibration-pool labels only.

    Each of B inner draws is a grouped permutation of the pool (replicate records of one composition
    contiguous, as in common.design_split). cal = its first k records, which is the deployed first-k
    rule. holdout = the remaining pool records whose composition does not occur in cal (whole groups
    only, so no composition sits on both sides; at least MIN_REM records). Returns the first grid k whose
    inner-holdout coverage has 5th percentile >= CRIT_P5 and mean >= CRIT_MEAN, and the trace
    {key: [p5, mean, n_valid_draws]}. Passing g_pool = arange(n) gives record-level inner splits."""
    nP = len(r_pool)
    _, gidx = np.unique(g_pool, return_inverse=True)
    ng = int(gidx.max()) + 1
    order = np.argsort(rng.random((B, ng))[:, gidx], axis=1, kind="stable")
    R = r_pool[order]
    G = gidx[order]
    pos = np.arange(nP)[None, :]
    trace, sel = {}, None
    for key, k in ks:
        j = j_rank(k, alpha)
        if j > k or k > nP - MIN_REM:          # infinite interval, or too few records left to hold out
            continue
        q = np.sort(R[:, :k], axis=1)[:, j - 1]
        hold = (pos >= k) & (G != G[:, k - 1:k])  # drop the group cut by the first-k boundary
        nh = hold.sum(1)
        okd = nh >= MIN_REM
        if okd.mean() < 0.95:
            continue
        cov = ((R <= q[:, None]) & hold).sum(1)[okd] / nh[okd]
        p5, mn = float(np.percentile(cov, 5)), float(cov.mean())
        trace[key] = [p5, mn, int(okd.sum())]
        if sel is None and p5 >= CRIT_P5 - TOL and mn >= CRIT_MEAN - TOL:
            sel = key
    return sel, trace


def cluster_labels(Xe, n_clusters=8, seed=0, min_size=10):
    """Same k-means construction as common.cluster_conditional_coverage, computed once per split."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(Xe)
    k = int(min(n_clusters, max(1, len(Xe) // min_size)))
    return KMeans(k, n_init=5, random_state=seed).fit_predict(Z)


def worst_cluster(lab, covered, min_size=10):
    per = [covered[lab == c].mean() for c in np.unique(lab) if (lab == c).sum() >= min_size]
    return float(min(per)) if per else None


def sweep_keys(ds, nP):
    m = headline_m(ds)
    ks = sorted(set([k for k in KGRID if k <= nP] + [m] + [v for v in RULE_N.values() if v and v <= nP]))
    keys = [(str(k), k) for k in ks] + [("full_pool", nP)]
    return keys


def one_split(ds, seed):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    sp = design_split(ds, seed)
    X, y = ds.X, ds.y
    model = make_model("RF", seed)
    model.set_params(n_jobs=RF_JOBS)
    model.fit(X[sp["train"]], y[sp["train"]])
    T = np.concatenate([sp["rpool"], sp["ttest"]])     # the seeded, grouped permutation of the target pool
    nP, nT = len(sp["rpool"]), len(T)
    muT = model.predict(X[T]); rT = np.abs(y[T] - muT)
    r_cal = np.abs(y[sp["cal"]] - model.predict(X[sp["cal"]]))
    r_pool, r_test = rT[:nP], rT[nP:]
    s_test = np.sort(r_test)
    yt, mut = y[sp["ttest"]], muT[nP:]
    iqr_T = iqr(y[ds.tgt])
    lab = cluster_labels(X[sp["ttest"]]) if len(yt) >= 40 else None
    out = {"seed": seed, "sizes": {k: int(len(v)) for k, v in sp.items()}, "n_test": int(len(yt)),
           "n_pool": int(nP), "n_T": int(nT)}
    keys = sweep_keys(ds, nP)
    rng = np.random.default_rng(10_000 + seed)

    # ---------------- (a)/(d) k-sweep on target-test, alpha 0.10 and 0.20 ----------------
    out["sweep"], out["sub"] = {}, {}
    for a in ALPHAS:
        q_src = conformal_q(r_cal, a)
        sw, sb = {}, {}
        for key, k in keys:
            cal_r = r_cal if k == 0 else r_pool[:k]
            q = conformal_q(cal_r, a)
            covered = r_test <= q
            rec = {"k": int(k), "coverage": float(covered.mean()), "finite": bool(np.isfinite(q)),
                   "n_distinct_compositions": int(len(np.unique(ds.groups[sp["rpool"][:k]]))) if k > 0 else 0,
                   "norm_width": float(2 * q / iqr_T) if np.isfinite(q) else None,
                   "width": float(2 * q) if np.isfinite(q) else None,
                   "wider_than_source_cal": bool(q > q_src)}
            if np.isfinite(q):
                rec["interval_score"] = interval_score(mut - q, mut + q, yt, a)
                if lab is not None:
                    rec["worst_cluster_cov"] = worst_cluster(lab, covered)
                if abs(a - ALPHA) < 1e-12:
                    qa = {aa: conformal_q(cal_r, aa) for aa in WIS_ALPHAS}
                    if all(np.isfinite(v) for v in qa.values()):
                        rec["wis"] = weighted_interval_score(yt, mut, {aa: (mut - v, mut + v) for aa, v in qa.items()})
                    lv = [l for l in LEVELS if np.isfinite(conformal_q(cal_r, 1 - l))]
                    cv = [float(np.mean(r_test <= conformal_q(cal_r, 1 - l))) for l in lv]
                    rec["cal_error_mean"], rec["cal_error_max"] = calibration_error(lv, cv)
                    rec["cal_error_n_levels"] = len(lv)
            sw[key] = rec
            if k > 0:      # supplement: random k-subsets of the pool, same target-test half
                qs = np.array([conformal_q(r_pool[rng.choice(nP, k, replace=False)], a) for _ in range(B_SUB)])
                sb[key] = {"cov": (np.searchsorted(s_test, qs, side="right") / len(s_test)),
                           "nw": np.where(np.isfinite(qs), 2 * qs / iqr_T, np.inf)}
        out["sweep"][akey(a)], out["sub"][akey(a)] = sw, sb

    # ---------------- (b) protocol-compliant budget selection inside the pool ----------------
    # grouped inner splits (PROTOCOL.md: one composition never on both sides of any split), B_SEL draws,
    # R_SEL independent inner-RNG replicates for the Monte Carlo spread; replicate 0 is the primary result.
    sw = out["sweep"][akey(ALPHA)]
    ks_sel = [(key, k) for key, k in keys if key != "full_pool"]
    gP = ds.groups[sp["rpool"]]
    reps = [pool_select(r_pool, gP, ks_sel, np.random.default_rng([20_000, seed, rep])) for rep in range(R_SEL)]
    sel, trace = reps[0]
    # sensitivity only: record-level inner splits (the previous, non-compliant version of this selection)
    sel_rec, _ = pool_select(r_pool, np.arange(nP), ks_sel, np.random.default_rng([30_000, seed]))
    out["pool_select"] = {"key": sel, "k": (sw[sel]["k"] if sel else None), "trace": trace,
                          "ttest_cov": (sw[sel]["coverage"] if sel else None),
                          "ttest_norm_width": (sw[sel]["norm_width"] if sel else None),
                          "rep_k": [(sw[s]["k"] if s else None) for s, _ in reps],
                          "rep_ttest_cov": [(sw[s]["coverage"] if s else None) for s, _ in reps],
                          "record_level_k": (sw[sel_rec]["k"] if sel_rec else None),
                          "record_level_ttest_cov": (sw[sel_rec]["coverage"] if sel_rec else None),
                          "n_pool_compositions": int(len(np.unique(gP))),
                          "fallback_full_pool_cov": sw["full_pool"]["coverage"],
                          "fallback_full_pool_norm_width": sw["full_pool"]["norm_width"]}

    # ---------------- (e) retrospective replay over the whole target pool ----------------
    sc = StandardScaler().fit(X[sp["train"]])
    nn = NearestNeighbors(n_neighbors=NOV_K).fit(sc.transform(X[sp["train"]]))
    sig = nn.kneighbors(sc.transform(X[T]))[0].mean(1)
    n_max = nT - MIN_REM
    q_src = conformal_q(r_cal, ALPHA)
    out["replay"] = {}
    gT = ds.groups[T]
    for name, o in [("random", np.arange(nT)), ("novelty", np.argsort(-sig, kind="stable"))]:
        r_o = rT[o]
        _, gi = np.unique(gT[o], return_inverse=True)
        g_tot = np.bincount(gi); g_meas = np.zeros_like(g_tot); s_cnt = 0
        cov = np.empty(n_max + 1); nw = np.empty(n_max + 1); mres = np.empty(n_max + 1)
        strad = np.zeros(n_max + 1, int)   # remainder records whose composition was already measured
        for n in range(n_max + 1):
            q = q_src if n == 0 else conformal_q(r_o[:n], ALPHA)
            cov[n] = np.mean(r_o[n:] <= q)
            nw[n] = 2 * q / iqr_T
            mres[n] = np.mean(r_o[n:]) / iqr_T
            strad[n] = s_cnt
            g = gi[n]                       # measure record n before the next step
            s_cnt += (g_tot[g] - 1) if g_meas[g] == 0 else -1
            g_meas[g] += 1
        out["replay"][name] = {"cov": cov, "nw": nw,
                               "rem_mean_abs_resid_norm": mres, "straddle": strad}
    out["novelty_spearman_resid"] = float(spearmanr(sig, rT).statistic)
    out["q_src_norm_width"] = float(2 * q_src / iqr_T)
    return out


# ================================================================= aggregation
def pctl(a, q):
    a = np.asarray(a, float)
    return float(np.percentile(a, q)) if len(a) else None


def agg_list(rows, a, key, field):
    return [r["sweep"][a][key].get(field) for r in rows]


def smallest_k(entries, ok):
    """entries: ordered list of (k, key); ok: dict key->bool. First k that satisfies, and
    the smallest k from which every larger k in the list satisfies ('stable')."""
    first = next((k for k, key in entries if ok[key]), None)
    stable = None
    for i in range(len(entries)):
        if all(ok[key] for _, key in entries[i:]):
            stable = entries[i][0]
            break
    return first, stable


def aggregate(ds, rows):
    nT = len(ds.tgt)
    m = headline_m(ds)
    nPs = [r["n_pool"] for r in rows]
    D = {"dataset": ds.summary(), "headline_m": m, "n_T": nT,
         "n_pool": {"min": int(min(nPs)), "median": float(np.median(nPs)), "max": int(max(nPs))},
         "n_test": {"min": int(min(r["n_test"] for r in rows)), "median": float(np.median([r["n_test"] for r in rows])),
                    "max": int(max(r["n_test"] for r in rows))},
         "sizes_seed0": rows[0]["sizes"]}
    common_keys = [key for key in rows[0]["sweep"][akey(ALPHA)]
                   if all(key in r["sweep"][akey(ALPHA)] for r in rows)]
    D["sweep_keys"] = common_keys

    # ---------------- (a) + (d) ----------------
    D["ksweep"] = {}
    for a in ALPHAS:
        A = akey(a); S = {}
        for key in common_keys:
            ks = [r["sweep"][A][key]["k"] for r in rows]
            covs = np.array(agg_list(rows, A, key, "coverage"))
            fin = np.array(agg_list(rows, A, key, "finite"))
            nws = [v for v in agg_list(rows, A, key, "norm_width") if v is not None]
            e = {"k": int(np.median(ks)), "k_range": [int(min(ks)), int(max(ks))],
                 "n_infinite_interval": int((~fin).sum()),
                 "coverage": summarize(covs), "p5": pctl(covs, 5), "p50": pctl(covs, 50), "p95": pctl(covs, 95),
                 "min": float(covs.min()), "frac_ge_nominal": float(np.mean(covs >= 1 - a - TOL)),
                 "frac_ge_0.80": float(np.mean(covs >= 0.80 - TOL)),
                 "norm_width": summarize(nws), "norm_width_median": pctl(nws, 50) if nws else None,
                 "frac_wider_than_source_cal": float(np.mean(agg_list(rows, A, key, "wider_than_source_cal"))),
                 "n_distinct_compositions_mean": float(np.mean(agg_list(rows, A, key, "n_distinct_compositions")))}
            for f in ["interval_score", "worst_cluster_cov", "wis", "cal_error_mean", "cal_error_max"]:
                vals = agg_list(rows, A, key, f)
                if any(v is not None for v in vals):
                    e[f] = summarize(vals)
            if key != "0":
                kk = int(np.median(ks))
                e["theory_beta_5_50_95"] = beta_coverage_quantiles(kk, alpha=a)
                e["theory_betabinom"] = bb_mixture([(r["sweep"][A][key]["k"], r["n_test"]) for r in rows], a)
                sc = np.concatenate([r["sub"][A][key]["cov"] for r in rows]).astype(float)
                sw = np.concatenate([r["sub"][A][key]["nw"] for r in rows]).astype(float)
                e["resampled"] = {"n_draws": int(len(sc)), "mean": float(sc.mean()), "p5": pctl(sc, 5),
                                  "p50": pctl(sc, 50), "p95": pctl(sc, 95),
                                  "frac_ge_nominal": float(np.mean(sc >= 1 - a - TOL)),
                                  "P_fail": {f"{ee:.2f}": float(np.mean(sc < 1 - a - ee - TOL)) for ee in EPS_GRID},
                                  "norm_width_median": float(np.median(sw))}
            S[key] = e
        D["ksweep"][A] = S

    # ---------------- (b) smallest adequate k ----------------
    S = D["ksweep"][akey(ALPHA)]
    ent = sorted([(S[key]["k"], key) for key in common_keys if key not in ("0",) and S[key]["k"] >= 1],
                 key=lambda t: (t[0], t[1] == "full_pool"))
    fin_ok = {key: S[key]["n_infinite_interval"] == 0 for _, key in ent}
    ok_prot = {key: fin_ok[key] and S[key]["p5"] >= CRIT_P5 - TOL and S[key]["coverage"]["mean"] >= CRIT_MEAN - TOL for _, key in ent}
    ok_res = {key: fin_ok[key] and S[key]["resampled"]["p5"] >= CRIT_P5 - TOL and S[key]["resampled"]["mean"] >= CRIT_MEAN - TOL
              for _, key in ent}
    ok_bb = {key: fin_ok[key] and S[key]["theory_betabinom"] is not None and
             S[key]["theory_betabinom"]["q5_50_95"][0] >= CRIT_P5 - TOL for _, key in ent}
    kp = smallest_k(ent, ok_prot); kr = smallest_k(ent, ok_res); kb = smallest_k(ent, ok_bb)
    # Beta (infinite test set) over all integers: mean >= 0.9 always, so only the 5th percentile binds
    nn_ = np.arange(1, 3001); jj = np.ceil((nn_ + 1) * (1 - ALPHA)).astype(int); fin_ = jj <= nn_
    p5b = np.full(len(nn_), -1.0); p5b[fin_] = beta_d.ppf(0.05, jj[fin_], nn_[fin_] + 1 - jj[fin_])
    b_first, b_stable = _first_stable(np.where(p5b >= CRIT_P5 - TOL, 0.0, 1.0), 0.5)
    # Beta-Binomial for the actual (median) target-test size, over ALL integer k (not only the grid):
    # the reliable reference for small target regions, where the 20-split p5 is essentially the 2nd-lowest split
    ntest_med = int(np.median([r["n_test"] for r in rows]))
    nPmin = int(min(nPs))
    p5bb = np.full(len(nn_), -1.0)
    p5bb[fin_] = betabinom.ppf(0.05, ntest_med, jj[fin_], nn_[fin_] + 1 - jj[fin_]) / ntest_med
    okbb = np.where(p5bb >= CRIT_P5 - TOL, 0.0, 1.0)
    bb_first, bb_stable = _first_stable(okbb, 0.5)
    bb_first_p, bb_stable_p = _first_stable(okbb[:nPmin], 0.5)
    ps = [r["pool_select"] for r in rows]
    chosen = [p["k"] for p in ps if p["k"] is not None]
    rep_cert = [sum(p["rep_k"][i] is not None for p in ps) for i in range(R_SEL)]
    rec_k = [p["record_level_k"] for p in ps]
    near = {key: S[key]["p5"] for _, key in ent if fin_ok[key] and CRIT_P5 - 0.01 <= S[key]["p5"] < CRIT_P5 - TOL}
    D["min_k"] = {
        "criterion": f"5th percentile of realised target-test coverage >= {CRIT_P5} and mean >= {CRIT_MEAN}",
        "retrospective_oracle_note": "uses target-test labels: descriptive only, never used to select anything",
        "protocol_first_k": {"first": kp[0], "stable": kp[1],
                             "first_frac_of_T": (kp[0] / nT) if kp[0] else None,
                             "stable_frac_of_T": (kp[1] / nT) if kp[1] else None,
                             "first_frac_of_pool": (kp[0] / np.median(nPs)) if kp[0] else None},
        "oracle_near_misses_p5_within_0.01_below": near,
        "resampled_subsets": {"first": kr[0], "stable": kr[1],
                              "first_frac_of_T": (kr[0] / nT) if kr[0] else None},
        "theory_betabinom_actual_test_size": {"first": kb[0], "stable": kb[1], "note": "grid k only, mixed over splits"},
        "theory_betabinom_all_integers": {"n_test": ntest_med, "first": bb_first, "stable": bb_stable,
                                          "first_within_pool": bb_first_p, "stable_within_pool": bb_stable_p,
                                          "pool_min": nPmin},
        "theory_beta_infinite_test": {"first": b_first, "stable": b_stable},
        "pool_internal_selection": {
            "note": (f"k chosen with recalibration-pool labels only: {B_SEL} grouped inner splits of the pool "
                     "(cal = first k records of a grouped permutation, holdout = remaining whole compositions, "
                     f">= {MIN_REM} records), then evaluated once on target-test. Primary = inner-RNG replicate 0; "
                     f"{R_SEL} independent replicates give the Monte Carlo spread."),
            "n_splits_certified": len(chosen), "n_splits_not_certifiable": len(ps) - len(chosen),
            "chosen_k_summary": summarize(chosen) if chosen else None,
            "chosen_k_values": [p["k"] for p in ps],
            "ttest_cov_when_certified": summarize([p["ttest_cov"] for p in ps if p["k"] is not None]),
            "ttest_cov_p5_when_certified": pctl([p["ttest_cov"] for p in ps if p["k"] is not None], 5) if chosen else None,
            "ttest_cov_min_when_certified": (float(min(p["ttest_cov"] for p in ps if p["k"] is not None)) if chosen else None),
            "ttest_frac_ge_0.80_when_certified": (float(np.mean([p["ttest_cov"] >= CRIT_P5 - TOL for p in ps if p["k"] is not None]))
                                                  if chosen else None),
            "ttest_norm_width_when_certified": summarize([p["ttest_norm_width"] for p in ps if p["k"] is not None]),
            "fallback_full_pool_cov_when_not_certified": summarize([p["fallback_full_pool_cov"] for p in ps if p["k"] is None]),
            "mc_replicates": {"B": B_SEL, "n_replicates": R_SEL, "n_certified_per_replicate": rep_cert,
                              "n_certified_min": int(min(rep_cert)), "n_certified_max": int(max(rep_cert)),
                              "chosen_k_per_split_per_replicate": [p["rep_k"] for p in ps],
                              "n_splits_certification_unstable": int(sum(len({v is None for v in p["rep_k"]}) > 1 for p in ps)),
                              "n_splits_chosen_k_unstable": int(sum(len(set(p["rep_k"])) > 1 for p in ps))},
            "record_level_inner_sensitivity": {
                "note": "previous (non-compliant) record-level inner splits, one replicate; sensitivity only",
                "n_splits_certified": int(sum(v is not None for v in rec_k)), "chosen_k_values": rec_k,
                "ttest_cov_when_certified": summarize([p["record_level_ttest_cov"] for p in ps if p["record_level_k"] is not None]),
                "ttest_cov_p5_when_certified": (pctl([p["record_level_ttest_cov"] for p in ps if p["record_level_k"] is not None], 5)
                                                if any(v is not None for v in rec_k) else None)},
            "n_pool_compositions_mean": float(np.mean([p["n_pool_compositions"] for p in ps])),
        }}

    # ---------------- (c) per-dataset empirical failure rates vs theory ----------------
    cross = {}
    for e in EPS_GRID:
        for d in DELTA_GRID:
            emp = next((S[key]["k"] for k, key in ent if fin_ok[key] and S[key]["resampled"]["P_fail"][f"{e:.2f}"] <= d + TOL), None)
            emp_stable = smallest_k(ent, {key: fin_ok[key] and S[key]["resampled"]["P_fail"][f"{e:.2f}"] <= d + TOL
                                          for _, key in ent})[1]
            bb = bb_rule(e, d, ntest_med, k_pool=nPmin)
            cross[f"eps{e:.2f}_delta{d:.2f}"] = {
                "empirical_first_k_resampled": emp, "empirical_stable_k_resampled": emp_stable,
                "betabinom_rule_first": bb["first"], "betabinom_rule_stable": bb["stable"],
                "betabinom_limit_P_fail_k_to_inf": bb["limit_P_fail_k_to_inf"],
                "betabinom_stably_reachable": bb["stably_reachable"],
                "betabinom_min_P_fail_k_upto_200": bb["min_P_fail_k_upto_200"],
                "betabinom_argmin_k_upto_200": bb["argmin_k_upto_200"],
                "betabinom_k_runs_meeting_delta_upto_200": bb["k_runs_meeting_delta_upto_200"],
                "betabinom_met_within_pool": bb["met_within_pool"],
                "betabinom_k_runs_meeting_delta_within_pool": bb["k_runs_meeting_delta_within_pool"],
                "betabinom_min_P_fail_within_pool": bb["min_P_fail_within_pool"],
                "betabinom_argmin_k_within_pool": bb["argmin_k_within_pool"],
                "exact_beta_rule_stable": exact_beta_rule(e, d)[1], "chebyshev_m": chebyshev_m(e, d),
                "chebyshev_feasible_in_pool": chebyshev_m(e, d) <= min(nPs),
                "chebyshev_feasible_in_T_replay": chebyshev_m(e, d) <= nT - MIN_REM}
    D["budget_rules"] = {"n_test_median": ntest_med, "by_eps_delta": cross,
                         "failure_curve_resampled_eps0.08": {S[key]["k"]: S[key]["resampled"]["P_fail"]["0.08"]
                                                             for k, key in ent if key != "full_pool"},
                         "failure_curve_betabinom_eps0.08": {S[key]["k"]: (S[key]["theory_betabinom"] or {}).get("P_fail", {}).get("0.08")
                                                             for k, key in ent if key != "full_pool"}}

    # ---------------- (d) alpha sensitivity ----------------
    A1, A2 = akey(0.10), akey(0.20)
    trade = {}
    for key in common_keys:
        n1 = agg_list(rows, A1, key, "norm_width"); n2 = agg_list(rows, A2, key, "norm_width")
        ratio = [b / a_ for a_, b in zip(n1, n2) if a_ is not None and b is not None and a_ > 0]
        trade[key] = {"k": D["ksweep"][A1][key]["k"],
                      "alpha_0.10": {"cov_mean": D["ksweep"][A1][key]["coverage"]["mean"], "cov_p5": D["ksweep"][A1][key]["p5"],
                                     "n_inf": D["ksweep"][A1][key]["n_infinite_interval"],
                                     "nw_median": D["ksweep"][A1][key]["norm_width_median"]},
                      "alpha_0.20": {"cov_mean": D["ksweep"][A2][key]["coverage"]["mean"], "cov_p5": D["ksweep"][A2][key]["p5"],
                                     "n_inf": D["ksweep"][A2][key]["n_infinite_interval"],
                                     "nw_median": D["ksweep"][A2][key]["norm_width_median"]},
                      "width_ratio_0.20_over_0.10": summarize(ratio) if ratio else None}
    min_fin = {}
    for a in ALPHAS:
        n_fin = next(n for n in range(1, 100) if np.isfinite(conformal_q(np.arange(n, dtype=float), a)))
        min_fin[akey(a)] = {"formula_ceil(1/alpha)-1": int(np.ceil(1 / a - 1e-12)) - 1, "checked_with_conformal_q": n_fin}
    D["alpha_sensitivity"] = {"min_k_finite_interval": min_fin, "tradeoff": trade}

    # ---------------- (e) replay ----------------
    n_max = nT - MIN_REM
    h_rem = max(STOP_MIN_REM, int(np.ceil(STOP_MIN_REM_FRAC * nT)))
    horizon = nT - h_rem
    R = {"candidate_set": "whole target pool (rpool + target-test)", "n_T": nT, "n_max_measured": n_max,
         "min_remainder": MIN_REM, "stopping_horizon_n": horizon, "stopping_horizon_min_remainder": h_rem, "novelty_spearman_with_abs_residual": summarize([r["novelty_spearman_resid"] for r in rows]),
         "source_cal_norm_width": summarize([r["q_src_norm_width"] for r in rows])}
    # exchangeable theory for random order: P(remainder coverage >= 0.9) with n measured
    th = np.full(n_max + 1, np.nan)
    for n in range(9, n_max + 1):
        j = j_rank(n); rem = nT - n
        thr = int(np.ceil(C1_COV * rem - 1e-9))
        th[n] = float(betabinom.sf(thr - 1, rem, j, n + 1 - j))
    for order in ("random", "novelty"):
        C = np.array([r["replay"][order]["cov"] for r in rows], float)
        W = np.array([r["replay"][order]["nw"] for r in rows], float)
        MR = np.array([r["replay"][order]["rem_mean_abs_resid_norm"] for r in rows], float)
        ST = np.array([r["replay"][order]["straddle"] for r in rows], int)
        mean, p5 = C.mean(0), np.percentile(C, 5, axis=0)
        frac = (C >= C1_COV - TOL).mean(0)
        wmed = np.median(W, 0)
        ns = np.arange(n_max + 1)
        valid = ns >= 9          # n < 9: infinite interval (trivial coverage 1), cannot qualify
        ok1 = valid & (frac >= C1_FRAC - TOL)
        ok2 = valid & (p5 >= CRIT_P5 - TOL) & (mean >= CRIT_MEAN - TOL)

        def stay(ok, hz=horizon):
            for n in np.where(valid[:max(0, hz - STAY_MIN_WINDOW + 1)])[0]:
                if ok[n:hz + 1].all():
                    return int(n)
            return None

        def first(ok):
            w = np.where(ok)[0]
            return int(w[0]) if len(w) else None

        def at(n):
            if n is None:
                return None
            wv = W[:, n]
            return {"n_measured": int(n), "frac_of_T_measured": n / nT, "frac_of_T_unmeasured": 1 - n / nT,
                    "remainder_cov": summarize(C[:, n]), "remainder_cov_p5": float(p5[n]),
                    "remainder_cov_min": float(C[:, n].min()),
                    "frac_splits_ge_0.90": float(frac[n]),
                    "norm_width_median": float(np.median(wv)) if np.all(np.isfinite(wv)) else None,
                    "norm_width": summarize(wv[np.isfinite(wv)]),
                    "straddle_records_mean": float(ST[:, n].mean()), "straddle_records_max": int(ST[:, n].max())}

        n1s, n2s = stay(ok1), stay(ok2)
        vh = valid & (ns <= horizon)
        # sensitivity of the 'stays from' stopping points to the horizon convention (minimum remainder size)
        hz_sens = {}
        for rm in sorted(set(HZ_GRID + [h_rem])):
            if nT - rm - STAY_MIN_WINDOW >= 9:
                hz_sens[str(rm)] = {"horizon_n": int(nT - rm), "C1_task": stay(ok1, nT - rm), "C2_b_analogue": stay(ok2, nT - rm)}
        rec = {"C1_task": {"definition": f"remainder coverage >= {C1_COV} in >= {int(100*C1_FRAC)}% of splits, for every later n up to the stopping horizon",
                           "first_hit": first(ok1), "stays_from": n1s, "at": at(n1s),
                           "stays_from_strict_to_min_remainder": stay(ok1, n_max),
                           "max_frac_splits_ge_0.90_over_n>=9": float(frac[vh].max()),
                           "n_at_max_frac": int(ns[vh][np.argmax(frac[vh])]),
                           "median_frac_splits_ge_0.90_over_n>=9": float(np.median(frac[vh]))},
               "C2_b_analogue": {"definition": f"5th pct of remainder coverage >= {CRIT_P5} and mean >= {CRIT_MEAN}, for every later n up to the stopping horizon",
                                 "first_hit": first(ok2), "stays_from": n2s, "at": at(n2s),
                                 "stays_from_strict_to_min_remainder": stay(ok2, n_max)},
               "horizon_sensitivity_by_min_remainder": hz_sens,
               "straddle": {"note": ("remainder records sharing a composition with an already measured record (the replay "
                                     "cut is at record level; groups are contiguous in both orders)"),
                            "mean_over_n9_to_nmax_and_splits": float(ST[:, 9:].mean()) if n_max >= 9 else None,
                            "max": int(ST.max()),
                            "max_share_of_remainder": float(max(ST[:, n].max() / (nT - n) for n in range(n_max + 1)))}}
        # pre-registered, label-free stopping rules
        pre = {}
        for rname, rn in RULE_N.items():
            n = m if rname == "headline_m" else rn
            pre[rname] = at(n) if (n is not None and n <= n_max) else {"n_measured": n, "infeasible": True,
                                                                       "reason": f"n > |T| - {MIN_REM} = {n_max}"}
        rec["preregistered_stops"] = pre
        step = 1 if n_max <= 300 else 2
        keep = np.arange(0, n_max + 1, step)
        rec["curve"] = {"n": keep.tolist(), "mean": np.round(mean[keep], 4).tolist(),
                        "p5": np.round(p5[keep], 4).tolist(), "p50": np.round(np.percentile(C, 50, axis=0)[keep], 4).tolist(),
                        "p95": np.round(np.percentile(C, 95, axis=0)[keep], 4).tolist(),
                        "frac_ge_0.90": np.round(frac[keep], 3).tolist(),
                        "norm_width_median": [None if not np.isfinite(v) else round(float(v), 4) for v in wmed[keep]],
                        "rem_mean_abs_resid_norm": np.round(MR.mean(0)[keep], 4).tolist()}
        if order == "random":
            rec["curve"]["theory_P_ge_0.90_exchangeable"] = [None if not np.isfinite(v) else round(float(v), 4) for v in th[keep]]
            rec["theory_P_ge_0.90_exchangeable_max_n>=9"] = float(np.nanmax(th[:horizon + 1]))
            rec["theory_P_ge_0.90_exchangeable_median_n>=9"] = float(np.nanmedian(th[:horizon + 1]))
            rec["theory_P_ge_0.90_argmax_n"] = int(np.nanargmax(th[:horizon + 1]))
        R[order] = rec
    # paired comparison of widths at the same n (novelty vs random)
    comp = {}
    for n in sorted(set([9, 15, m, 30, 60, 120]) | {R[o][c]["stays_from"] for o in ("random", "novelty")
                                                   for c in ("C1_task", "C2_b_analogue") if R[o][c]["stays_from"]}):
        if n > n_max:
            continue
        cr = np.array([r["replay"]["random"]["cov"][n] for r in rows], float)
        cn = np.array([r["replay"]["novelty"]["cov"][n] for r in rows], float)
        wr = np.array([r["replay"]["random"]["nw"][n] for r in rows], float)
        wn = np.array([r["replay"]["novelty"]["nw"][n] for r in rows], float)
        comp[str(n)] = {"random_cov": summarize(cr), "novelty_cov": summarize(cn),
                        "random_nw_median": float(np.median(wr)), "novelty_nw_median": float(np.median(wn)),
                        "novelty_over_random_width": summarize(wn / wr)}
    R["novelty_vs_random_same_n"] = comp
    D["replay"] = R
    D["per_split"] = [{"seed": r["seed"], "n_pool": r["n_pool"], "n_test": r["n_test"],
                       "cov_at_m": r["sweep"][akey(ALPHA)][str(m)]["coverage"],
                       "nw_at_m": r["sweep"][akey(ALPHA)][str(m)]["norm_width"],
                       "pool_selected_k": r["pool_select"]["k"],
                       "novelty_spearman": r["novelty_spearman_resid"]} for r in rows]
    return D


# ================================================================= claim re-computation
def claims_check(DS):
    """Recompute the submitted manuscript's '65% and 90% summed across six cases'."""
    names = list(DS)
    nT = {n: DS[n]["n_T"] for n in names}
    tot = sum(nT.values())
    m = {n: DS[n]["headline_m"] for n in names}
    cheb = chebyshev_m(MANUSCRIPT_EPS, MANUSCRIPT_DELTA)
    # as coded in demo/exp4_campaign_replay.py (m_rule capped at n_target - 15, then at n_target - 5)
    orig_rule = {n: min(min(cheb, max(3, nT[n] - 15)), nT[n] - 5) for n in names}
    orig_heur = {n: min(m[n], nT[n] - 5) for n in names}
    out = {"n_T": nT, "total_T": tot, "chebyshev_m": cheb,
           "submitted_as_coded": {
               "heuristic_m": orig_heur, "heuristic_saved_pct_summed": 100 * (1 - sum(orig_heur.values()) / tot),
               "rule_m_as_coded": orig_rule, "rule_saved_pct_summed": 100 * (1 - sum(orig_rule.values()) / tot),
               "rule_truncated_in": [n for n in names if orig_rule[n] < cheb],
               "note": "the 'rule' is truncated to |T|-15 wherever |T| < 155, i.e. the rule itself was not applied there"},
           "protocol": {}}
    P = out["protocol"]
    P["heuristic_saved_pct_summed"] = 100 * (1 - sum(m.values()) / tot)
    P["heuristic_saved_pct_per_dataset"] = {n: 100 * (1 - m[n] / nT[n]) for n in names}
    P["heuristic_saved_pct_macro_mean"] = float(np.mean(list(P["heuristic_saved_pct_per_dataset"].values())))
    P["heuristic_saved_pct_summed_without_polymer"] = 100 * (1 - sum(m[n] for n in names if n != "polymer_Tg") /
                                                             sum(nT[n] for n in names if n != "polymer_Tg"))
    P["polymer_share_of_total_T"] = nT.get("polymer_Tg", 0) / tot
    P["heuristic_ttest_coverage"] = {n: {"mean": DS[n]["ksweep"][akey(ALPHA)][str(m[n])]["coverage"]["mean"],
                                         "lo": DS[n]["ksweep"][akey(ALPHA)][str(m[n])]["coverage"]["lo"],
                                         "min": DS[n]["ksweep"][akey(ALPHA)][str(m[n])]["min"],
                                         "frac_splits_ge_0.90": DS[n]["ksweep"][akey(ALPHA)][str(m[n])]["frac_ge_nominal"]}
                                     for n in names}
    P["chebyshev_feasible_in_recal_pool"] = {n: DS[n]["n_pool"]["min"] >= cheb for n in names}
    P["chebyshev_feasible_in_replay"] = {n: nT[n] - MIN_REM >= cheb for n in names}
    capped = {n: min(cheb, DS[n]["n_pool"]["min"]) for n in names}
    P["rule_capped_at_pool_saved_pct_summed"] = 100 * (1 - sum(capped.values()) / tot)
    eb = RULE_N["exact_beta_eps0.08_delta0.10"]
    P["exact_beta_rule_n"] = eb
    P["exact_beta_feasible_in_recal_pool"] = {n: DS[n]["n_pool"]["min"] >= eb for n in names}
    P["exact_beta_saved_pct_summed_if_feasible_everywhere"] = 100 * (1 - len(names) * eb / tot)
    for order in ("random", "novelty"):
        for crit in ("C1_task", "C2_b_analogue"):
            st = {n: DS[n]["replay"][order][crit]["stays_from"] for n in names}
            key = f"replay_{order}_{crit}"
            P[key] = {"stop_n": st, "n_datasets_reached": sum(v is not None for v in st.values()),
                      "stop_n_strict_min_remainder_10": {n: DS[n]["replay"][order][crit]["stays_from_strict_to_min_remainder"]
                                                         for n in names},
                      "note": "oracle stopping points under the default horizon convention; see horizon_sensitivity_by_min_remainder"}
            if all(v is not None for v in st.values()):
                P[key]["unmeasured_pct_summed"] = 100 * (1 - sum(st.values()) / tot)
                P[key]["unmeasured_pct_macro_mean"] = float(np.mean([100 * (1 - st[n] / nT[n]) for n in names]))
    # the submitted statement is about coverage on the *remaining* (unmeasured) records at the heuristic budget:
    # the matching protocol quantity is the random-order replay remainder at the pre-registered headline m
    P["heuristic_replay_remainder_random"] = {
        n: {k_: DS[n]["replay"]["random"]["preregistered_stops"]["headline_m"][k_]
            for k_ in ("n_measured", "frac_of_T_measured", "frac_of_T_unmeasured", "remainder_cov", "remainder_cov_min",
                       "remainder_cov_p5", "frac_splits_ge_0.90", "norm_width_median")} for n in names}
    return out


# ================================================================= data-driven interpretation
def interpret(res):
    DS = res["datasets"]; TH = res["theory"]; CC = res["claims_check"]; P = CC["protocol"]
    A = akey(ALPHA); names = list(DS)
    out = []
    hm = {n: DS[n]["ksweep"][A][str(DS[n]["headline_m"])] for n in names}
    nwlo = min(v["norm_width_median"] for v in hm.values()); nwhi = max(v["norm_width_median"] for v in hm.values())
    # (a) source-calibrated collapse and agreement with exchangeable theory
    k0 = {n: DS[n]["ksweep"][A]["0"]["coverage"]["mean"] for n in names}
    kc = [DS[n]["ksweep"][A][key]["coverage"]["mean"] for n in names for key in DS[n]["sweep_keys"]
          if key != "0" and DS[n]["ksweep"][A][key]["n_infinite_interval"] == 0]
    out.append(f"Source-calibrated intervals (k = 0) cover {min(k0.values()):.2f}-{max(k0.values()):.2f} of the target-test "
               f"half (mean over splits); every finite k >= 9 gives split-mean coverage {min(kc):.2f}-{max(kc):.2f} (nominal "
               "0.90), so the question is how variable the realised coverage is, and at what width.")
    d_res, d_first = [], []
    for n in names:
        S = DS[n]["ksweep"][A]
        for key in DS[n]["sweep_keys"]:
            if key in ("0", "full_pool") or S[key]["n_infinite_interval"] > 0 or not S[key].get("theory_betabinom"):
                continue
            th5 = S[key]["theory_betabinom"]["q5_50_95"][0]
            d_res.append(S[key]["resampled"]["p5"] - th5); d_first.append(S[key]["p5"] - th5)
    out.append("Because the recalibration pool and the target-test half are a random (grouped) partition of the same "
               "target pool, calibration and test residuals are exchangeable by construction, and realised coverage "
               f"follows the exact Beta-Binomial law: across datasets and k >= 9 the 5th percentile of the {B_SUB}-subset "
               f"resampling differs from the theoretical one by a median {np.median(d_res):+.3f} (range {min(d_res):+.2f} "
               f"to {max(d_res):+.2f}); the 20-split first-k percentiles scatter more (median {np.median(d_first):+.3f}, "
               f"range {min(d_first):+.2f} to {max(d_first):+.2f}) because 20 nested draws estimate a 5th percentile "
               "poorly. The agreement is guaranteed by the design, not an empirical property of the materials: under a "
               "random partition the coverage budget is a function of k and n_test only. It does not establish "
               "exchangeability for prospective, non-random acquisitions (the novelty-order replay in (e) breaks it), and "
               "the width cost, i.e. the usefulness of the interval, is material-dependent (median normalised width at the "
               f"headline m {nwlo:.1f}-{nwhi:.1f} x the target-property IQR).")
    out.append(f"At the headline m ({min(DS[n]['headline_m'] for n in names)}-{max(DS[n]['headline_m'] for n in names)} "
               "target records) mean target-test coverage is "
               f"{min(v['coverage']['mean'] for v in hm.values()):.3f}-{max(v['coverage']['mean'] for v in hm.values()):.3f}, "
               f"but single splits fall to {min(v['min'] for v in hm.values()):.2f} and only "
               f"{min(v['frac_ge_nominal'] for v in hm.values()):.2f}-{max(v['frac_ge_nominal'] for v in hm.values()):.2f} "
               f"of splits reach 0.90; the median normalised width is {nwlo:.1f}-{nwhi:.1f} x the target-property IQR.")
    # grouping caveat (polymer)
    if "polymer_Tg" in DS:
        S = DS["polymer_Tg"]["ksweep"][A]
        out.append("Protocol caveat (polymer_Tg): replicate records of one composition stay contiguous in the grouped "
                   f"permutation, so the first k = 9 pool records contain on average {S['9']['n_distinct_compositions_mean']:.1f} "
                   f"distinct compositions; first-k mean coverage at k = 9/10 is {S['9']['coverage']['mean']:.3f}/"
                   f"{S['10']['coverage']['mean']:.3f} versus {S['9']['resampled']['mean']:.3f}/{S['10']['resampled']['mean']:.3f} "
                   "for record-level random subsets of the same pool (theory 0.900/0.909). Small recalibration budgets "
                   "should be counted in distinct compositions when replicates exist.")
    # (b) oracle
    mk = {n: DS[n]["min_k"] for n in names}
    s = "; ".join(f"{n} {nv(mk[n]['protocol_first_k']['first'])}"
                  + (f" (stable {nv(mk[n]['protocol_first_k']['stable'])})" if mk[n]['protocol_first_k']['stable'] != mk[n]['protocol_first_k']['first'] else "")
                  + (f" = {100*mk[n]['protocol_first_k']['first']/DS[n]['n_T']:.0f}% of |T|" if mk[n]['protocol_first_k']['first'] else "")
                  for n in names)
    near = []
    for n in names:
        for key, p5v in (mk[n].get("oracle_near_misses_p5_within_0.01_below") or {}).items():
            near.append(f"{n} p5 = {p5v:.4f} at k = {DS[n]['ksweep'][A][key]['k']}{' (pool)' if key == 'full_pool' else ''}")
    tha = {n: mk[n]["theory_betabinom_all_integers"] for n in names}
    out.append(f"(b, oracle) Smallest grid k with 5th-percentile coverage >= {CRIT_P5} and mean >= {CRIT_MEAN}: {s}. "
               "For the small target regions these first/stable labels are noise-level: with 20 splits the 5th percentile "
               "is essentially the second-lowest split"
               + (f", and several verdicts rest on near misses ({'; '.join(near)})" if near else "")
               + ". The reliable reference is exchangeable theory for the actual target-test size over all integer k "
                 "(Beta-Binomial 5th percentile >= 0.80; first / stable over k <= 3000; stable within the pool): "
               + "; ".join(f"{n} (n_test {tha[n]['n_test']}) {nv(tha[n]['first'])} / {nv(tha[n]['stable'])}"
                           f"; within pool ({tha[n]['pool_min']}) {nv(tha[n]['stable_within_pool'])}" for n in names)
               + f". With an infinite test set the Beta law needs n = {mk[names[0]]['theory_beta_infinite_test']['first']} "
                 f"(stably {mk[names[0]]['theory_beta_infinite_test']['stable']})."
               + (f" Never stable at any k: {', '.join(n for n in names if tha[n]['stable'] is None)} (the Binomial(n_test, 0.9) "
                  "5th percentile, the limit as k grows, is below 0.80, so the target-test half is too small for the criterion)."
                  if any(tha[n]["stable"] is None for n in names) else "")
               + (" Stable only beyond the recalibration pool: "
                  + ", ".join(f"{n} (from k = {tha[n]['stable']}, pool {tha[n]['pool_min']})" for n in names
                              if tha[n]["stable"] is not None and tha[n]["stable_within_pool"] is None) + "."
                  if any(tha[n]["stable"] is not None and tha[n]["stable_within_pool"] is None for n in names) else ""))
    # (b) protocol-compliant
    ps = {n: mk[n]["pool_internal_selection"] for n in names}
    ck = [v for n in names for v in ps[n]["chosen_k_values"] if v is not None]
    kdist = {k: ck.count(k) for k in sorted(set(ck))}
    n_unst = sum(ps[n]["mc_replicates"]["n_splits_certification_unstable"] for n in names)
    n_kunst = sum(ps[n]["mc_replicates"]["n_splits_chosen_k_unstable"] for n in names)
    none_cert = [n for n in names if ps[n]["mc_replicates"]["n_certified_max"] == 0]
    out.append(f"(b, protocol-compliant) Choosing k from the recalibration pool alone ({B_SEL} grouped inner splits: the "
               "first k records of a grouped permutation calibrate, the remaining whole compositions are held out) certifies "
               "a budget in "
               + ", ".join(f"{n} {ps[n]['n_splits_certified']}/20 (range {ps[n]['mc_replicates']['n_certified_min']}-"
                           f"{ps[n]['mc_replicates']['n_certified_max']} over {R_SEL} inner-RNG replicates)" for n in names)
               + f" splits. The counts are approximate: the criterion is a knife edge on a coarse lattice (steps of "
                 f"1/(pool - k) of the inner holdout), and {n_unst} of the 120 split-level certify / not-certify decisions "
                 f"and {n_kunst} chosen k values change between replicates. The inner holdout (pool - k records) is smaller "
                 "than the target-test half, so the inner 5th percentile is more pessimistic than the one it predicts. Chosen k "
                 f"(primary replicate, all certified splits): {kdist}. Target-test coverage at the chosen k: "
               + ", ".join(f"{n} mean {ps[n]['ttest_cov_when_certified']['mean']:.3f} (p5 {ps[n]['ttest_cov_p5_when_certified']:.2f}, "
                           f"min {ps[n]['ttest_cov_min_when_certified']:.2f})"
                           for n in names if ps[n]['n_splits_certified'])
               + (f". {', '.join(none_cert)} cannot certify any budget from their own pool in any replicate." if none_cert else ".")
               + " Certification inside the pool does not transfer to a single target-test realisation: at the chosen k the "
                 "target-test 5th percentile is "
               + ", ".join(f"{n} {ps[n]['ttest_cov_p5_when_certified']:.2f}" for n in names if ps[n]['n_splits_certified'])
               + " and the worst split "
               + ", ".join(f"{n} {ps[n]['ttest_cov_min_when_certified']:.2f}" for n in names if ps[n]['n_splits_certified'])
               + ", i.e. below the 0.80 that was certified for "
               + (", ".join(n for n in names if ps[n]['n_splits_certified'] and ps[n]['ttest_cov_p5_when_certified'] < CRIT_P5 - TOL) or "no dataset")
               + ". The inner criterion describes the distribution over inner holdouts; it is not a guarantee for one "
                 "evaluation half, which is itself a finite sample.")
    def kset(vals):
        v = [x for x in vals if x is not None]
        return ", ".join(f"{k}x{v.count(k)}" for k in sorted(set(v))) if v else "none"
    rep_ds = [n for n in names if ps[n]["n_pool_compositions_mean"] < 0.95 * DS[n]["n_pool"]["median"]]
    norep_ds = [n for n in names if n not in rep_ds]
    diff = []
    for n in rep_ds:
        rl = ps[n]["record_level_inner_sensitivity"]
        diff.append(f"{n} (pool {DS[n]['n_pool']['median']:.0f} records, {ps[n]['n_pool_compositions_mean']:.0f} compositions): "
                    f"record-level chose k = {kset(rl['chosen_k_values'])} with target-test p5 {f2(rl['ttest_cov_p5_when_certified'])}; "
                    f"grouped chooses k = {kset(ps[n]['chosen_k_values'])} with target-test p5 {f2(ps[n]['ttest_cov_p5_when_certified'])}")
    out.append("Record-level vs grouped inner splits. Where the pool contains replicate records, "
               + "; ".join(diff) + ". With record-level inner splits, replicates of one composition sit on both sides of "
               "the inner split, so the pool looks easier than the deployed first-k rule, which calibrates on fewer distinct "
               "compositions. For polymer_Tg this mismatch explains why the previous version certified k = 15 in every split "
               "and then saw a target-test 5th percentile below the 0.80 it had certified."
               + (f" {', '.join(norep_ds)} have (almost) no replicates, so the two versions are the same procedure and "
                  "differ only by Monte Carlo noise ("
                  + "; ".join(f"{n} record-level {ps[n]['record_level_inner_sensitivity']['n_splits_certified']}/20 vs grouped "
                              f"{ps[n]['n_splits_certified']}/20 (replicates {ps[n]['mc_replicates']['n_certified_min']}-"
                              f"{ps[n]['mc_replicates']['n_certified_max']})" for n in norep_ds)
                  + "), another sign that the certification counts are knife-edge." if norep_ds else ""))
    # (c)
    r = next(x for x in TH[A] if abs(x["eps"] - 0.08) < 1e-9 and abs(x["delta"] - 0.10) < 1e-9)
    ratios = [x["chebyshev_m"] / x["exact_beta_stable"] for x in TH[A]]
    out.append(f"(c) The submitted Chebyshev rule is valid but {min(ratios):.1f}-{max(ratios):.1f}x more demanding than the "
               f"exact Beta rule (stable n) at alpha = 0.10; for the manuscript's eps = 0.08, delta = 0.10 it asks for "
               f"{r['chebyshev_m']} records where {r['exact_beta_stable']} suffice (exact failure probability at "
               f"{r['chebyshev_m']}: {r['P_beta_fail_at_chebyshev_m']:.3f}). The manuscript's 'knee at about 13' matches the "
               f"first exact n ({r['exact_beta_first']}), which lies on a conservative saw-tooth dip; the stable exact value "
               f"is {r['exact_beta_stable']}. {r['chebyshev_m']} fits inside the recalibration pool only for "
               f"{', '.join(n for n, v in P['chebyshev_feasible_in_recal_pool'].items() if v) or 'no dataset'}.")
    BR = {n: DS[n]["budget_rules"]["by_eps_delta"]["eps0.08_delta0.10"] for n in names}
    unst = [n for n in names if not BR[n]["betabinom_stably_reachable"]]
    parts = []
    for n in unst:
        v = BR[n]
        parts.append(f"{n} (n_test {DS[n]['budget_rules']['n_test_median']}): limit {v['betabinom_limit_P_fail_k_to_inf']:.3f}, "
                     f"delta met at k = {runs_str(v['betabinom_k_runs_meeting_delta_upto_200'], 4)}, minimum "
                     f"{v['betabinom_min_P_fail_k_upto_200']:.3f} at k = {v['betabinom_argmin_k_upto_200']}; within the {DS[n]['n_pool']['min']}-record "
                     f"pool: {runs_str(v['betabinom_k_runs_meeting_delta_within_pool'])}")
    out.append("With a finite target-test half, the exact finite-test (Beta-Binomial) failure probability tends, as k grows, "
               "to the Binomial(n_test, 0.9) tail; at eps = 0.08 this limit is "
               + ", ".join(f"{n} {BR[n]['betabinom_limit_P_fail_k_to_inf']:.3f}" for n in names)
               + ". It is a limit, not a floor"
               + (f". Where it exceeds delta = 0.10 ({', '.join(unst)}), no budget meets delta stably, but delta is still met at "
                  "conservative saw-tooth sizes where the interval is the sample maximum or close to it (" + "; ".join(parts)
                  + "). A budget read off such a dip is fragile: adding one record can break it. Small design regions therefore "
                  "cannot support a stable (eps 0.08, delta 0.10) guarantee on their own evaluation set." if unst else "."))
    # (d)
    wr = {n: DS[n]["alpha_sensitivity"]["tradeoff"][str(DS[n]["headline_m"])] for n in names}
    out.append("(d) alpha = 0.20 gives a finite interval from k = 4 (vs 9 at alpha = 0.10). At the headline m the mean over "
               "splits of the per-split width ratio (alpha 0.20 / alpha 0.10) is "
               + ", ".join(f"{n} {wr[n]['width_ratio_0.20_over_0.10']['mean']:.2f}" for n in names)
               + ", with mean coverage "
               f"{min(wr[n]['alpha_0.20']['cov_mean'] for n in names):.2f}-{max(wr[n]['alpha_0.20']['cov_mean'] for n in names):.2f} "
               f"(5th percentile down to {min(wr[n]['alpha_0.20']['cov_p5'] for n in names):.2f}); at k = 4 the 5th "
               f"percentile is {min(DS[n]['alpha_sensitivity']['tradeoff']['4']['alpha_0.20']['cov_p5'] for n in names):.2f}-"
               f"{max(DS[n]['alpha_sensitivity']['tradeoff']['4']['alpha_0.20']['cov_p5'] for n in names):.2f}. A looser "
               "alpha buys fewer required records and narrower intervals at the price of a weaker, noisier guarantee.")
    # (e)
    rr = {n: DS[n]["replay"] for n in names}
    hz_lbl = f"remainder >= max({STOP_MIN_REM}, {int(100*STOP_MIN_REM_FRAC)}% of |T|)"
    c1_any = [f"{n} n = {v['C1_task']} (remainder >= {rm})" for n in names
              for rm, v in rr[n]["random"]["horizon_sensitivity_by_min_remainder"].items() if v["C1_task"] is not None]
    c2r = {n: rr[n]["random"]["C2_b_analogue"]["stays_from"] for n in names}
    c2s = {n: rr[n]["random"]["C2_b_analogue"]["stays_from_strict_to_min_remainder"] for n in names}
    out.append("(e) Random-order replay: the task criterion C1 (remainder coverage >= 0.90 in >= 80% of splits for every "
               f"later n, verified over >= {STAY_MIN_WINDOW} further measurements) is reached in "
               f"{sum(rr[n]['random']['C1_task']['stays_from'] is not None for n in names)}/6 datasets under the default "
               "horizon" + (" and under every other horizon tested" if not c1_any else
                            f"; under other horizons only {'; '.join(c1_any)}")
               + ". Exchangeable theory predicts this. The weaker C2 criterion is reached, under the default horizon "
               f"convention ({hz_lbl}), at "
               + ", ".join(f"{n} {'never' if c2r[n] is None else c2r[n]}" for n in names)
               + f"; with the strict convention (remainder down to {MIN_REM}) these become "
               + ", ".join(f"{n} {nv(c2s[n])}" for n in names)
               + ". The random-order C2 stopping points therefore depend on an analyst-chosen horizon and must not be quoted "
                 "as a share of |T| 'saved' without that caveat (default-horizon values: "
               + ", ".join(f"{n} {100*c2r[n]/DS[n]['n_T']:.0f}% of |T|" for n in names if c2r[n] is not None) + ").")
    c1n = {n: rr[n]["novelty"]["C1_task"]["stays_from"] for n in names}
    wrat, wmax = {}, {}
    for n in names:
        comp = rr[n]["novelty_vs_random_same_n"]
        if c1n[n] is not None and str(c1n[n]) in comp:
            wrat[n] = comp[str(c1n[n])]["novelty_over_random_width"]["mean"]
        wmax[n] = max(((v["novelty_over_random_width"]["mean"], int(k)) for k, v in comp.items()), default=(None, None))
    out.append("Novelty-order replay reaches C1 (default horizon) at " + ", ".join(
        f"{n} {'never' if c1n[n] is None else f'{c1n[n]} ({100*c1n[n]/DS[n]['n_T']:.0f}%)'}" for n in names)
        + " measurements. The mechanism is non-exchangeability in the favourable direction: the novel candidates measured "
        "first tend to have larger errors than the familiar ones left unmeasured (mean |residual| of the remainder falls as "
        "measurement proceeds), so a quantile calibrated on the former over-covers the latter. Sometimes this costs width: "
        "novelty/random width ratio at the C1 stopping n is " + ", ".join(f"{n} {v:.2f}" for n, v in wrat.items())
        + "; the largest ratio among the compared n is " + ", ".join(f"{n} {v[0]:.2f} (n = {v[1]})" for n, v in wmax.items())
        + ". It is not a guarantee: whether it over- or under-covers depends on how novelty relates to error in each dataset.")
    chg = []
    for n in names:
        for order in ("random", "novelty"):
            for crit in ("C1_task", "C2_b_analogue"):
                a_ = rr[n][order][crit]["stays_from"]; b_ = rr[n][order][crit]["stays_from_strict_to_min_remainder"]
                if a_ != b_:
                    chg.append(f"{n} {order} {crit.split('_')[0]} {nv(a_)} -> {nv(b_)}")
    if chg:
        out.append(f"Horizon dependence of the oracle stopping points (default {hz_lbl} -> strict remainder >= {MIN_REM}): "
                   + "; ".join(chg) + ". Full grid in the horizon-sensitivity table. For novelty order a change means that "
                   "coverage on the last, most familiar candidates breaks the criterion after the default horizon.")
    under = []
    for n in names:
        c = rr[n]["novelty"]["curve"]
        vals = [(c["mean"][i], c["n"][i]) for i in range(len(c["n"])) if 9 <= c["n"][i] <= DS[n]["n_T"] - MIN_REM]
        lo = min(vals)
        if lo[0] < 0.88:
            under.append(f"{n} mean remainder coverage {lo[0]:.2f} at n = {lo[1]}")
    if under:
        out.append("Novelty-order under-coverage observed: " + "; ".join(under) + ".")
    sp = {n: rr[n]["novelty_spearman_with_abs_residual"]["mean"] for n in names}
    out.append("Spearman correlation of the novelty score with |residual| on the target pool: "
               + ", ".join(f"{n} {v:.2f}" for n, v in sp.items()) + ".")
    stv = {n: max(rr[n][o]["straddle"]["mean_over_n9_to_nmax_and_splits"] or 0 for o in ("random", "novelty")) for n in names}
    stm = {n: max(rr[n][o]["straddle"]["max"] for o in ("random", "novelty")) for n in names}
    sts = []
    for n in names:
        for o in ("random", "novelty"):
            for crit in ("C1_task", "C2_b_analogue"):
                at_ = rr[n][o][crit]["at"]
                if at_ is not None:
                    sts.append(at_["straddle_records_mean"])
    out.append("Grouping in the replay: the cut between measured and unmeasured records is at record level (groups are "
               "contiguous in both orders), so a remainder record can share a composition with a measured one. Mean over n "
               "and splits (max): " + ", ".join(f"{n} {stv[n]:.2f} ({stm[n]})" for n in names)
               + f"; at the oracle stopping points the mean is at most {max(sts) if sts else 0:.2f} records. The effect on "
                 "remainder coverage is negligible, but strictly the replay treats replicate records as separate candidates.")
    # claims
    S0 = CC["submitted_as_coded"]
    hc = P["heuristic_ttest_coverage"]
    hr = P["heuristic_replay_remainder_random"]
    per = P["heuristic_saved_pct_per_dataset"]
    out.append(f"'65% and 90% summed across six cases': as coded in demo/exp4 these are {S0['rule_saved_pct_summed']:.1f}% "
               f"(rule) and {S0['heuristic_saved_pct_summed']:.1f}% (heuristic). Both are arithmetic in |T| rather than "
               f"experimental outcomes; the rule was truncated to |T|-15 in {len(S0['rule_truncated_in'])}/6 cases (so the 65% "
               "is not the rule), and the summed figure is dominated by polymer_Tg "
               f"({100*P['polymer_share_of_total_T']:.0f}% of all target records). Under the protocol the Chebyshev rule is "
               f"executable in {sum(P['chebyshev_feasible_in_recal_pool'].values())}/6 datasets; the heuristic's "
               f"{P['heuristic_saved_pct_summed']:.1f}% summed ({P['heuristic_saved_pct_macro_mean']:.1f}% per-dataset mean, "
               f"{min(per.values()):.0f}-{max(per.values()):.0f}% per dataset) is unchanged arithmetically.")
    lo_m = min(v["remainder_cov"]["mean"] for v in hr.values()); hi_m = max(v["remainder_cov"]["mean"] for v in hr.values())
    below = [n for n in names if hr[n]["remainder_cov"]["mean"] < 0.90 - TOL or hc[n]["mean"] < 0.90 - TOL]
    out.append("The accompanying statement that coverage on the remaining measurements 'stays at or above the nominal 0.90 "
               "throughout (0.91 to 0.97)' has two readings. (i) As a mean over re-partitions, which is how the original "
               f"figures were computed: under the protocol, mean remainder coverage in the random-order replay at the heuristic "
               f"m is {lo_m:.3f}-{hi_m:.3f} and mean target-test coverage at m is "
               f"{min(v['mean'] for v in hc.values()):.3f}-{max(v['mean'] for v in hc.values()):.3f}, so the mean reading "
               "essentially holds"
               + (f" ({', '.join(below)} marginally below 0.90: replay "
                  + ", ".join(f"{n} {hr[n]['remainder_cov']['mean']:.3f}" for n in below) + ", target-test "
                  + ", ".join(f"{n} {hc[n]['mean']:.3f}" for n in below) + ")" if below else "")
               + "; the original 0.91-0.97 were partly inflated by the one-rank-too-high quantile. (ii) Per realisation it "
               f"does not hold: {100*(1-max(v['frac_splits_ge_0.90'] for v in hc.values())):.0f}-"
               f"{100*(1-min(v['frac_splits_ge_0.90'] for v in hc.values())):.0f}% of splits fall below 0.90 on the target-test "
               f"half (worst split {min(v['min'] for v in hc.values()):.2f}) and "
               f"{100*(1-max(v['frac_splits_ge_0.90'] for v in hr.values())):.0f}-"
               f"{100*(1-min(v['frac_splits_ge_0.90'] for v in hr.values())):.0f}% on the replay remainder (worst split "
               f"{min(v['remainder_cov_min'] for v in hr.values()):.2f}). Split conformal never promises per-realisation "
               "coverage, so 'throughout' should be replaced by the mean plus the worst-split range.")
    fm = [hr[n]["frac_of_T_measured"] for n in names]
    out.append("A defensible replacement sentence (all numbers from the random-order replay at the pre-registered headline m): "
               f"'In a retrospective replay in which {min(DS[n]['headline_m'] for n in names)}-{max(DS[n]['headline_m'] for n in names)} "
               f"target records ({100*min(fm):.0f}-{100*max(fm):.0f}% of each design region) were measured and the rest "
               f"received calibrated intervals, mean coverage on the unmeasured {100*(1-max(fm)):.0f}-{100*(1-min(fm)):.0f}% was "
               f"{lo_m:.2f}-{hi_m:.2f} across 20 random splits (worst split "
               f"{min(v['remainder_cov_min'] for v in hr.values()):.2f}-{max(v['remainder_cov_min'] for v in hr.values()):.2f}) at "
               f"median widths of {min(v['norm_width_median'] for v in hr.values()):.1f}-"
               f"{max(v['norm_width_median'] for v in hr.values()):.1f} x the target-property IQR. These are coverage-certified "
               "intervals, not measurements, and the replay is not a prospective campaign.'")
    oc = TH["original_cq_check"]
    budgets = sorted(set(CC["submitted_as_coded"]["heuristic_m"].values()) | set(CC["submitted_as_coded"]["rule_m_as_coded"].values()))
    affected = [b for b in budgets if int(np.quantile(np.arange(1.0, b + 1), min(np.ceil((b + 1) * (1 - ALPHA)) / b, 1.0),
                                                      method="higher")) != j_rank(b)]
    unaffected = [b for b in budgets if b not in affected]
    hi = [x for x in oc if x["conformal_rank"] != "inf" and x["orig_rank"] > x["conformal_rank"]]
    ex = next((x for x in oc if x["n"] == 22), hi[0] if hi else None)
    out.append("The original demo quantile np.quantile(r, min(ceil((n+1)(1-a))/n, 1), method='higher') returns one order "
               f"statistic too high for n in {[x['n'] for x in hi]} (checked sizes)"
               + (f", e.g. rank {ex['orig_rank']} of {ex['n']} instead of {ex['conformal_rank']} (E[coverage] "
                  f"{ex['orig_expected_cov']:.3f} vs {ex['conformal_expected_cov']:.3f})" if ex else "")
               + ", and a finite sample maximum instead of an infinite interval for n < 9. This biases the submitted "
               "replay coverages (0.91-0.97) upwards at the budgets used there that are affected: "
               f"{affected} (unaffected: {unaffected}).")
    return out


# ================================================================= markdown
def f3(x):
    return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def nv(x):
    return "never" if x is None else str(x)


def f2(x):
    return "n/a" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


def write_md(res):
    DS = res["datasets"]; TH = res["theory"]; CC = res["claims_check"]
    L = []
    w = L.append
    w("# R8 budget: the cost of target recalibration and a retrospective replay\n")
    w(f"Script `revision/code/exp_R8_budget.py`; JSON `revision/results/R8_budget.json`. Reference model RF "
      f"(common.make_model), unified protocol (grouped 60/20/20 source split; target pool halves = recalibration pool / "
      f"target-test), alpha = 0.10 unless stated, finite-sample conformal quantile, 20 splits (seeds 0-19). "
      f"Intervals across splits are 2.5-97.5 percentiles; the splits resample one finite dataset, so they describe "
      f"split-to-split variability, not population sampling error. Runtime {res['meta']['runtime_min']:.1f} min.\n")
    w("Scope statements that apply to everything below:\n")
    w("* Unmeasured candidates receive calibrated **intervals**, not measured values. Any 'saving' is a saving in "
      "coverage-certified-prediction terms only (fewer candidates need a measurement to obtain an interval with the "
      "stated marginal coverage); it is not a saving in experimental outcomes, and the interval is wide "
      "(normalised widths below).")
    w("* The replay assumes that outcomes do not depend on acquisition order, uniform cost per measurement, no batch "
      "or instrument effects, and that historical records are an unbiased stand-in for fresh measurements. It is a "
      "retrospective look-up-table replay of existing records, **not a prospective closed loop**.")
    w("* Quantities marked *oracle* read target-test / remainder labels and are descriptive; nothing is selected with "
      "them. Protocol-compliant counterparts (selection inside the recalibration pool only, or label-free "
      "pre-registered stopping rules) are reported alongside.\n")

    # ---- (a)
    w("## (a) Realised target-test coverage vs recalibration size k\n")
    w("Recalibration on the first k records of the recalibration pool, evaluation on the same target-test half. "
      "`Beta` = exchangeable conditional-coverage law for an infinite test set (common.beta_coverage_quantiles); "
      "`BetaBin` = exact exchangeable law for the actual target-test size (mixed over splits). k = 0 is the "
      "source-calibrated interval; k < 9 gives an infinite interval at alpha = 0.10 (reported, not dropped). "
      "`nw` = interval width / IQR of the target property; `res p5` = 5th percentile over 20 x "
      f"{B_SUB} random k-subsets of the pool (supplement).\n")
    for name, D in DS.items():
        S = D["ksweep"]["alpha_0.10"]
        w(f"### {name} (|T| = {D['n_T']}, pool {D['n_pool']['median']:.0f}, target-test {D['n_test']['median']:.0f}, headline m = {D['headline_m']})\n")
        w("| k | inf. | cov mean [2.5, 97.5] | p5 / p50 / p95 | Beta 5/50/95 | BetaBin 5/50/95 | frac >= 0.90 | res p5 | nw median |")
        w("|---|---|---|---|---|---|---|---|---|")
        for key in D["sweep_keys"]:
            e = S[key]
            kl = f"{e['k']}" + (" (pool)" if key == "full_pool" else "")
            b = e.get("theory_beta_5_50_95"); bb = e.get("theory_betabinom")
            bs = "n/a" if (b is None or b[0] is None or (isinstance(b[0], float) and np.isnan(b[0]))) else "/".join(f"{x:.2f}" for x in b)
            bbs = "inf" if (key != "0" and bb is None) else ("n/a" if bb is None else "/".join(f"{x:.2f}" for x in bb["q5_50_95"]))
            cs = e["coverage"]
            w(f"| {kl} | {e['n_infinite_interval']} | {cs['mean']:.3f} [{cs['lo']:.2f}, {cs['hi']:.2f}] | "
              f"{e['p5']:.2f} / {e['p50']:.2f} / {e['p95']:.2f} | {bs} | {bbs} | {e['frac_ge_nominal']:.2f} | "
              f"{f2(e['resampled']['p5']) if 'resampled' in e else 'n/a'} | {f2(e['norm_width_median'])} |")
        w("")

    # ---- (b)
    w("## (b) Smallest adequate k (5th-percentile coverage >= 0.80 and mean >= 0.88)\n")
    w("*Oracle* columns use target-test labels (descriptive). 'first' = smallest grid k meeting the criterion; "
      "'stable' = smallest k from which every larger grid k also meets it. With 20 splits the 5th percentile is "
      "essentially the second-lowest split, so for the small target regions the oracle labels are noise-level; the "
      "reliable reference is the Beta-Binomial theory column (exact exchangeable law for the actual target-test size, "
      "evaluated at every integer k, not only the grid).\n")
    w("| dataset | |T| | pool | oracle first / stable | k/|T| (first) | resampled first | BetaBin theory, all integer k: first / stable (stable within pool) |")
    w("|---|---|---|---|---|---|---|")
    for name, D in DS.items():
        M = D["min_k"]; th = M["theory_betabinom_all_integers"]
        w(f"| {name} | {D['n_T']} | {D['n_pool']['median']:.0f} | {nv(M['protocol_first_k']['first'])} / "
          f"{nv(M['protocol_first_k']['stable'])} | {f2(M['protocol_first_k']['first_frac_of_T'])} | "
          f"{nv(M['resampled_subsets']['first'])} | {nv(th['first'])} / {nv(th['stable'])} "
          f"({nv(th['stable_within_pool'])}, pool {th['pool_min']}; n_test {th['n_test']}) |")
    bi = next(iter(DS.values()))["min_k"]["theory_beta_infinite_test"]
    w(f"\nExchangeable theory with an infinite test set (Beta): 5th percentile >= 0.80 first at n = {bi['first']}, "
      f"stably from n = {bi['stable']}.\n")
    nm = [f"{n}: p5 = {v:.4f} at k = {DS[n]['ksweep']['alpha_0.10'][key]['k']}{' (pool)' if key == 'full_pool' else ''}"
          for n in DS for key, v in (DS[n]["min_k"].get("oracle_near_misses_p5_within_0.01_below") or {}).items()]
    if nm:
        w("Oracle near misses (5th percentile within 0.01 below 0.80): " + "; ".join(nm) + ".\n")
    w(f"**Protocol-compliant selection.** k is chosen from the recalibration pool alone and then evaluated once on "
      f"target-test. Each of {B_SEL} inner draws is a grouped permutation of the pool; the first k records calibrate "
      "(this is the deployed first-k rule) and the remaining pool records whose composition does not occur in the "
      f"calibration set are held out (whole compositions only, >= {MIN_REM} records). The first grid k whose inner-holdout "
      "coverage has 5th percentile >= 0.80 and mean >= 0.88 is chosen. Primary = inner-RNG replicate 0; "
      f"{R_SEL} independent replicates give the Monte Carlo spread of the certification count. The inner holdout "
      "(pool - k records) is smaller than the target-test half, so the inner 5th percentile is more pessimistic than "
      "the target-test one. Record-level inner splits (the previous version, which put replicate records on both sides "
      "of the inner split and so violated the grouped protocol) are shown as a sensitivity only.\n")
    w("| dataset | pool compositions (mean) | certified splits (range over replicates) | splits whose certification / chosen k changes across replicates | chosen k (primary) | target-test cov at chosen k: mean, p5, min | record-level (sensitivity): certified, chosen k, target-test p5 |")
    w("|---|---|---|---|---|---|---|")
    for name, D in DS.items():
        P = D["min_k"]["pool_internal_selection"]; mc = P["mc_replicates"]; rl = P["record_level_inner_sensitivity"]
        kv = [v for v in P["chosen_k_values"] if v is not None]
        kd = ", ".join(f"{k}x{kv.count(k)}" for k in sorted(set(kv))) if kv else "none"
        tc = P["ttest_cov_when_certified"]
        tcs = "n/a" if tc["mean"] is None else (f"{tc['mean']:.3f}, {f2(P['ttest_cov_p5_when_certified'])}, "
                                               f"{f2(P['ttest_cov_min_when_certified'])}")
        rv = [v for v in rl["chosen_k_values"] if v is not None]
        rkd = ", ".join(f"{k}x{rv.count(k)}" for k in sorted(set(rv))) if rv else "none"
        w(f"| {name} | {P['n_pool_compositions_mean']:.0f} | {P['n_splits_certified']}/20 ({mc['n_certified_min']}-"
          f"{mc['n_certified_max']}) | {mc['n_splits_certification_unstable']} / {mc['n_splits_chosen_k_unstable']} | {kd} | {tcs} | "
          f"{rl['n_splits_certified']}/20, {rkd}, {f2(rl['ttest_cov_p5_when_certified'])} |")
    w("")

    # ---- (c)
    w("## (c) Chebyshev budget rule (submitted manuscript) vs exact Beta rule\n")
    w("Chebyshev: m = alpha(1-alpha)/(delta eps^2) - 1 (rounded up), as in the submitted manuscript and demo/exp6. "
      "Exact: smallest n with P_Beta(coverage < 1-alpha-eps) <= delta, where coverage ~ Beta(j, n+1-j), "
      "j = ceil((n+1)(1-alpha)) (conditional coverage over an infinite test population). Because j is discrete the "
      "failure probability is saw-toothed in n; 'first' is the first n meeting the target, 'stable' the n from which it "
      "is met for every larger n. Cantelli = one-sided Chebyshev with the same variance bound.\n")
    for a in ALPHAS:
        w(f"alpha = {a:.2f}\n")
        w("| eps | delta | Chebyshev m | Cantelli m | exact Beta first / stable | Chebyshev / exact | P_Beta(fail) at Chebyshev m |")
        w("|---|---|---|---|---|---|---|")
        for r in TH[akey(a)]:
            w(f"| {r['eps']:.2f} | {r['delta']:.2f} | {r['chebyshev_m']} | {r['cantelli_m']} | {r['exact_beta_first']} / "
              f"{r['exact_beta_stable']} | {f2(r['chebyshev_over_exact'])} | {r['P_beta_fail_at_chebyshev_m']:.2g} |")
        w("")
    w(f"Variance bound used in the manuscript (Var <= alpha(1-alpha)/(m+1)) holds for n = 9..2000: "
      f"{TH['variance_bound_check']['holds']} (max ratio {TH['variance_bound_check']['max_Var_over_bound_n9_2000']:.3f}).\n")
    w("Per dataset, the realised coverage is measured on a finite target-test half, so binomial test-set noise adds to "
      "the calibration-set noise. As k grows, the exact finite-test (Beta-Binomial) failure probability tends to the "
      "Binomial(n_test, 0.9) tail. This is a **limit, not a floor**: at conservative saw-tooth sizes (the conformal order "
      "statistic is the sample maximum or close to it) the failure probability dips below it. Where the limit exceeds "
      "delta, no k meets delta *stably* ('stable' = never), but delta can still be met at such dips; the table gives "
      "the minimum over k and the k values (up to 200, and within the recalibration pool) at which delta is met. "
      "Empirical = resampled k-subsets of the pool.\n")
    w("| dataset | n_test | eps | delta | empirical first / stable k (resampled, grid) | BetaBin first / stable k | BetaBin limit as k->inf | BetaBin min P(fail) over k <= 200 (at k) | k meeting delta (<= 200) | within pool | Chebyshev m | Chebyshev m feasible in pool / in replay |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, D in DS.items():
        B = D["budget_rules"]
        for kk, v in B["by_eps_delta"].items():
            e = kk.split("_")[0][3:]; d = kk.split("_")[1][5:]
            emp = f"{nv(v['empirical_first_k_resampled'])} / {nv(v['empirical_stable_k_resampled'])}"
            bbr = f"{nv(v['betabinom_rule_first'])} / {nv(v['betabinom_rule_stable'])}"
            w(f"| {name} | {B['n_test_median']} | {e} | {d} | {emp} | {bbr} | "
              f"{v['betabinom_limit_P_fail_k_to_inf']:.3f}{'' if v['betabinom_stably_reachable'] else ' (> delta: not stably reachable)'} | "
              f"{v['betabinom_min_P_fail_k_upto_200']:.3f} ({v['betabinom_argmin_k_upto_200']}) | "
              f"{runs_str(v['betabinom_k_runs_meeting_delta_upto_200'], 4)} | {runs_str(v['betabinom_k_runs_meeting_delta_within_pool'], 4)} | "
              f"{v['chebyshev_m']} | "
              f"{'yes' if v['chebyshev_feasible_in_pool'] else 'no'} / {'yes' if v['chebyshev_feasible_in_T_replay'] else 'no'} |")
    w("")
    w("'first' values below the 'stable' value come from the saw-tooth: at sizes where the conformal order statistic "
      "is the sample maximum (k = 9..18 at alpha = 0.10 gives j = k) or just below it, the interval is conservative "
      "(E[coverage] = j/(k+1) well above 0.90), so the failure probability dips; it rises again when j/(k+1) falls "
      "back towards 0.90. A budget read off such a dip is fragile.\n")

    # ---- (d)
    w("## (d) alpha sensitivity (0.10 vs 0.20)\n")
    mk = next(iter(DS.values()))["alpha_sensitivity"]["min_k_finite_interval"]
    w(f"Minimum k for a finite interval: alpha 0.10 -> {mk['alpha_0.10']['checked_with_conformal_q']} "
      f"(ceil(1/alpha)-1 = {mk['alpha_0.10']['formula_ceil(1/alpha)-1']}); alpha 0.20 -> "
      f"{mk['alpha_0.20']['checked_with_conformal_q']} (formula {mk['alpha_0.20']['formula_ceil(1/alpha)-1']}).\n")
    w("| dataset | k | alpha 0.10: cov mean / p5 / nw median | alpha 0.20: cov mean / p5 / nw median | width ratio 0.20/0.10 (mean) |")
    w("|---|---|---|---|---|")
    for name, D in DS.items():
        T = D["alpha_sensitivity"]["tradeoff"]
        for key in D["sweep_keys"]:
            if key in ("0", "5", "9", "10", "15", "20", "30", str(D["headline_m"]), "full_pool") or key == "4":
                t = T[key]
                a1, a2 = t["alpha_0.10"], t["alpha_0.20"]
                s1 = "infinite" if a1["n_inf"] == 20 else f"{a1['cov_mean']:.3f} / {a1['cov_p5']:.2f} / {f2(a1['nw_median'])}"
                s2 = "infinite" if a2["n_inf"] == 20 else f"{a2['cov_mean']:.3f} / {a2['cov_p5']:.2f} / {f2(a2['nw_median'])}"
                wr = t["width_ratio_0.20_over_0.10"]
                w(f"| {name} | {t['k']}{' (pool)' if key == 'full_pool' else ''} | {s1} | {s2} | {f2(wr['mean']) if wr else 'n/a'} |")
    w("")

    # ---- (e)
    w("## (e) Retrospective replay over the whole target pool\n")
    w(f"Candidates = the whole target pool T. After n measurements (random = protocol order; novelty = decreasing "
      f"mean distance to the {NOV_K} nearest standardised training inputs), recalibrate on the n measured records and "
      f"evaluate on the |T| - n unmeasured ones (n <= |T| - {MIN_REM}; n < 9 gives an infinite interval and cannot "
      "qualify). C1 (task) = remainder coverage >= 0.90 in >= 80% of splits for every later n; C2 = the (b) criterion "
      "(5th percentile >= 0.80 and mean >= 0.88) for every later n. 'Every later n' runs up to a stopping horizon at "
      f"which the remainder is still >= max({STOP_MIN_REM}, {int(100*STOP_MIN_REM_FRAC)}% of |T|) records (on smaller "
      "remainders one miss moves coverage by more than 0.05). **This horizon is an analyst convention and the stopping "
      "points depend on it strongly** (see the horizon-sensitivity table below; the strict variant runs to a remainder of "
      f"{MIN_REM}). A stop must be verified over at least {STAY_MIN_WINDOW} further measurements before the horizon; "
      "otherwise a stop at the very end of the horizon would hold trivially over one or two points. Stopping points "
      "are *oracle* (they read remainder labels). 'unmeasured' = share of the target region that receives an interval "
      "instead of a measurement.\n")
    w("| dataset | |T| | order | C1 stop n (frac of T) | max share of splits >= 0.90 | C2 stop n (frac of T) | unmeasured at C2 | "
      "remainder cov at C2 (mean, p5) | nw median at C2 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for name, D in DS.items():
        R = D["replay"]
        for order in ("random", "novelty"):
            c1, c2 = R[order]["C1_task"], R[order]["C2_b_analogue"]
            s1 = "never" if c1["stays_from"] is None else f"{c1['stays_from']} ({c1['stays_from']/D['n_T']:.2f})"
            if c2["stays_from"] is None:
                s2, un, rc, nw = "never", "n/a", "n/a", "n/a"
            else:
                at = c2["at"]
                s2 = f"{c2['stays_from']} ({at['frac_of_T_measured']:.2f})"
                un = f"{at['frac_of_T_unmeasured']:.2f}"
                rc = f"{at['remainder_cov']['mean']:.3f}, {at['remainder_cov_p5']:.2f}"
                nw = f2(at["norm_width_median"])
            w(f"| {name} | {D['n_T']} | {order} | {s1} | {c1['max_frac_splits_ge_0.90_over_n>=9']:.2f} | {s2} | {un} | {rc} | {nw} |")
    w("")
    w("Horizon sensitivity of the oracle stopping points: 'stays from' n for C1 / C2 when 'every later n' runs up to a "
      "remainder of at least the given number of records (default = max(20, 20% of |T|), marked *; 10 = strict), "
      f"each stop verified over >= {STAY_MIN_WINDOW} further measurements ('-' = horizon too short to verify any stop). "
      "A stopping point that moves or disappears as the horizon is extended rests on the convention.\n")
    rms = sorted({int(rm) for D in DS.values() for o in ("random", "novelty")
                  for rm in D["replay"][o]["horizon_sensitivity_by_min_remainder"]})
    w("| dataset | order | " + " | ".join(f"rem >= {rm}" for rm in rms) + " | default (rem >= ...) |")
    w("|---|---|" + "---|" * (len(rms) + 1))
    for name, D in DS.items():
        R = D["replay"]
        for order in ("random", "novelty"):
            H = R[order]["horizon_sensitivity_by_min_remainder"]
            cells = []
            for rm in rms:
                v = H.get(str(rm))
                cells.append("-" if v is None else f"{nv(v['C1_task'])} / {nv(v['C2_b_analogue'])}"
                             + ("*" if rm == R["stopping_horizon_min_remainder"] else ""))
            w(f"| {name} | {order} | " + " | ".join(cells) + f" | {R['stopping_horizon_min_remainder']} |")
    w("")
    w("Composition straddle at the replay cut. The replay cuts between measured and unmeasured at record level; "
      "replicate records are contiguous in both orders, so at most one composition per cut can have records on both sides. "
      "Shown: remainder records sharing a composition with a measured record, mean over n >= 9 and splits (max), "
      "and the largest share of the remainder they ever make up.\n")
    w("| dataset | random: mean (max), max share | novelty: mean (max), max share |")
    w("|---|---|---|")
    for name, D in DS.items():
        R = D["replay"]
        cells = [f"{f2(R[o]['straddle']['mean_over_n9_to_nmax_and_splits'])} ({R[o]['straddle']['max']}), "
                 f"{R[o]['straddle']['max_share_of_remainder']:.3f}" for o in ("random", "novelty")]
        w(f"| {name} | " + " | ".join(cells) + " |")
    w("")
    w("C1 at the C1 stopping point (where reached): median normalised width the unmeasured candidates receive.\n")
    w("| dataset | order | C1 stop n | frac of T | remainder cov mean | share of splits >= 0.90 | nw median [2.5, 97.5] |")
    w("|---|---|---|---|---|---|---|")
    for name, D in DS.items():
        for order in ("random", "novelty"):
            at = D["replay"][order]["C1_task"]["at"]
            if at is None:
                w(f"| {name} | {order} | never | - | - | - | - |")
            else:
                nwv = at["norm_width"]
                w(f"| {name} | {order} | {at['n_measured']} | {at['frac_of_T_measured']:.2f} | {at['remainder_cov']['mean']:.3f} | "
                  f"{at['frac_splits_ge_0.90']:.2f} | {f2(at['norm_width_median'])} [{f2(nwv['lo'])}, {f2(nwv['hi'])}] |")
    w("")
    meds = [D["replay"]["random"]["theory_P_ge_0.90_exchangeable_median_n>=9"] for D in DS.values()]
    w("Why C1 is not reached under random order: split conformal guarantees the *mean* coverage, not coverage >= 0.90 "
      f"in 80% of realisations. Under exchangeability P(remainder coverage >= 0.90) has a median over n of "
      f"{min(meds):.2f}-{max(meds):.2f} and "
      "exceeds 0.8 only at a few conservative saw-tooth sizes where the conformal order statistic is the sample "
      "maximum (n <= 18), so 'for every later n' cannot hold:\n")
    w("| dataset | theory P(>= 0.90), random order: median / max over n (at n) | observed max share of splits (random) | observed max share (novelty) | Spearman(novelty, |residual|) |")
    w("|---|---|---|---|---|")
    for name, D in DS.items():
        R = D["replay"]
        sp = R["novelty_spearman_with_abs_residual"]
        w(f"| {name} | {R['random']['theory_P_ge_0.90_exchangeable_median_n>=9']:.2f} / "
          f"{R['random']['theory_P_ge_0.90_exchangeable_max_n>=9']:.2f} ({R['random']['theory_P_ge_0.90_argmax_n']}) | "
          f"{R['random']['C1_task']['max_frac_splits_ge_0.90_over_n>=9']:.2f} | "
          f"{R['novelty']['C1_task']['max_frac_splits_ge_0.90_over_n>=9']:.2f} | {sp['mean']:.2f} [{sp['lo']:.2f}, {sp['hi']:.2f}] |")
    w("")
    w("Label-free, pre-registered stopping rules (protocol-compliant; for novelty order the exchangeability "
      "assumption behind the Beta rule does not hold):\n")
    w("| dataset | rule (n) | order | frac of T measured | remainder cov mean [2.5, 97.5] | p5 | share >= 0.90 | nw median |")
    w("|---|---|---|---|---|---|---|---|")
    for name, D in DS.items():
        for rname in RULE_N:
            for order in ("random", "novelty"):
                v = D["replay"][order]["preregistered_stops"][rname]
                if v is None or v.get("infeasible"):
                    w(f"| {name} | {rname} ({v['n_measured'] if v else '-'}) | {order} | infeasible (> |T| - {MIN_REM}) | - | - | - | - |")
                    continue
                cv = v["remainder_cov"]
                w(f"| {name} | {rname} ({v['n_measured']}) | {order} | {v['frac_of_T_measured']:.2f} | {cv['mean']:.3f} "
                  f"[{cv['lo']:.2f}, {cv['hi']:.2f}] | {v['remainder_cov_p5']:.2f} | {v['frac_splits_ge_0.90']:.2f} | "
                  f"{f2(v['norm_width_median'])} |")
    w("")
    w("Novelty vs random order at the same n (remainder coverage mean and median normalised width):\n")
    w("| dataset | n | random cov / nw | novelty cov / nw | novelty/random width (mean) |")
    w("|---|---|---|---|---|")
    for name, D in DS.items():
        for n, v in D["replay"]["novelty_vs_random_same_n"].items():
            w(f"| {name} | {n} | {v['random_cov']['mean']:.3f} / {f2(v['random_nw_median'])} | {v['novelty_cov']['mean']:.3f} / "
              f"{f2(v['novelty_nw_median'])} | {f2(v['novelty_over_random_width']['mean'])} |")
    w("")

    # ---- claims
    w("## Re-computing the submitted '65% and 90% summed across six cases'\n")
    S0 = CC["submitted_as_coded"]; P = CC["protocol"]
    w(f"* Total target-region size across the six cases: {CC['total_T']} records; the polymer contributes "
      f"{100*P['polymer_share_of_total_T']:.0f}% of it, so any summed saving is dominated by one dataset.")
    w(f"* As coded in demo/exp4 the 'rule' budget is min(140, max(3, |T|-15)), capped again at |T|-5: "
      f"{S0['rule_m_as_coded']}. It equals the Chebyshev m = {CC['chebyshev_m']} only where |T| >= 155; it was "
      f"truncated in {len(S0['rule_truncated_in'])}/6 cases ({', '.join(S0['rule_truncated_in'])}). Summed saving as "
      f"coded: rule {S0['rule_saved_pct_summed']:.1f}%, heuristic {S0['heuristic_saved_pct_summed']:.1f}%.")
    w(f"* Under the protocol the recalibration draws must come from the recalibration pool (half of T). The Chebyshev "
      f"m = {CC['chebyshev_m']} fits inside the pool only for "
      f"{[n for n, v in P['chebyshev_feasible_in_recal_pool'].items() if v]} and inside the whole-T replay only for "
      f"{[n for n, v in P['chebyshev_feasible_in_replay'].items() if v]}; the '65%' therefore does not describe the "
      f"rule. Capping the rule at the pool size gives {P['rule_capped_at_pool_saved_pct_summed']:.1f}% summed, but "
      f"that is again not the rule.")
    w(f"* The heuristic's 'saving' is arithmetic in |T| (1 - sum(m)/sum|T|) = {P['heuristic_saved_pct_summed']:.1f}% "
      f"summed, {P['heuristic_saved_pct_macro_mean']:.1f}% as a per-dataset mean "
      f"({', '.join(f'{k} {v:.0f}%' for k, v in P['heuristic_saved_pct_per_dataset'].items())}); "
      f"{P['heuristic_saved_pct_summed_without_polymer']:.1f}% summed without the polymer. It is not an experimental "
      "result; what is empirical is whether coverage holds at that m:")
    for n, v in P["heuristic_ttest_coverage"].items():
        h = P["heuristic_replay_remainder_random"][n]
        w(f"  * {n}: at m = {DS[n]['headline_m']} ({100*h['frac_of_T_measured']:.0f}% of |T|). Target-test half: mean "
          f"{v['mean']:.3f}, 2.5% {v['lo']:.2f}, min {v['min']:.2f}, share of splits >= 0.90 = {v['frac_splits_ge_0.90']:.2f}. "
          f"Replay remainder (random order, the {100*h['frac_of_T_unmeasured']:.0f}% unmeasured): mean "
          f"{h['remainder_cov']['mean']:.3f}, min {h['remainder_cov_min']:.2f}, share >= 0.90 = {h['frac_splits_ge_0.90']:.2f}, "
          f"median nw {f2(h['norm_width_median'])}.")
    w("* Two readings of 'stays at or above the nominal 0.90 throughout (0.91 to 0.97)': as a mean over re-partitions "
      "(how the original numbers were computed) it essentially holds under the protocol (see the means above; the "
      "original values were partly inflated by the one-rank-too-high quantile); per realisation it does not hold "
      "(see the minima and shares above). Split conformal promises only the former.")
    w(f"* The exact Beta rule for (eps 0.08, delta 0.10) is n = {P['exact_beta_rule_n']} (stable); it fits inside the "
      f"recalibration pool for {[n for n, v in P['exact_beta_feasible_in_recal_pool'].items() if v]}.")
    for order in ("random", "novelty"):
        for crit in ("C1_task", "C2_b_analogue"):
            v = P[f"replay_{order}_{crit}"]
            s = (f"{v['unmeasured_pct_summed']:.1f}% summed, {v['unmeasured_pct_macro_mean']:.1f}% per-dataset mean"
                 if "unmeasured_pct_summed" in v else f"reached in {v['n_datasets_reached']}/6 datasets only")
            w(f"* Replay (oracle), {order} order, {crit}: stop n (default horizon) = {v['stop_n']}; unmeasured share: {s}. "
              f"Strict horizon (remainder >= {MIN_REM}): {v['stop_n_strict_min_remainder_10']}.")
    w("")
    w("## Original-code quantile check (demo/exp4, demo/exp6)\n")
    w("The demo scripts compute the conformal quantile as np.quantile(r, min(ceil((n+1)(1-a))/n, 1), method='higher'). "
      "Order statistic returned vs the finite-sample conformal one (common.conformal_q):\n")
    w("| n | original rank | conformal rank | E[coverage] original | E[coverage] conformal |")
    w("|---|---|---|---|---|")
    for r in TH["original_cq_check"]:
        w(f"| {r['n']} | {r['orig_rank']} | {r['conformal_rank']} | {r['orig_expected_cov']:.3f} | {r['conformal_expected_cov']:.3f} |")
    w("")
    extra = res.get("interpretation")
    if extra:
        w("## Interpretation\n")
        for s in extra:
            w(f"* {s}")
        w("")
    w("## Fix round: independent-verifier issues and how they were addressed\n")
    th20 = {(r_["eps"], r_["delta"]): r_ for r_ in TH["alpha_0.20"]}
    ch = th20.get((0.08, 0.10)); ch5 = th20.get((0.08, 0.05))
    for s in [
        "1. *'Floor' of the finite-test failure rate* (adopted). The Binomial(n_test, 0.9) tail is the limit as k -> "
        "infinity, not a floor; at conservative saw-tooth sizes the exact Beta-Binomial failure probability falls below "
        "it. The field `betabinom_reachable` was renamed `betabinom_stably_reachable` (= a stable k exists), and the JSON "
        "and the (c) table now give the minimum failure probability over k, its argmin, and the k values (up to 200 and "
        "within the recalibration pool) at which delta is met. Wording in (c) and in the interpretation was corrected.",
        f"2. *Record-level inner splits in the protocol-compliant (b) selection* (adopted). Inner splits are now grouped "
        f"(cal = first k records of a grouped permutation of the pool, which is the deployed first-k rule; holdout = "
        f"remaining whole compositions). B raised from {B_SUB} to {B_SEL}, and {R_SEL} independent inner-RNG replicates "
        "report the Monte Carlo spread of the certification counts, which are stated as approximate. The record-level "
        "version is kept as a labelled sensitivity.",
        "3. *Replay horizon as an undisclosed analyst degree of freedom* (adopted). A horizon-sensitivity table covers "
        "every dataset, order and criterion. The interpretation lists every stopping point that changes between the "
        "default and strict horizons, and states that random-order C2 is reached only under the default convention. "
        f"Building that table exposed a second artefact: without a minimum verification window, a 'stays from' stop at the "
        f"very end of a horizon holds trivially (e.g. polymer_Tg random order 'reached' at n = 953 of 953). Every stop "
        f"must now hold over >= {STAY_MIN_WINDOW} further measurements. The default-horizon stops were far from their "
        "horizon and did not change.",
        "4. *Exchangeability overreach* (adopted). The agreement with Beta-Binomial theory is now described as guaranteed "
        "by the random pool/test partition, not as a property of the materials; the width cost is material-dependent, and "
        "prospective non-random acquisition is not covered.",
        "5. *'Throughout' wording* (adopted). Both readings are stated. The mean reading essentially holds under the "
        "protocol (on the replay remainder and on target-test). The per-realisation reading does not, and split conformal "
        "never promised it.",
        "6. *Float ceil in the Chebyshev / Cantelli budgets* (adopted). Exact rational arithmetic (fractions.Fraction). At "
        "alpha = 0.20, eps = 0.08 the Chebyshev budgets are now "
        + (f"{ch5['chebyshev_m']} (delta 0.05) and {ch['chebyshev_m']} (delta 0.10); Cantelli {ch5['cantelli_m']} and "
           f"{ch['cantelli_m']}" if ch and ch5 else "corrected")
        + ". The alpha = 0.10 values were unaffected (140 etc.).",
        "7. *Unsupported causal explanation for 'no stable k'* (adopted). Replaced by: the oracle labels are noise-level for "
        "small regions (the 20-split p5 is essentially the second-lowest split; near misses listed), and the reliable "
        "reference is the Beta-Binomial theory at every integer k for the actual test size (new column in (b)).",
        "8. *'Median width' label in (d)* (adopted). The text now says 'mean over splits of the per-split width ratio'.",
        "9. *Proposed manuscript sentence mixing populations* (adopted). The replacement sentence now uses only the "
        "random-order replay at the pre-registered headline m. Coverage, worst split and width all refer to the unmeasured "
        "remainder, which is the population the '67-97% unmeasured' statement is about.",
        "10. *Composition straddle at the replay cut* (adopted as disclosure). Straddle counts are computed in the script "
        "and tabulated in (e); they are negligible.",
    ]:
        w(s)
    w("\n## Verifier issues not adopted\n")
    w("None. All ten issues were judged valid and addressed as described above.\n")
    open(os.path.join(RES, f"{TAG}.md"), "w", encoding="utf-8").write("\n".join(L))


# ================================================================= main
def clean(o):
    """Replace non-finite floats (infinite intervals, undefined Beta quantiles for k < 9) by None so
    the JSON is strict; the md and 'n_infinite_interval' fields state where intervals were infinite."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def main(n_jobs=6):
    t0 = time.time()
    res = {"meta": {"script": "revision/code/exp_R8_budget.py", "model": "RF (common.make_model, n_jobs set to %d)" % RF_JOBS,
                    "alpha": ALPHA, "alphas": ALPHAS, "seeds": SEEDS, "kgrid": KGRID, "B_sub": B_SUB,
                    "B_sel": B_SEL, "R_sel": R_SEL, "horizon_grid_min_remainder": HZ_GRID,
                    "stay_min_window": STAY_MIN_WINDOW,
                    "default_horizon": f"remainder >= max({STOP_MIN_REM}, {STOP_MIN_REM_FRAC} |T|)",
                    "min_remainder": MIN_REM, "novelty_k": NOV_K, "rule_n": RULE_N,
                    "criterion_b": [CRIT_P5, CRIT_MEAN], "criterion_C1": [C1_COV, C1_FRAC],
                    "notes": ["splits resample one finite dataset; 2.5-97.5 percentile intervals describe split-to-split variability",
                              "k < ceil(1/alpha)-1 gives an infinite interval (coverage 1) and is reported as such",
                              "oracle quantities (min k, replay stopping points) read target-test / remainder labels and are descriptive only",
                              "resampled supplement draws random k-subsets of the recalibration pool only; target-test is unchanged"]},
           "theory": theory_tables(), "datasets": {}}
    for ds in design_shift_datasets():
        t1 = time.time()
        rows = Parallel(n_jobs=n_jobs)(delayed(one_split)(ds, s) for s in SEEDS)
        D = aggregate(ds, rows)
        res["datasets"][ds.name] = D
        M = D["min_k"]; R = D["replay"]
        print(f"{ds.name:11s} |T|={D['n_T']:4d} m={D['headline_m']:2d} cov@m={D['ksweep'][akey(ALPHA)][str(D['headline_m'])]['coverage']['mean']:.3f}"
              f" mink(first/stable)={M['protocol_first_k']['first']}/{M['protocol_first_k']['stable']}"
              f" poolsel={M['pool_internal_selection']['n_splits_certified']}/20"
              f" C1 rand/nov={R['random']['C1_task']['stays_from']}/{R['novelty']['C1_task']['stays_from']}"
              f" C2 rand/nov={R['random']['C2_b_analogue']['stays_from']}/{R['novelty']['C2_b_analogue']['stays_from']}"
              f" ({time.time()-t1:.0f}s)", flush=True)
    res["claims_check"] = claims_check(res["datasets"])
    res["interpretation"] = interpret(res)
    res["meta"]["runtime_min"] = (time.time() - t0) / 60
    res = clean(res)
    dump(res, f"{TAG}.json")
    write_md(res)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--md-only":
        res = json.load(open(os.path.join(RES, f"{TAG}.json")))
        res["interpretation"] = interpret(res)
        res = clean(res)
        dump(res, f"{TAG}.json")
        write_md(res)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
