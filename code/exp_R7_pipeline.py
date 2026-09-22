"""R7: the trust layer as ONE decision procedure, evaluated end-to-end (Reviewer 2,
comments 2, 4 and 7).

For each design-shift dataset (common.design_shift_datasets) and split seed 0..19
(common.design_split, grouped 60/20/20 source split, target pool halves):

  candidate pool  = source-test U target-test (the designer does not know which is which;
                    the mix is reported);
  specification   = acceptable iff y >= tau. HEADLINE tau = median of y over the split's
                    RECALIBRATION POOL (PROTOCOL.md: anything that uses target labels is
                    taken from the recalibration pool only; no target-test label enters it).
                    Sensitivity: rpool_q75 (75th percentile of the recalibration pool);
                    rpool_holdout_median (median of the recalibration-pool records NOT used
                    for recalibration, so tau is also independent of the target-stratum
                    calibration labels); tgt_median / tgt_q75 (the task's literal definition,
                    median / 75th percentile of the WHOLE target pool; these use target-test
                    labels, a protocol deviation, and are reported only as sensitivity).
                    tau is never used for fitting, calibration, flagging or recalibration;
  models          = RF (reference) and HistGB (common.make_model, fixed hyperparameters).

Arms (all built from one TrustLayer object; A1-A5 are ablations of the same procedure):
  A0 point prediction    accept iff mu(x) >= tau, otherwise reject; no measurement.
  A1 conformal only      source-stratum interval for every candidate, no shift check.
  A2 layer, no target    shift check; flagged -> not_certified (-> measured).
  A3 full layer          shift check + recalibration budget m = headline_m spent on the
                         first m records of the recalibration POOL (never on candidates)
                         -> target stratum; flagged candidates decided with target-stratum
                         intervals; undecided -> measure.
  A4 layer + selection   A3, but 'accept' = conformal selection (Jin & Candes 2023) with BH
                         at q = 0.10 (target stratum for flagged, source for in-domain),
                         one BH over all candidates (FDR of the whole accepted set);
                         not selected -> A3's reject stays reject, everything else measured.
  A4s                    as A4 but BH run separately inside each stratum (FDR within the
                         flagged and within the in-domain accepted set; added to show the
                         scope of the guarantee, not requested in the task).
  A5 negative control    conformal selection at q = 0.10 with SOURCE calibration for every
                         candidate; not selected -> A1's reject stays, else measured.
  D3, D4 (diagnostics, not deployable): A3 and A4 with the shift-check flag replaced by
                         the TRUE target membership (a perfect shift check). They separate
                         the error due to the shift check's power from the error that
                         remains even with perfect routing. They use membership labels,
                         never property labels.

Per arm and split: n accepted, FDP among accepted (mean over splits = FDR), power,
power after measurement, measurements used (measure + not_certified + budget m for every
arm that spends it: A3, A4, A4s, D3, D4), false-reject proportion, unconditional
false-accept probability (false accepts / pool), interval miscoverage; each also by
in-domain / flagged, true source / true target, and the two routing errors (target not
flagged, source flagged). Additionally: pooled FDP over the 20 splits (total false accepts
/ total accepts; informative when a subgroup has < 1 accept per split), the A3-minus-A2
contrast (what target recalibration adds), and the p-value granularity of per-stratum BH.
Shift check: AUROC of the novelty p-value for true target membership, share of target /
source candidates flagged. Outputs results/R7_pipeline.json and results/R7_pipeline.md.
"""
import math
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from common import (ALPHA, SEEDS, design_shift_datasets, design_split, headline_m, make_model,
                    summarize, dump, conformal_q, iqr, RES)
import trustlayer as TLm
from trustlayer import TrustLayer, ACCEPT, REJECT, MEASURE, NOT_CERTIFIED

MODELS = ["RF", "HistGB"]
Q_FDR = 0.10
NOVELTY_K, NOVELTY_LEVEL = 10, 0.05
THREADS = 2                                   # per worker (6 workers -> 12 cores)
ARMS = ["A0_point", "A1_conformal", "A2_layer_no_target", "A3_layer", "A4_layer_fdr",
        "A4s_layer_fdr_by_stratum", "A5_selection_source_only",
        "D3_layer_oracle_routing", "D4_layer_fdr_oracle_routing"]
SUBSETS = ["all", "in_domain", "flagged", "true_source", "true_target", "target_unflagged",
           "source_flagged"]
TAUS = ["rpool_median", "rpool_q75", "rpool_holdout_median", "tgt_median", "tgt_q75"]
HEAD, HEAD_Q75 = "rpool_median", "rpool_q75"
TAU_DOC = {
    "rpool_median": "median of y over the split's recalibration pool (HEADLINE; protocol-compliant: "
                    "no target-test label; the pool contains the m recalibration records)",
    "rpool_q75": "75th percentile of the recalibration pool (protocol-compliant sensitivity)",
    "rpool_holdout_median": "median of the recalibration-pool records not used for recalibration "
                            "(independent of both target-stratum calibration labels and candidates)",
    "tgt_median": "median of the whole target pool (task's literal definition; uses target-test labels "
                  "= protocol deviation; sensitivity only)",
    "tgt_q75": "75th percentile of the whole target pool (task's literal sensitivity; uses target-test labels)"}
BUDGET_ARMS = {"A3_layer", "A4_layer_fdr", "A4s_layer_fdr_by_stratum", "D3_layer_oracle_routing",
               "D4_layer_fdr_oracle_routing"}


def arm_metrics(dec, y, tau, lo, hi, masks, budget):
    ok = y >= tau
    acc, rej = dec == ACCEPT, dec == REJECT
    mea, nc = dec == MEASURE, dec == NOT_CERTIFIED
    miss = None if lo is None else ~((y >= lo) & (y <= hi))
    out = {}
    for name, S in masks.items():
        n = int(S.sum())
        if n == 0:
            out[name] = None
            continue
        na = int((acc & S).sum()); fa = int((acc & ~ok & S).sum()); ta = na - fa
        nok = int((ok & S).sum()); nr = int((rej & S).sum()); fr = int((rej & ok & S).sum())
        nm = int((mea & S).sum()); nn = int((nc & S).sum())
        meas_ok = int(((mea | nc) & ok & S).sum())
        out[name] = {
            "n": n, "n_acceptable": nok, "n_accept": na, "n_reject": nr, "n_measure": nm,
            "n_not_certified": nn, "false_accepts": fa, "false_rejects": fr,
            "accept_rate": na / n,
            "fdp": fa / na if na else 0.0,                 # 0 when nothing accepted -> mean = FDR
            "fdp_if_any": fa / na if na else None,
            "any_accepted": float(na > 0),
            "power": ta / nok if nok else None,
            "power_after_measurement": (ta + meas_ok) / nok if nok else None,
            "measurements": nm + nn + (budget if name == "all" else 0),
            "measure_share": (nm + nn) / n,
            "false_reject_prop": fr / nr if nr else None,  # among rejected
            "false_reject_rate_of_acceptable": fr / nok if nok else None,
            "uncond_false_accept": fa / n,
            "miscoverage": float(miss[S].mean()) if miss is not None else None,
        }
    return out


def one_split(ds, seed, model_name):
    with threadpool_limits(limits=THREADS):
        sp = design_split(ds, seed)
        X, y = ds.X, ds.y
        mh = headline_m(ds)
        model = make_model(model_name, seed)
        if "n_jobs" in model.get_params():
            model.set_params(n_jobs=THREADS)
        tl = TrustLayer(model, alpha=ALPHA, novelty_k=NOVELTY_K, novelty_level=NOVELTY_LEVEL)
        tl.fit(X[sp["train"]], y[sp["train"]])
        tl.calibrate(X[sp["cal"]], y[sp["cal"]])

        cand = np.r_[sp["stest"], sp["ttest"]]
        is_t = np.r_[np.zeros(len(sp["stest"]), bool), np.ones(len(sp["ttest"]), bool)]
        Xc, yc = X[cand], y[cand]
        n = len(cand)
        zero = np.zeros(n, bool)
        mu = tl.predict(Xc)
        pval, flag = tl.shift_check(Xc)
        _, flag_bh = tl.shift_check(Xc, multiplicity="bh")
        nov = tl.novelty_score(Xc)
        masks = {"all": np.ones(n, bool), "in_domain": ~flag, "flagged": flag,
                 "true_source": ~is_t, "true_target": is_t,
                 "target_unflagged": is_t & ~flag, "source_flagged": ~is_t & flag}
        rp_all = sp["rpool"]
        rp, rp_rest = rp_all[:mh], rp_all[mh:]
        assert len(rp_rest) >= 5, (ds.name, len(rp_rest))
        taus = {"rpool_median": float(np.median(y[rp_all])),
                "rpool_q75": float(np.percentile(y[rp_all], 75)),
                "rpool_holdout_median": float(np.median(y[rp_rest])),
                "tgt_median": float(np.median(y[ds.tgt])),
                "tgt_q75": float(np.percentile(y[ds.tgt], 75))}

        # ---- state 1: source calibration only (A1, A2, A5)
        lo1, hi1 = tl.interval(Xc, zero)
        lo2, hi2 = tl.interval(Xc, flag)
        pre = {tn: {"A1": tl.decide(Xc, t, flags=zero), "A2": tl.decide(Xc, t, flags=flag),
                    "A5sel": tl.select_fdr(Xc, t, Q_FDR, flags=zero)} for tn, t in taus.items()}
        q_src = tl.state()["source"]["q"]

        # ---- state 2: + budgeted recalibration from the recalibration POOL (A3, A4)
        assert len(np.intersect1d(rp, cand)) == 0
        tl.recalibrate(X[rp], y[rp])
        st = tl.state()
        n_t = st["target"]["n"]
        rp_flag = tl.shift_check(X[rp])[1]
        lo3, hi3 = tl.interval(Xc, flag)
        lo_o, hi_o = tl.interval(Xc, is_t)          # diagnostic: perfect shift check

        out = {"seed": seed, "model": model_name, "m": mh,
               "sizes": {k: int(len(v)) for k, v in sp.items()},
               "mix": {"n_pool": n, "n_source_cand": int((~is_t).sum()), "n_target_cand": int(is_t.sum()),
                       "target_share": float(is_t.mean())},
               "shift": {"auroc_pvalue": float(roc_auc_score(is_t, -pval)),
                         "auroc_novelty_score": float(roc_auc_score(is_t, nov)),
                         "target_flagged": float(flag[is_t].mean()),
                         "source_flagged": float(flag[~is_t].mean()),
                         "target_flagged_bh": float(flag_bh[is_t].mean()),
                         "source_flagged_bh": float(flag_bh[~is_t].mean()),
                         "flagged_share_of_pool": float(flag.mean()),
                         "target_share_among_flagged": float(is_t[flag].mean()) if flag.any() else None,
                         "recal_records_flagged": float(rp_flag.mean())},
               "state": {"n_source_cal": st["source"]["n"], "n_target_records": n_t,
                         "q_source": q_src, "q_target": st["target"]["q"],
                         "q_source_over_tgt_iqr": q_src / iqr(y[ds.tgt]),
                         "q_target_over_tgt_iqr": st["target"]["q"] / iqr(y[ds.tgt]),
                         "q_matches_common": bool(np.isclose(q_src, conformal_q(np.abs(y[sp["cal"]] - tl.predict(X[sp["cal"]])), ALPHA)))},
               "rmse": {"source_cand": float(np.sqrt(np.mean((yc[~is_t] - mu[~is_t]) ** 2))),
                        "target_cand": float(np.sqrt(np.mean((yc[is_t] - mu[is_t]) ** 2)))},
               "tau": {}}

        for tn, t in taus.items():
            ok = yc >= t
            d0 = np.where(mu >= t, ACCEPT, REJECT)
            d1, d2 = pre[tn]["A1"], pre[tn]["A2"]
            d3 = tl.decide(Xc, t, flags=flag)
            sel4, p4 = tl.select_fdr(Xc, t, Q_FDR, flags=flag, return_pvalues=True)
            d4 = np.where(sel4, ACCEPT, np.where(d3 == ACCEPT, MEASURE, d3))
            sel4s = tl.select_fdr(Xc, t, Q_FDR, flags=flag, scope="stratum")
            d4s = np.where(sel4s, ACCEPT, np.where(d3 == ACCEPT, MEASURE, d3))
            d5 = np.where(pre[tn]["A5sel"], ACCEPT, np.where(d1 == ACCEPT, MEASURE, d1))
            do3 = tl.decide(Xc, t, flags=is_t)
            selo4 = tl.select_fdr(Xc, t, Q_FDR, flags=is_t)
            do4 = np.where(selo4, ACCEPT, np.where(do3 == ACCEPT, MEASURE, do3))
            spec = {"A0_point": (d0, None, None), "A1_conformal": (d1, lo1, hi1),
                    "A2_layer_no_target": (d2, lo2, hi2), "A3_layer": (d3, lo3, hi3),
                    "A4_layer_fdr": (d4, lo3, hi3), "A4s_layer_fdr_by_stratum": (d4s, lo3, hi3),
                    "A5_selection_source_only": (d5, lo1, hi1),
                    "D3_layer_oracle_routing": (do3, lo_o, hi_o), "D4_layer_fdr_oracle_routing": (do4, lo_o, hi_o)}
            arms = {a: arm_metrics(d, yc, t, lo, hi, masks, mh if a in BUDGET_ARMS else 0)
                    for a, (d, lo, hi) in spec.items()}
            # Code-consistency check (NOT empirical support for G1): an interval-accept of a
            # violating candidate implies y < tau <= lo, i.e. miscoverage, so this can only fail
            # through a bug in decide()/interval().
            for a in ["A1_conformal", "A2_layer_no_target", "A3_layer", "D3_layer_oracle_routing"]:
                for s_ in masks:
                    r = arms[a][s_]
                    if r is not None:
                        assert r["uncond_false_accept"] <= r["miscoverage"] + 1e-12, (a, s_)

            # ---- what target recalibration adds: A3 minus A2. In-domain candidates get the same
            # source interval in both arms and flagged ones are not_certified in A2, so A2's accept
            # and reject sets are subsets of A3's (asserted): recalibration can only ADD decisions.
            assert np.all((d2 == ACCEPT) <= (d3 == ACCEPT)) and np.all((d2 == REJECT) <= (d3 == REJECT))
            ex_a = (d3 == ACCEPT) & (d2 != ACCEPT)
            ex_r = (d3 == REJECT) & (d2 != REJECT)
            a3m2 = {"n_extra_accept": int(ex_a.sum()), "extra_false_accepts": int((ex_a & ~ok).sum()),
                    "n_extra_accept_true_target": int((ex_a & is_t).sum()),
                    "extra_false_accepts_true_target": int((ex_a & ~ok & is_t).sum()),
                    "any_extra_accept_true_target": float((ex_a & is_t).any()),
                    "n_extra_reject": int(ex_r.sum()), "extra_false_rejects": int((ex_r & ok).sum()),
                    "extra_measurements": arms["A3_layer"]["all"]["measurements"] - arms["A2_layer_no_target"]["all"]["measurements"],
                    "target_false_accepts_A2": arms["A2_layer_no_target"]["true_target"]["false_accepts"],
                    "target_false_accepts_A3": arms["A3_layer"]["true_target"]["false_accepts"]}

            # ---- p-value granularity of BH inside the flagged (target) stratum
            fl = flag.copy()
            n_f = int(fl.sum())
            p_min = 1.0 / (n_t + 1.0)
            k_needed = int(math.ceil(n_f * p_min / Q_FDR - 1e-9)) if n_f else None
            n_sel_f = int(sel4s[fl].sum())
            if n_sel_f:          # any non-empty BH selection in the stratum has >= k_needed members
                assert n_sel_f >= k_needed, (ds.name, seed, n_sel_f, k_needed)
            gran = {"n_flagged": n_f, "p_min": p_min, "k_needed": k_needed,
                    "k_needed_share_of_flagged": (k_needed / n_f) if n_f else None,
                    "n_flagged_at_p_min": int((p4[fl] <= p_min * (1 + 1e-9)).sum()),
                    "n_selected_flagged_A4s": n_sel_f, "any_selected_flagged_A4s": float(n_sel_f > 0),
                    "n_selected_flagged_A4": int(sel4[fl].sum()),
                    "bh_threshold_joint_A4": (Q_FDR * int(sel4.sum()) / n) if sel4.any() else None,
                    "bh_threshold_joint_D4": (Q_FDR * int(selo4.sum()) / n) if selo4.any() else None}

            out["tau"][tn] = {"value": t,
                              "acceptable_share": {"source_cand": float((yc[~is_t] >= t).mean()),
                                                   "target_cand": float((yc[is_t] >= t).mean())},
                              "arms": arms, "a3_minus_a2": a3m2, "a4s_granularity": gran}
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


METRICS = ["n", "n_acceptable", "n_accept", "n_reject", "n_measure", "n_not_certified", "false_accepts",
           "false_rejects", "accept_rate", "fdp", "fdp_if_any", "any_accepted", "power",
           "power_after_measurement", "measurements", "measure_share", "false_reject_prop",
           "false_reject_rate_of_acceptable", "uncond_false_accept", "miscoverage"]


def _pooled(rows, tn, a, s_):
    na = fa = nr = fr = 0
    for r in rows:
        v = r["tau"][tn]["arms"][a][s_]
        if v is not None:
            na += v["n_accept"]; fa += v["false_accepts"]; nr += v["n_reject"]; fr += v["false_rejects"]
    return {"n_accept_total": na, "false_accepts_total": fa, "fdp_pooled": fa / na if na else None,
            "n_reject_total": nr, "false_rejects_total": fr, "false_reject_prop_pooled": fr / nr if nr else None}


def summarise_model(rows):
    S = {"m": rows[0]["m"], "sizes_seed0": rows[0]["sizes"],
         "mix": {k: agg(rows, ["mix", k]) for k in rows[0]["mix"]},
         "shift_check": {k: agg(rows, ["shift", k]) for k in rows[0]["shift"]},
         "state": {k: agg(rows, ["state", k]) for k in rows[0]["state"] if k != "q_matches_common"},
         "q_matches_common_all_splits": bool(all(r["state"]["q_matches_common"] for r in rows)),
         "rmse": {k: agg(rows, ["rmse", k]) for k in rows[0]["rmse"]}, "tau": {}}
    for tn in TAUS:
        T = {"definition": TAU_DOC[tn], "value": agg(rows, ["tau", tn, "value"]),
             "acceptable_share": {k: agg(rows, ["tau", tn, "acceptable_share", k]) for k in ["source_cand", "target_cand"]},
             "arms": {}, "pooled": {}}
        for a in ARMS:
            T["arms"][a] = {s_: {mt: agg(rows, ["tau", tn, "arms", a, s_, mt]) for mt in METRICS} for s_ in SUBSETS}
            T["pooled"][a] = {s_: _pooled(rows, tn, a, s_) for s_ in SUBSETS}
        X = {k: agg(rows, ["tau", tn, "a3_minus_a2", k]) for k in rows[0]["tau"][tn]["a3_minus_a2"]}
        tot = lambda k: sum(r["tau"][tn]["a3_minus_a2"][k] for r in rows)
        X["pooled"] = {"n_extra_accept_total": tot("n_extra_accept"), "extra_false_accepts_total": tot("extra_false_accepts"),
                       "fdp": tot("extra_false_accepts") / tot("n_extra_accept") if tot("n_extra_accept") else None,
                       "n_extra_accept_true_target_total": tot("n_extra_accept_true_target"),
                       "extra_false_accepts_true_target_total": tot("extra_false_accepts_true_target"),
                       "fdp_true_target": (tot("extra_false_accepts_true_target") / tot("n_extra_accept_true_target")
                                           if tot("n_extra_accept_true_target") else None),
                       "n_extra_reject_total": tot("n_extra_reject"), "extra_false_rejects_total": tot("extra_false_rejects")}
        T["a3_minus_a2"] = X
        T["a4s_granularity"] = {k: agg(rows, ["tau", tn, "a4s_granularity", k]) for k in rows[0]["tau"][tn]["a4s_granularity"]}
        S["tau"][tn] = T
    S["per_split_headline"] = [
        {"seed": r["seed"], "tau_name": HEAD, "tau": r["tau"][HEAD]["value"],
         "auroc": r["shift"]["auroc_pvalue"], "target_flagged": r["shift"]["target_flagged"],
         "a3_minus_a2": r["tau"][HEAD]["a3_minus_a2"],
         **{a: {s_: ({k: r["tau"][HEAD]["arms"][a][s_][k] for k in
                      ["n_accept", "false_accepts", "fdp", "power", "measurements", "uncond_false_accept", "miscoverage"]}
                     if r["tau"][HEAD]["arms"][a][s_] is not None else None)
                for s_ in ["all", "true_target", "flagged", "target_unflagged"]} for a in ARMS}} for r in rows]
    return S


# ------------------------------------------------------------------ markdown helpers
def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def ci(s, d=3):
    if s is None or s.get("mean") is None:
        return "n/a"
    return f"{s['mean']:.{d}f} [{s['lo']:.{d}f}, {s['hi']:.{d}f}]"


def pc(x, d=0):
    return "n/a" if x is None else f"{100 * x:.{d}f} %"


SHORT = {"A0_point": "A0 point", "A1_conformal": "A1 conformal", "A2_layer_no_target": "A2 layer, no target",
         "A3_layer": "A3 layer (m)", "A4_layer_fdr": "A4 layer + FDR sel. (joint BH)",
         "A4s_layer_fdr_by_stratum": "A4s layer + FDR sel. (BH per stratum)", "A5_selection_source_only": "A5 sel., source only",
         "D3_layer_oracle_routing": "D3 = A3, oracle routing (diag.)",
         "D4_layer_fdr_oracle_routing": "D4 = A4, oracle routing (diag.)"}

THEORY = r"""
## What is and is not guaranteed (theory, stated once)

Let C(x) = [mu(x) - q, mu(x) + q] be the split-conformal interval of a stratum and let a
candidate be 'accepted' by the interval rule iff the whole interval satisfies the
specification (y >= tau means lo >= tau).

* **G1, coverage.** If a candidate is exchangeable with the calibration records of the
  stratum used for it, P(Y in C(X)) >= 1 - alpha (marginal over calibration and candidate).
  An interval-accept of a candidate that violates the specification requires Y < tau <= lo,
  i.e. Y outside C(X). Hence **P(accept AND violate) <= P(Y not in C(X)) <= alpha.** (The
  per-split inequality 'false accepts / n <= miscoverage share' holds by construction and is
  asserted in the code only as a consistency check of the implementation; it is not
  empirical evidence for G1. The empirical evidence is the realised miscoverage per stratum.)
* **G1 does not bound the false-accept rate among accepted candidates.**
  P(violate | accept) = P(accept AND violate) / P(accept); with alpha = 0.1 and only 10 % of
  candidates accepted this ratio can reach 1. The self-test in `trustlayer._selftest`
  (exchangeable synthetic data, all assumptions true) gives a concrete case:
  interval-accept keeps P(accept AND violate) = {gap_fa_int:.3f} <= 0.10 but its FDR among
  accepted is {gap_fdr_int:.3f}.
* **G2, FDR.** Conformal selection (Jin & Candes 2023) with p_j =
  (1 + #{{i : y_i <= tau, mu_i >= mu_j}}) / (n + 1) (derived from their clipped score
  V(x, y) = M 1{{y > tau}} + tau - mu(x); numerically identical to the generic formula,
  max |diff| = {pdiff}) and Benjamini-Hochberg at q bounds E[FDP] <= q when each candidate
  is exchangeable with the calibration records of its stratum. Same synthetic case:
  FDR = {gap_fdr_sel:.3f} <= 0.10, at the price of accepting {gap_n_sel:.1f} instead of
  {gap_n_int:.1f} candidates per batch. The bound is for the set BH is run on: joint BH
  bounds the FDR of the whole accepted set, not of a subgroup (e.g. the design-region
  candidates); per-stratum BH bounds it within each stratum, not for the union.
* **G3, shift check.** Conformal novelty p-values (Bates et al. 2023) are super-uniform for
  in-domain candidates, so an in-domain candidate is flagged with probability <= 0.05
  (synthetic check: {flag_in:.3f}). There is no guarantee that a shifted candidate is
  flagged; a shifted candidate that is NOT flagged is decided with source quantiles, for
  which neither G1 nor G2 holds.
* **Correct routing is not enough.** Even a shifted candidate routed to the target stratum
  only has G1 (a bound on P(accept AND violate)) or, under joint BH, G2 for the whole
  accepted set. Neither bounds the error rate among the (few) accepted design-region
  candidates; only BH run inside the target stratum does (the oracle-routing diagnostics D3
  and D4 below show this empirically).
* **None of G1-G3 is a clinical safety or reliability guarantee.** They are statements about
  a measured material property under an exchangeability assumption that the design-region
  shift itself violates for the source stratum. Safety of a biomedical material requires
  biological evidence (ISO 10993 biocompatibility, in-vitro and in-vivo testing); the most a
  trust layer can do is decide which candidates to send to such testing and bound the
  error rate of the statistical screen that precedes it.
"""

GLASS = ["glass_Tg", "glass_E", "glass_HV", "glass_Tliq"]
STEEL, POLY = "steel_yield", "polymer_Tg"


def _S(res, dn, mo):
    return res[dn]["models"][mo]


def _m(res, dn, mo, tn, arm, sub, met, key="mean"):
    return _S(res, dn, mo)["tau"][tn]["arms"][arm][sub][met].get(key)


def _pool(res, dn, mo, tn, arm, sub):
    return _S(res, dn, mo)["tau"][tn]["pooled"][arm][sub]


def _z(res, dn, mo, tn, arm, sub="all", met="fdp", ref=Q_FDR):
    s = _S(res, dn, mo)["tau"][tn]["arms"][arm][sub][met]
    se = s["sd"] / math.sqrt(s["n"]) if s.get("n", 0) > 1 else float("nan")
    return (s["mean"] - ref) / se if se > 0 else float("nan"), se


def _span(vals, d=2, pct=False):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "n/a"
    lo, hi = min(v), max(v)
    k = 100.0 if pct else 1.0
    fmt = (lambda x: f"{k * x:.0f} %") if pct else (lambda x: f"{x:.{d}f}")
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}-{fmt(hi)}"


def findings(res, selftest):
    """Findings generated from the summary JSON (every number traceable to it)."""
    ALL = list(res)
    H = HEAD
    A = lambda dsets, arm, sub, met, mo="RF", tn=H, d=2, pct=False: _span(
        [_m(res, dn, mo, tn, arm, sub, met) for dn in dsets], d, pct)
    SC = lambda dsets, k, mo="RF", d=2: _span([_S(res, dn, mo)["shift_check"][k]["mean"] for dn in dsets], d)
    ST = lambda dsets, k, mo="RF", d=2: _span([_S(res, dn, mo)["state"][k]["mean"] for dn in dsets], d)
    pool = lambda dn, mo="RF": _S(res, dn, mo)["mix"]["n_pool"]["mean"]
    share = lambda dsets, arm, mo="RF", tn=H: _span(
        [_m(res, dn, mo, tn, arm, "all", "measurements") / pool(dn, mo) for dn in dsets], pct=True)
    fa_t = lambda dn, mo, a, tn=H: _m(res, dn, mo, tn, a, "true_target", "false_accepts")

    def decided(dsets, mo="RF"):
        v = []
        for dn in dsets:
            n = _m(res, dn, mo, H, "A3_layer", "flagged", "n")
            a = _m(res, dn, mo, H, "A3_layer", "flagged", "n_accept")
            r = _m(res, dn, mo, H, "A3_layer", "flagged", "n_reject")
            v.append((a + r) / n if n else None)
        return _span(v, pct=True)

    def reduction(dsets, arm, mo="RF"):
        return _span([1 - fa_t(dn, mo, arm) / fa_t(dn, mo, "A0_point") if fa_t(dn, mo, "A0_point") else None
                      for dn in dsets], pct=True)

    g, h = selftest["gap_demo"], selftest.get("homoscedastic_demo", {})
    P = []
    # 1 ---------------------------------------------------------------- scope of "one procedure"
    P.append("1. **One object, one pass per candidate; what that does and does not cover.** Every arm is the same "
             "`TrustLayer` object with a component switched off (A1: shift check off; A2: no target records; A5: shift "
             "check and target records off) or its accept rule swapped (A4/A4s: conformal selection). R7 therefore "
             "demonstrates manuscript components (1) distribution-free UQ and (2) shift robustness (novelty shift check "
             "plus budgeted target recalibration), joined by one accept/reject/measure/not-certified rule. It does NOT "
             "demonstrate component (4), incomplete-modality handling: Mondrian groups keyed by a missing-measurement "
             "pattern are implemented in `trustlayer.py` but every call here uses groups=None, and only `_selftest` checks "
             "their routing; that evaluation belongs to the missing-pattern experiment. Component (3), cost/risk-aware "
             "SEQUENTIAL experimentation (manuscript Sec. 4.3), is represented only by a static 'measure' label and a "
             "fixed recalibration budget m; there is no sequential acquisition policy here.")
    # 2 ---------------------------------------------------------------- shift check
    P.append(f"2. **Shift check: its false-flag guarantee holds; its power is partial.** Source candidates flagged: "
             f"{SC(ALL, 'source_flagged', d=3)} (guarantee: <= 0.05). Target candidates flagged: glass "
             f"{SC(GLASS, 'target_flagged')}, steel {SC([STEEL], 'target_flagged')}, polymer "
             f"{SC([POLY], 'target_flagged')}; AUROC of the novelty p-value for target membership "
             f"{SC(ALL, 'auroc_pvalue')}. A BH-corrected check would flag {SC(GLASS + [STEEL], 'target_flagged_bh')} "
             f"of glass/steel target candidates (polymer {SC([POLY], 'target_flagged_bh')}), which is why "
             f"the per-candidate rule is used (the shift check does not depend on tau).")
    # 3 ---------------------------------------------------------------- no shift check
    tsh_g = _span([_S(res, dn, "RF")["mix"]["target_share"]["mean"] for dn in GLASS], pct=True)
    P.append(f"3. **Without the shift check, calibrated intervals still accept failing design-region candidates.** "
             f"Among accepted TARGET candidates the false-accept proportion is {A(GLASS, 'A0_point', 'true_target', 'fdp')} "
             f"for point prediction (A0) and {A(GLASS, 'A1_conformal', 'true_target', 'fdp')} for source-calibrated conformal "
             f"(A1) on the four glass properties (RF; HistGB {A(GLASS, 'A0_point', 'true_target', 'fdp', 'HistGB')} and "
             f"{A(GLASS, 'A1_conformal', 'true_target', 'fdp', 'HistGB')}). On glass, A1's FDR over the whole candidate pool "
             f"is only {A(GLASS, 'A1_conformal', 'all', 'fdp')}, because target candidates are just {tsh_g} of the glass "
             f"pool: a pooled error rate hides the design-region failure. This dilution argument is specific to glass. "
             f"On steel A1 accepts {A([STEEL], 'A1_conformal', 'all', 'n_accept')} candidates per split, so its FDR of "
             f"{A([STEEL], 'A1_conformal', 'all', 'fdp')} says nothing. On polymer, target candidates are "
             f"{pc(_S(res, POLY, 'RF')['mix']['target_share']['mean'])} of the pool, and A1's target FDR is itself only "
             f"{A([POLY], 'A1_conformal', 'true_target', 'fdp')} (overall {A([POLY], 'A1_conformal', 'all', 'fdp')}).")
    # 4 ---------------------------------------------------------------- attribution: A2 vs A3
    n_gt, n_eq = 0, 0
    for dn in GLASS:
        for mo in MODELS:
            a2, a3 = fa_t(dn, mo, "A2_layer_no_target"), fa_t(dn, mo, "A3_layer")
            n_gt += a3 > a2 + 1e-12
            n_eq += abs(a3 - a2) <= 1e-12

    def extra(dn, mo):
        X = _S(res, dn, mo)["tau"][H]["a3_minus_a2"]
        pt = X["pooled"]
        return (f"{dn} {mo} {X['extra_false_accepts_true_target']['mean']:.2f}/{X['n_extra_accept_true_target']['mean']:.2f}"
                f" ({pc(pt['fdp_true_target'])})")
    xm = lambda dsets, mo="RF": _span([_S(res, dn, mo)["tau"][H]["a3_minus_a2"]["extra_measurements"]["mean"] for dn in dsets], 1)
    P.append(f"4. **The drop in design-region false accepts comes from the shift check plus abstention (A2), not from "
             f"target recalibration.** False accepts among target candidates per split (RF, glass): A0 "
             f"{A(GLASS, 'A0_point', 'true_target', 'false_accepts', d=1)}, A1 {A(GLASS, 'A1_conformal', 'true_target', 'false_accepts', d=1)}, "
             f"A2 {A(GLASS, 'A2_layer_no_target', 'true_target', 'false_accepts', d=1)}, A3 "
             f"{A(GLASS, 'A3_layer', 'true_target', 'false_accepts', d=1)}. Reduction relative to A0 on glass: A2 (no target "
             f"data) {reduction(GLASS, 'A2_layer_no_target')}, A3 {reduction(GLASS, 'A3_layer')} (HistGB: A2 "
             f"{reduction(GLASS, 'A2_layer_no_target', 'HistGB')}, A3 {reduction(GLASS, 'A3_layer', 'HistGB')}; polymer RF: A2 "
             f"{reduction([POLY], 'A2_layer_no_target')}, A3 {reduction([POLY], 'A3_layer')}). By construction A3's accept "
             f"and reject sets contain A2's (in-domain candidates get the same source interval; A2 leaves flagged ones "
             f"uncertified), so recalibration can only add decisions. It adds target false accepts in {n_gt} of 8 glass "
             f"dataset-model pairs (equal in {n_eq}), and costs m = "
             f"{_span([_S(res, dn, 'RF')['m'] for dn in GLASS], 0)} extra measurements, partly offset by the flagged "
             f"candidates it decides: net extra measurements per split {xm(GLASS)} on glass (RF). The design-region accepts "
             f"that recalibration adds, as false/added per split (pooled false share over 20 splits): "
             + "; ".join(extra(dn, mo) for dn in GLASS + [POLY] for mo in MODELS) +
             ". So target recalibration with m <= 30 buys very few design-region acceptances, and on some glass "
             "properties most of them are false.")
    # 5 ---------------------------------------------------------------- counts vs rates, oracle routing
    # pooled totals (not per-split means): subset means are averaged only over the splits in which
    # the subset is non-empty, so the ratio of two subset means would not be a share.
    def fa_tot(dn, mo, a, sub):
        return _pool(res, dn, mo, H, a, sub)["false_accepts_total"]
    unfl_share = _span([fa_tot(dn, "RF", "A3_layer", "target_unflagged") / fa_tot(dn, "RF", "A3_layer", "true_target")
                        if fa_tot(dn, "RF", "A3_layer", "true_target") else None for dn in GLASS], pct=True)
    unfl_cnt = "; ".join(f"{dn} {fa_tot(dn, 'RF', 'A3_layer', 'target_unflagged')} of "
                         f"{fa_tot(dn, 'RF', 'A3_layer', 'true_target')}" for dn in GLASS)

    def d_txt(arm, dn, mo):
        s = _S(res, dn, mo)["tau"][H]["arms"][arm]["true_target"]
        pl = _pool(res, dn, mo, H, arm, "true_target")
        return (f"{dn} {mo} {f(s['fdp']['mean'], 2)} ({f(s['n_accept']['mean'], 2)} accepts/split, at least one in "
                f"{pc(s['any_accepted']['mean'])} of splits, pooled {f(pl['fdp_pooled'], 2)} = "
                f"{pl['false_accepts_total']}/{pl['n_accept_total']})")
    worse = [f"{dn} {mo} ({f(_m(res, dn, mo, H, 'D4_layer_fdr_oracle_routing', 'true_target', 'fdp'), 2)} vs "
             f"{f(_m(res, dn, mo, H, 'A4_layer_fdr', 'true_target', 'fdp'), 2)})"
             for dn in GLASS for mo in MODELS
             if _m(res, dn, mo, H, "D4_layer_fdr_oracle_routing", "true_target", "fdp") >
             _m(res, dn, mo, H, "A4_layer_fdr", "true_target", "fdp")]
    thr_g = _span([_S(res, dn, "RF")["tau"][H]["a4s_granularity"]["bh_threshold_joint_D4"]["mean"] for dn in GLASS], 3)
    pmin_g = _span([_S(res, dn, "RF")["tau"][H]["a4s_granularity"]["p_min"]["mean"] for dn in GLASS], 3)
    P.append(f"5. **Missed shifted candidates carry most of the COUNT of residual false accepts; correct routing does not "
             f"make the design-region error RATE small.** Under A3, {unfl_share} of the target false accepts on glass (RF) "
             f"come from target candidates the shift check missed, which are decided with source quantiles (totals over "
             f"the 20 splits: {unfl_cnt}). With oracle routing (D3, diagnostic only) "
             f"the count falls to {A(GLASS, 'D3_layer_oracle_routing', 'true_target', 'false_accepts', d=2)} per split "
             f"(HistGB {A(GLASS, 'D3_layer_oracle_routing', 'true_target', 'false_accepts', 'HistGB', d=2)}). But the few "
             f"design-region accepts that remain are often false. D3 target FDR: "
             + "; ".join(d_txt("D3_layer_oracle_routing", dn, mo) for dn in GLASS for mo in MODELS) +
             ". D4 (joint-BH selection with oracle routing) target FDR: "
             + "; ".join(d_txt("D4_layer_fdr_oracle_routing", dn, mo) for dn in GLASS for mo in MODELS) +
             f". D4's target FDR is higher than A4's (non-oracle) in {len(worse)} of 8 glass pairs"
             + (f": {', '.join(worse)}" if worse else "") +
             ". This is what the theory predicts. Under D3 the design-region accepts come from target-stratum intervals, "
             "and G1 bounds only P(accept AND violate) among target candidates, not the rate among the few accepted. "
             f"Under D4, joint BH bounds the FDR of the whole accepted set. That set is dominated by in-domain accepts, "
             f"so the BH threshold q|selected|/n is {thr_g} on glass, well above the smallest target-stratum p-value "
             f"1/(m+1) = {pmin_g}: a target candidate is selected as soon as only one or two of the m target records "
             f"that fail the specification have a prediction at least as high as its own. "
             "Only BH run inside the target stratum (A4s, finding 8) targets the design-region subgroup. Several of these "
             "subgroup FDRs rest on fewer than one accept per split, so their percentile ranges span [0, 1] (tables below).")
    # 6 ---------------------------------------------------------------- recalibration: coverage scope, decisiveness
    P.append(f"6. **Target recalibration restores coverage only for FLAGGED candidates and rarely produces a decision.** "
             f"With m = {_span([_S(res, dn, 'RF')['m'] for dn in ALL], 0)} target records, A3's miscoverage on flagged "
             f"candidates is {A(ALL, 'A3_layer', 'flagged', 'miscoverage')} (RF; HistGB "
             f"{A(ALL, 'A3_layer', 'flagged', 'miscoverage', 'HistGB')}; nominal 0.10). Across ALL true target candidates "
             f"A3's miscoverage on glass is {A(GLASS, 'A3_layer', 'true_target', 'miscoverage')}, because unflagged target "
             f"candidates keep source quantiles, whose miscoverage on them is "
             f"{A(GLASS, 'A3_layer', 'target_unflagged', 'miscoverage')} (steel {A([STEEL], 'A3_layer', 'target_unflagged', 'miscoverage')}). "
             f"With oracle routing (D3) the target miscoverage is {A(ALL, 'D3_layer_oracle_routing', 'true_target', 'miscoverage')}. "
             f"The target-stratum half-width is {ST(ALL, 'q_target_over_tgt_iqr')} x the target IQR (source stratum "
             f"{ST(ALL, 'q_source_over_tgt_iqr')}), so A3 decides (accepts or rejects) only {decided(GLASS)} of flagged "
             f"candidates on glass, {decided([STEEL])} on steel and {decided([POLY])} on polymer; the rest go to "
             f"measurement. Measurements used, including m, are {share(ALL, 'A3_layer')} of the pool for A3, "
             f"{share(ALL, 'A2_layer_no_target')} for A2 (no target data) and {share(ALL, 'A1_conformal')} for A1. "
             f"Recalibration with m <= 30 thus buys a valid 'measure' decision for flagged design-region candidates, "
             f"rarely a certified acceptance. This is the cost of target data raised in Reviewer 2 comment 7.")
    # 7 ---------------------------------------------------------------- A4 joint BH
    med = [(dn, mo) + _z(res, dn, mo, H, "A4_layer_fdr") + (_m(res, dn, mo, H, "A4_layer_fdr", "all", "fdp"),)
           for dn in ALL for mo in MODELS]
    mx = max(med, key=lambda r: r[4])
    over_h = [r for r in med if r[4] > Q_FDR]
    sig_h = [r for r in over_h if r[2] >= 2]
    q75 = [(dn, mo) + _z(res, dn, mo, HEAD_Q75, "A4_layer_fdr") + (_m(res, dn, mo, HEAD_Q75, "A4_layer_fdr", "all", "fdp"),)
           for dn in GLASS for mo in MODELS]
    over75 = [r for r in q75 if r[4] > Q_FDR]
    sig75 = [r for r in over75 if r[2] >= 2]
    d4_75 = [(dn, mo) + _z(res, dn, mo, HEAD_Q75, "D4_layer_fdr_oracle_routing") +
             (_m(res, dn, mo, HEAD_Q75, "D4_layer_fdr_oracle_routing", "all", "fdp"),) for dn in GLASS for mo in MODELS]
    d4sig = [r for r in d4_75 if r[4] > Q_FDR and r[2] >= 2]
    lab = lambda r: f"{r[0]} {r[1]} {r[4]:.3f} (z = {r[2]:.1f})"
    P.append(f"7. **FDR-controlled acceptance with one joint BH (A4) controls the accepted set as a whole, not the "
             f"design-region subgroup.** At the headline tau, A4's FDR over all accepted candidates is "
             f"{A(ALL, 'A4_layer_fdr', 'all', 'fdp', d=3)} (RF; HistGB {A(ALL, 'A4_layer_fdr', 'all', 'fdp', 'HistGB', d=3)}); "
             f"the largest is {mx[0]} {mx[1]} {mx[4]:.3f}, z = {mx[2]:.1f}, where z = (FDR - q) / SE and SE is the "
             f"split-to-split sd divided by sqrt(20). "
             + (f"{len(over_h)} of {len(med)} dataset-model pairs exceed q = 0.10 nominally, "
                + (f"of which {', '.join(lab(r) for r in sig_h)} by >= 2 SE. " if sig_h else "none by 2 SE or more. ")
                if over_h else "no pair exceeds q = 0.10. ") +
             f"Measurements fall to {share(GLASS, 'A4_layer_fdr')} of the pool on glass. Among accepted TARGET candidates, "
             f"however, A4's FDR is {A(GLASS, 'A4_layer_fdr', 'true_target', 'fdp')} on glass (RF; HistGB "
             f"{A(GLASS, 'A4_layer_fdr', 'true_target', 'fdp', 'HistGB')}). At the stricter tau = 75th percentile of the "
             f"recalibration pool, A4's overall FDR on glass is {_span([r[4] for r in q75], 3)}. It exceeds 0.10 in "
             f"{len(over75)} of {len(q75)} glass pairs, "
             + (f"but by 2 SE or more only for {', '.join(lab(r) for r in sig75)}; the rest are within noise. "
                if sig75 else "none of them by 2 SE or more (within split-to-split noise). ") +
             f"Replacing the flag by true membership (D4) gives {_span([r[4] for r in d4_75], 3)} at this threshold ("
             + (f"still >= 2 SE above q for {', '.join(lab(r) for r in d4sig)}"
                if d4sig else "no pair 2 SE or more above q") +
             "). So the excess of A4 is consistent with unflagged target candidates being routed to the source stratum, "
             "whose calibration records they are not exchangeable with.")
    # 8 ---------------------------------------------------------------- A4s and p-value granularity
    a4s_fl = max((_m(res, dn, mo, tn, "A4s_layer_fdr_by_stratum", "flagged", "fdp") or 0, dn, mo, tn)
                 for dn in ALL for mo in MODELS for tn in [H, HEAD_Q75])

    def poly_txt(mo):
        s = _S(res, POLY, mo)["tau"][H]["arms"]["A4s_layer_fdr_by_stratum"]["true_target"]
        G = _S(res, POLY, mo)["tau"][H]["a4s_granularity"]
        anyp = s["any_accepted"]["mean"]
        given = s["n_accept"]["mean"] / anyp if anyp else None
        fia = s["fdp_if_any"]["mean"]
        return (f"{mo}: target candidates accepted in {round(20 * anyp)} of 20 splits, about "
                f"{f(given, 0)} per split when any (mean over all splits {f(s['n_accept']['mean'], 1)}); target FDP in "
                f"those splits {f(fia, 3)} ({'above' if fia is not None and fia > Q_FDR else 'at or below'} q), FDR over "
                f"all splits {f(s['fdp']['mean'], 3)}; flagged stratum {f(G['n_flagged']['mean'], 0)} candidates, "
                f"smallest possible p = 1/(m+1) = {f(G['p_min']['mean'], 3)}, so any non-empty BH selection in it has at "
                f"least k = {f(G['k_needed']['mean'], 0)} members (candidates tied at the minimum p: "
                f"{f(G['n_flagged_at_p_min']['mean'], 1)} on average)")
    P.append(f"8. **BH within each stratum (A4s) controls the flagged stratum, but is all-or-nothing with m <= 30 target "
             f"records.** The FDR inside the flagged stratum is at most {a4s_fl[0]:.3f} ({a4s_fl[1]} {a4s_fl[2]}, "
             f"{a4s_fl[3]}) over all datasets, both models and both recalibration-pool thresholds. On polymer "
             f"({pc(_S(res, POLY, 'RF')['shift_check']['target_flagged']['mean'])} of target candidates flagged): "
             f"{poly_txt('RF')}. {poly_txt('HistGB')}. The cause is p-value "
             f"granularity: with m target records no candidate can have p below 1/(m+1), and BH at q = 0.10 over "
             f"n_f flagged candidates selects nothing unless about n_f/(10(m+1)) candidates or more tie at that minimum. "
             f"Per-stratum BH is therefore not a route to many certified design-region acceptances at m <= 30. The FDR of "
             f"the union of both strata's selections on polymer is {A([POLY], 'A4s_layer_fdr_by_stratum', 'all', 'fdp', d=3)} "
             f"(HistGB {A([POLY], 'A4s_layer_fdr_by_stratum', 'all', 'fdp', 'HistGB', d=3)}); per-stratum BH does not "
             f"bound the union.")
    # 9 ---------------------------------------------------------------- A5 negative control
    P.append(f"9. **Negative control.** Conformal selection calibrated on the source only (A5) has FDR "
             f"{A(GLASS + [POLY], 'A5_selection_source_only', 'all', 'fdp')} overall and "
             f"{A(GLASS + [POLY], 'A5_selection_source_only', 'true_target', 'fdp')} on target candidates (glass and polymer, "
             f"RF; HistGB {A(GLASS + [POLY], 'A5_selection_source_only', 'all', 'fdp', 'HistGB')} and "
             f"{A(GLASS + [POLY], 'A5_selection_source_only', 'true_target', 'fdp', 'HistGB')}): an FDR procedure calibrated "
             f"on the source does not protect the design region.")
    # 10 --------------------------------------------------------------- steel
    st = _S(res, STEEL, "RF")

    def any20(arm, mo):
        return round(20 * _m(res, STEEL, mo, H, arm, "all", "any_accepted"))
    calib = ["A1_conformal", "A2_layer_no_target", "A3_layer", "A4_layer_fdr", "A4s_layer_fdr_by_stratum",
             "A5_selection_source_only"]
    anyacc = "; ".join(f"{mo}: " + ", ".join(f"{a.split('_')[0]} {any20(a, mo)}" for a in calib) for mo in MODELS)
    rej_fp = _pool(res, STEEL, "RF", H, "A3_layer", "true_target")
    P.append(f"10. **Steel is small and its rates are noisy.** The pool has {f(st['mix']['n_pool']['mean'], 0)} candidates "
             f"({f(st['mix']['n_target_cand']['mean'], 0)} target), the source calibration set "
             f"{f(st['state']['n_source_cal']['mean'], 0)} records and m = {st['m']}. Point prediction accepts "
             f"{A([STEEL], 'A0_point', 'all', 'n_accept', d=1)} candidates per split, at FDR "
             f"{ci(st['tau'][H]['arms']['A0_point']['all']['fdp'], 2)} (HistGB "
             f"{ci(_S(res, STEEL, 'HistGB')['tau'][H]['arms']['A0_point']['all']['fdp'], 2)}). The calibrated arms accept "
             f"almost nothing. Splits (of 20) with any accept: {anyacc}. They are not uniformly cautious, though. Pooled "
             f"over the 20 splits A3 rejects {rej_fp['n_reject_total']} target candidates (out of "
             f"{f(20 * st['mix']['n_target_cand']['mean'], 0)} target candidates in total), "
             f"{_pool(res, STEEL, 'RF', H, 'A3_layer', 'target_unflagged')['n_reject_total']} of them unflagged candidates "
             f"decided with source-quantile intervals, whose miscoverage on them is "
             f"{A([STEEL], 'A3_layer', 'target_unflagged', 'miscoverage')}; "
             f"{rej_fp['false_rejects_total']} of those rejects ({pc(rej_fp['false_reject_prop_pooled'])}) are false. "
             f"The design region is otherwise sent to measurement.")
    # 11 --------------------------------------------------------------- coverage is not FDR
    P.append(f"11. **Coverage is not an error rate among accepted candidates.** In the exchangeable source subset, "
             f"interval-accept (A1) has FDR {A(ALL, 'A1_conformal', 'true_source', 'fdp', d=3)} at miscoverage "
             f"{A(ALL, 'A1_conformal', 'true_source', 'miscoverage', d=3)}; it is conservative in these data, but nothing "
             f"guarantees that. In the synthetic check with all assumptions true, the same rule has FDR "
             f"{g['interval_accept']['FDR_among_accepted']:.3f} with P(accept and violate) = "
             f"{g['interval_accept']['P(accept and violate)']:.3f} <= 0.10. Conformal selection is tight at q on clean data "
             f"(synthetic homoscedastic case: FDR {h.get('selection_FDR', float('nan')):.3f}; in-domain source candidates "
             f"here: {A(ALL, 'A4_layer_fdr', 'true_source', 'fdp', d=3)}).")
    # 12 --------------------------------------------------------------- threshold definition
    diffs = []
    for dn in ALL:
        for mo in MODELS:
            for a in ARMS:
                for s_ in ["all", "true_target", "flagged"]:
                    x0 = _m(res, dn, mo, H, a, s_, "fdp")
                    for tn in ["tgt_median", "rpool_holdout_median"]:
                        x1 = _m(res, dn, mo, tn, a, s_, "fdp")
                        if x0 is not None and x1 is not None:
                            diffs.append((abs(x1 - x0), tn, dn, mo, a, s_, x0, x1))
    NOTST = [dn for dn in ALL if dn != STEEL]
    top = sorted([d_ for d_ in diffs if d_[1] == "tgt_median" and d_[2] != STEEL], reverse=True)[:4]
    gl = max(d_[0] for d_ in diffs if d_[1] == "tgt_median" and d_[2] in NOTST)
    gl_all = max(d_[0] for d_ in diffs if d_[1] == "tgt_median" and d_[2] in NOTST and d_[5] == "all")
    st_d = max(d_[0] for d_ in diffs if d_[1] == "tgt_median" and d_[2] == STEEL)
    ho_top = max((d_ for d_ in diffs if d_[1] == "rpool_holdout_median" and d_[2] != STEEL))
    ho = ho_top[0]
    ho_st = max(d_[0] for d_ in diffs if d_[1] == "rpool_holdout_median" and d_[2] == STEEL)
    a4_tgt = max(_m(res, dn, mo, "tgt_median", "A4_layer_fdr", "all", "fdp") for dn in ALL for mo in MODELS)
    # stability of the qualitative conclusions across all five threshold definitions (12 cells each)
    cells = [(dn, mo) for dn in ALL for mo in MODELS]
    chk = {"A3 target false accepts <= A1's":
           lambda tn, dn, mo: _m(res, dn, mo, tn, "A3_layer", "true_target", "false_accepts") <=
           _m(res, dn, mo, tn, "A1_conformal", "true_target", "false_accepts") + 1e-9,
           "A4s FDR inside the flagged stratum <= q":
           lambda tn, dn, mo: (_m(res, dn, mo, tn, "A4s_layer_fdr_by_stratum", "flagged", "fdp") or 0) <= Q_FDR + 1e-9,
           "A5 target FDR >= A3 target FDR":
           lambda tn, dn, mo: (_m(res, dn, mo, tn, "A5_selection_source_only", "true_target", "fdp") or 0) >=
           (_m(res, dn, mo, tn, "A3_layer", "true_target", "fdp") or 0) - 1e-9,
           "A4 overall FDR <= q + 2 SE":
           lambda tn, dn, mo: _z(res, dn, mo, tn, "A4_layer_fdr")[0] <= 2}
    stab = "; ".join(f"{k}: " + "/".join(str(sum(fn(tn, dn, mo) for dn, mo in cells)) for tn in TAUS) + " of 12"
                     for k, fn in chk.items())
    P.append(f"12. **How tau is defined matters for small subgroups.** The headline tau is the recalibration-pool "
             f"median (no target-test label). The task's literal definition, the whole-target-pool median (which uses "
             f"target-test labels), changes the FDR of any arm by at most {gl_all:.3f} over all candidates of a dataset "
             f"(glass and polymer) and by up to {gl:.3f} in their subgroups (true target, flagged); on steel, where one "
             f"decision moves a rate by 0.2 or more, differences reach {st_d:.3f}. Largest differences outside steel, "
             f"headline -> whole-target median: "
             + "; ".join(f"{d_[2]} {d_[3]} {d_[4].split('_')[0]} {d_[5]} {d_[6]:.3f} -> {d_[7]:.3f}" for d_ in top) +
             f". With the whole-target median, A4's largest overall FDR is {a4_tgt:.3f}. Using only the "
             f"recalibration-pool records NOT used for recalibration (independent of the target-stratum calibration "
             f"labels; about 7-13 records on steel, glass_E and glass_HV) moves subgroup FDRs by up to {ho:.3f} outside "
             f"steel ({ho_top[2]} {ho_top[3]} {ho_top[4].split('_')[0]} {ho_top[5]} {ho_top[6]:.3f} -> {ho_top[7]:.3f}) "
             f"and {ho_st:.3f} on steel; with 7-13 records that threshold is itself noisy, which is why it is a "
             f"sensitivity and not the headline. Stability of the statements above across the five definitions (number of the "
             f"12 dataset-model cells satisfying each check, in the order " + "/".join(TAUS) + f") - {stab}. The first "
             f"three checks, which carry findings 4, 8 and 9, hold in essentially every cell under every definition; "
             f"whether A4's overall FDR exceeds q by more than 2 SE does depend on the threshold, which is finding 7.")
    # 13 --------------------------------------------------------------- claims
    P.append("13. **What the manuscript can claim.** One object with one decision rule combines distribution-free UQ and "
             "shift handling (components 1-2). Incomplete modalities (4) are supported by the same object but evaluated "
             "elsewhere, and sequential experimentation (3) is not shown. Coverage (G1) and FDR (G2) hold only for "
             "candidates exchangeable with the stratum they are decided with. G2 holds only for the set BH is run on. The "
             "shift check (G3) bounds false flags but guarantees nothing about catching shifted candidates. The supported "
             "empirical claim, relative to point prediction: the shift check plus abstention (not target recalibration) "
             "removes most false accepts among design-region candidates and turns the design region into explicit "
             "measurement requests. With m <= 30 target records the layer certifies very few design-region candidates, and "
             "those it does certify through intervals or joint BH are often false, even with perfect routing. The "
             "design-region error rate is reduced, not bounded. None of this is a clinical safety guarantee.")
    return P


def deviations_md(res):
    return ["## Deviations from the protocol and the task specification", "",
            "* **Specification threshold.** The task defines tau as the median (sensitivity: 75th percentile) of y over "
            "the dataset's whole target pool. That uses target-test labels, which PROTOCOL.md forbids for anything "
            "selected from target labels. The headline therefore uses the median of the split's recalibration pool "
            "(`rpool_median`; sensitivity `rpool_q75`). The recalibration pool contains the m records used for "
            "recalibration, so tau is not independent of the target-stratum calibration labels (a small departure "
            "from the exchangeability G2 assumes). The variant `rpool_holdout_median` removes that dependence "
            "(7-13 records on steel, glass_E and glass_HV, so it is noisy there). The task's literal thresholds "
            "(`tgt_median`, `tgt_q75`) are reported as sensitivity only. No threshold is used by any trust-layer "
            "component for fitting, calibration, flagging or recalibration. Real differences: finding 12 and the "
            "sensitivity table.",
            "* **Candidate pool** = source-test U target-test, as the task specifies. Metrics are reported pooled and "
            "split by true membership and by flag.",
            "* **Metric set.** The full interval-metric set (interval score, WIS, calibration error, worst-cluster "
            "coverage, AURC) is not recomputed here; R1_core covers it. This script reports decision metrics (FDP/FDR, "
            "pooled FDP, power, measurements, false-reject proportion, unconditional false-accept probability) plus "
            "interval miscoverage and q/IQR.",
            "* **Extra arms** beyond the specification: A4s (BH per stratum) shows the scope of the FDR guarantee. D3 "
            "and D4 (A3/A4 with routing by true target membership) are diagnostics only; they use membership, never "
            "property labels.",
            "* **Measurements** include the recalibration budget m for every arm that spends it (A3, A4, A4s, D3, D4).",
            "* **Conformal selection** uses the task's formula, which counts y_i <= tau, i.e. it tests the null "
            "Y <= tau. For the specification y >= tau this is valid, and conservative only through calibration labels "
            "exactly equal to tau (derivation and numerical check in `trustlayer.selection_pvalues_closed_form` and "
            "`_selftest`).",
            "* **Shift check multiplicity.** Each candidate is flagged if p < 0.05, without multiplicity correction "
            "(justified in `TrustLayer.shift_check`; the BH variant is reported). Novelty settings (k = 10, level "
            "0.05) were fixed a priori with no tuning.",
            "* **Compute.** joblib ran 6 workers with 2 threads each (threadpoolctl; RF n_jobs = 2).", ""]


VERIFIER_MD = ["## Verifier issues (fix round): what changed", "",
               "All ten issues raised by the independent verifier were adopted; none was rejected.",
               "",
               "1. Counts vs rates (D3/D4). Finding 5 now separates the COUNT of residual false accepts (mostly shift-check "
               "misses) from the RATE among the few design-region accepts, which stays high even with oracle routing. D3/D4 "
               "subgroup FDRs are reported with accepts per split, any-accepted share and pooled FDP.",
               "2. Attribution. Finding 4 credits the reduction to the shift check plus abstention (A2), and reports A3 minus "
               "A2 (extra accepts, their false share, extra measurements; table 'What target recalibration adds'). Finding 6 "
               "limits 'restores coverage' to flagged candidates and reports miscoverage over all target candidates. "
               "Clarification: A3 >= A2 in target false accepts is structural. A3's accept set contains A2's (asserted in "
               "the code), so recalibration can only add accepts; the question is whether they are correct.",
               "3. A4s on polymer. Finding 8 and the granularity table report any_accepted, the count when any, fdp_if_any "
               "and the minimum BH selection size 1/(m+1)-granularity; the phrase 'many certified acceptances' is gone.",
               "4. Protocol. The headline threshold is now the recalibration-pool median (no target-test label). The task's "
               "target-pool thresholds are reported as sensitivity with their real differences (finding 12, deviations "
               "section). The earlier claim that the threshold definition changes FDRs by at most about 0.02 was wrong and "
               "is withdrawn.",
               "5. Scope of 'one procedure'. Findings 1 and 13 and the component mapping in the `trustlayer.py` docstring now "
               "say that R7 exercises components 1-2 plus the decision rule; component 4 is implemented but evaluated "
               "elsewhere; component 3 is not sequential here.",
               "6. The pooled-dilution explanation (finding 3) is restricted to glass; steel and polymer are described "
               "separately.",
               "7. Steel wording (finding 10): A3's target rejects through source-quantile intervals, any-accept counts per "
               "arm, and the small-sample caveat.",
               "8. A4 at the 75th percentile (finding 7): exceedances are tested against split-to-split SE, and D4 is cited as "
               "the routing diagnostic.",
               "9. Subgroup tables now carry n accepted, any-accepted share, FDP given any accept and pooled FDP beside every "
               "subgroup FDR.",
               "10. The header counts m in measurements for A3, A4, A4s, D3 and D4. The deviations are listed in the md. The G1 "
               "per-split inequality is described as a code-consistency check, not as evidence.", "",
               "### Verifier issues not adopted", "", "None.", ""]


def write_md(results, selftest, runtime_min, path=None):
    L = ["# R7 - the trust layer as one decision procedure (Reviewer 2, comments 2, 4 and 7)", "",
         "Script: `revision/code/exp_R7_pipeline.py`; library: `revision/code/trustlayer.py`. "
         "Protocol: `revision/code/PROTOCOL.md` (grouped 60/20/20 source split, target pool halves, "
         "alpha = 0.10, finite-sample conformal quantile, 20 seeds). Numbers are mean "
         "[2.5-97.5 percentile] over 20 splits; the splits resample one finite dataset, so the "
         "interval describes split-to-split variability, not population sampling error. "
         f"Runtime {runtime_min:.1f} min.", "",
         "Candidate pool = source-test U target-test. Specification: acceptable iff y >= tau. **Headline tau = median of "
         "y over the split's recalibration pool** (protocol-compliant: no target-test label enters it; it is used by no "
         "trust-layer component). Sensitivity thresholds: see the deviations section. FDR = mean over splits of the "
         "false-accept proportion among accepted (FDP = 0 when nothing is accepted). 'FDP if any' = mean over the splits "
         "with at least one accept in that subset. 'Pooled FDP' = total false accepts / total accepts over the 20 splits "
         "(more stable than the FDR when a subgroup has fewer than one accept per split). 'Any' = share of splits with at "
         "least one accept in that subset. 'Measurements' = measure + not_certified + recalibration budget m (m counted "
         "for A3, A4, A4s, D3 and D4). 'Uncond. FA' = false accepts / pool size. A subset metric is averaged over the "
         "splits in which that subset is non-empty (e.g. 'target not flagged' is empty in some polymer splits), so the "
         "ratio of two subset means is not a share; shares of counts in the text are computed from pooled totals. Each "
         "summary in the JSON carries its own `n` of splits.", ""]
    g = selftest["gap_demo"]
    L.append(THEORY.format(gap_fa_int=g["interval_accept"]["P(accept and violate)"],
                           gap_fdr_int=g["interval_accept"]["FDR_among_accepted"],
                           gap_fdr_sel=g["conformal_selection"]["FDR_among_accepted"],
                           gap_n_sel=g["conformal_selection"]["mean_n_accepted"],
                           gap_n_int=g["interval_accept"]["mean_n_accepted"],
                           pdiff=selftest["selection_pvalue_closed_vs_generic_maxabsdiff"],
                           flag_in=selftest["novelty_flag_rate_in_domain_mean"]))
    L += ["## Findings (RF, headline tau = recalibration-pool median, unless stated)", ""] + findings(results, selftest) + [""]
    L += deviations_md(results)
    L += VERIFIER_MD

    # ---- what target recalibration adds (A3 minus A2)
    L += ["## What target recalibration adds: A3 minus A2 (headline tau)", "",
          "Per split, mean over 20 splits. 'Extra accepts' are candidates accepted by A3 but not by A2 (all flagged, "
          "decided with target-stratum intervals); 'pooled' = over all 20 splits.", "",
          "| dataset | model | m | extra accepts (target) | false among them | pooled false share (target) | splits with any | "
          "extra rejects | false among extra rejects | net extra measurements | target FA A2 -> A3 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for dn, D in results.items():
        for mo in MODELS:
            S = D["models"][mo]
            X = S["tau"][HEAD]["a3_minus_a2"]
            L.append(f"| {dn} | {mo} | {S['m']} | {f(X['n_extra_accept_true_target']['mean'], 2)} | "
                     f"{f(X['extra_false_accepts_true_target']['mean'], 2)} | {f(X['pooled']['fdp_true_target'], 2)} "
                     f"({X['pooled']['extra_false_accepts_true_target_total']}/{X['pooled']['n_extra_accept_true_target_total']}) | "
                     f"{f(X['any_extra_accept_true_target']['mean'], 2)} | {f(X['n_extra_reject']['mean'], 2)} | "
                     f"{f(X['extra_false_rejects']['mean'], 2)} | {f(X['extra_measurements']['mean'], 1)} | "
                     f"{f(X['target_false_accepts_A2']['mean'], 2)} -> {f(X['target_false_accepts_A3']['mean'], 2)} |")
    L.append("")

    # ---- granularity of per-stratum BH
    L += ["## Per-stratum BH (A4s) inside the flagged stratum: p-value granularity (headline tau)", "",
          "p_min = 1/(n_target + 1) is the smallest attainable selection p-value; any non-empty BH selection at q = 0.10 "
          "among n_f flagged candidates has at least k = ceil(n_f p_min / q) members (asserted per split).", "",
          "| dataset | model | n_target | p_min | n flagged | k needed | k / n flagged | at p_min | splits with any A4s selection | "
          "A4s selected (flagged) | A4 (joint BH) selected (flagged) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for dn, D in results.items():
        for mo in MODELS:
            S = D["models"][mo]
            G = S["tau"][HEAD]["a4s_granularity"]
            L.append(f"| {dn} | {mo} | {f(S['state']['n_target_records']['mean'], 0)} | {f(G['p_min']['mean'], 3)} | "
                     f"{f(G['n_flagged']['mean'], 1)} | {f(G['k_needed']['mean'], 1)} | {f(G['k_needed_share_of_flagged']['mean'], 2)} | "
                     f"{f(G['n_flagged_at_p_min']['mean'], 1)} | {f(G['any_selected_flagged_A4s']['mean'], 2)} | "
                     f"{f(G['n_selected_flagged_A4s']['mean'], 1)} | {f(G['n_selected_flagged_A4']['mean'], 1)} |")
    L.append("")

    # ---- A4 FDR vs q with SE
    L += ["## Joint-BH FDR (A4, and D4 with oracle routing) versus q = 0.10, all candidates", "",
          "SE = split-to-split sd / sqrt(number of splits); z = (FDR - 0.10) / SE.", "",
          "| dataset | model | tau | A4 FDR | A4 SE | A4 z | D4 FDR | D4 z |", "|---|---|---|---|---|---|---|---|"]
    for dn in results:
        for mo in MODELS:
            for tn in [HEAD, HEAD_Q75, "tgt_median", "tgt_q75"]:
                z4, se4 = _z(results, dn, mo, tn, "A4_layer_fdr")
                zd, _ = _z(results, dn, mo, tn, "D4_layer_fdr_oracle_routing")
                L.append(f"| {dn} | {mo} | {tn} | {f(_m(results, dn, mo, tn, 'A4_layer_fdr', 'all', 'fdp'))} | {f(se4)} | "
                         f"{f(z4, 1)} | {f(_m(results, dn, mo, tn, 'D4_layer_fdr_oracle_routing', 'all', 'fdp'))} | {f(zd, 1)} |")
    L.append("")

    # ---- per-dataset tables
    for model in MODELS:
        L += [f"## Results, model {model}, headline tau (recalibration-pool median)", ""]
        for dn, D in results.items():
            S = D["models"][model]
            T = S["tau"][HEAD]
            sc = S["shift_check"]
            L += [f"### {dn} (m = {S['m']}; pool {f(S['mix']['n_pool']['mean'], 0)} = "
                  f"{f(S['mix']['n_source_cand']['mean'], 0)} source + {f(S['mix']['n_target_cand']['mean'], 0)} target candidates; "
                  f"tau = {ci(T['value'], 2)} {D['dataset']['unit']}; source calibration n = {f(S['state']['n_source_cal']['mean'], 0)})", "",
                  f"Acceptable share: source candidates {f(T['acceptable_share']['source_cand']['mean'])}, "
                  f"target candidates {f(T['acceptable_share']['target_cand']['mean'])}. "
                  f"Shift check: AUROC {ci(sc['auroc_pvalue'])}; target flagged {ci(sc['target_flagged'])}; "
                  f"source flagged {ci(sc['source_flagged'])}. q_source/IQR_T {f(S['state']['q_source_over_tgt_iqr']['mean'], 2)}, "
                  f"q_target/IQR_T {f(S['state']['q_target_over_tgt_iqr']['mean'], 2)}. "
                  f"RMSE source/target candidates {f(S['rmse']['source_cand']['mean'], 1)}/{f(S['rmse']['target_cand']['mean'], 1)}.", "",
                  "All candidates:", "",
                  "| arm | n accepted | FDR | pooled FDP | uncond. FA | power | measurements | power after meas. | false-reject prop. | miscoverage |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for a in ARMS:
                A = T["arms"][a]["all"]
                pl = T["pooled"][a]["all"]
                L.append(f"| {SHORT[a]} | {ci(A['n_accept'], 1)} | {ci(A['fdp'])} | {f(pl['fdp_pooled'])} | {ci(A['uncond_false_accept'])} | "
                         f"{ci(A['power'])} | {ci(A['measurements'], 1)} | {ci(A['power_after_measurement'])} | "
                         f"{ci(A['false_reject_prop'])} | {f(A['miscoverage']['mean'])} |")
            L += ["", "Design-region subgroups (true target candidates; flagged candidates; target candidates the shift "
                  "check missed). 'Flagged' always means flagged by the shift check, also in the D3/D4 rows, which route "
                  "by true membership instead; each subgroup metric is averaged over the splits in which the subgroup is "
                  "non-empty, so 'pooled FDP' (with its totals) is the reliable rate when accepts are rare:", "",
                  "| arm | target: n acc. | any | FDR | FDP if any | pooled FDP | false acc. | miscov. | flagged: n acc. | any | FDR | FDP if any | miscov. | target not flagged: FA of n | miscov. |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for a in ARMS:
                tt, fl, tu = T["arms"][a]["true_target"], T["arms"][a]["flagged"], T["arms"][a]["target_unflagged"]
                pt = T["pooled"][a]["true_target"]
                L.append(f"| {SHORT[a]} | {f(tt['n_accept']['mean'], 2)} | {f(tt['any_accepted']['mean'], 2)} | {ci(tt['fdp'])} | "
                         f"{f(tt['fdp_if_any']['mean'])} | {f(pt['fdp_pooled'])} ({pt['false_accepts_total']}/{pt['n_accept_total']}) | "
                         f"{f(tt['false_accepts']['mean'], 2)} | {f(tt['miscoverage']['mean'])} | "
                         f"{f(fl['n_accept']['mean'], 2)} | {f(fl['any_accepted']['mean'], 2)} | {ci(fl['fdp'])} | "
                         f"{f(fl['fdp_if_any']['mean'])} | {f(fl['miscoverage']['mean'])} | "
                         f"{f(tu['false_accepts']['mean'], 2)} of {f(tu['n']['mean'], 1)} | {f(tu['miscoverage']['mean'])} |")
            L.append("")

    # ---- sensitivity table
    L += ["## Sensitivity: specification threshold (RF and HistGB)", "",
          "FDR (all candidates) / FDR (true target candidates) / measurements, mean over splits. "
          + "; ".join(f"`{k}` = {v}" for k, v in TAU_DOC.items()) + ".", "",
          "| dataset | model | tau | A0 | A1 | A2 | A3 | A4 | A4s | A5 | D3 | D4 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for dn, D in results.items():
        for model in MODELS:
            for tn in TAUS:
                T = D["models"][model]["tau"][tn]["arms"]
                cells = [f"{f(T[a]['all']['fdp']['mean'], 2)}/{f(T[a]['true_target']['fdp']['mean'], 2)}/"
                         f"{f(T[a]['all']['measurements']['mean'], 0)}" for a in ARMS]
                L.append(f"| {dn} | {model} | {tn} | " + " | ".join(cells) + " |")
    L += ["", "## Self-test of trustlayer.py (synthetic, assumptions true by construction)", "",
          "```", str(selftest), "```", ""]
    open(path or f"{RES}/R7_pipeline.md", "w", encoding="utf-8").write("\n".join(L))


def main(n_jobs=6):
    t0 = time.time()
    # the layer's quantile must equal the protocol's
    rng = np.random.default_rng(0)
    for n in [1, 5, 8, 9, 10, 13, 30, 200]:
        s = rng.normal(size=n)
        assert TLm.conformal_quantile(s, ALPHA) == conformal_q(s, ALPHA) or (
            np.isinf(TLm.conformal_quantile(s, ALPHA)) and np.isinf(conformal_q(s, ALPHA)))
    selftest = TLm._selftest()
    print("selftest", selftest, flush=True)
    results = {}
    for ds in design_shift_datasets():
        D = {"dataset": ds.summary(), "m": headline_m(ds), "models": {}}
        for model in MODELS:
            rows = Parallel(n_jobs=n_jobs)(delayed(one_split)(ds, s, model) for s in SEEDS)
            S = summarise_model(rows)
            D["models"][model] = S
            T = S["tau"][HEAD]["arms"]
            print(f"{ds.name:12s} {model:6s} AUROC {S['shift_check']['auroc_pvalue']['mean']:.2f} "
                  f"tflag {S['shift_check']['target_flagged']['mean']:.2f} sflag {S['shift_check']['source_flagged']['mean']:.3f} | "
                  + " | ".join(f"{a.split('_')[0]} FDR {T[a]['all']['fdp']['mean']:.2f}/T {T[a]['true_target']['fdp']['mean'] if T[a]['true_target']['fdp']['mean'] is not None else float('nan'):.2f}"
                               f" acc {T[a]['all']['n_accept']['mean']:.0f} meas {T[a]['all']['measurements']['mean']:.0f}"
                               for a in ARMS), flush=True)
        results[ds.name] = D
    runtime = (time.time() - t0) / 60
    meta = {"script": "revision/code/exp_R7_pipeline.py", "library": "revision/code/trustlayer.py",
            "alpha": ALPHA, "q_fdr": Q_FDR, "novelty_k": NOVELTY_K, "novelty_level": NOVELTY_LEVEL,
            "novelty_multiplicity": "none (per-candidate flag p < level); BH variant reported as *_bh",
            "seeds": SEEDS, "models": MODELS, "arms": ARMS, "subsets": SUBSETS, "taus": TAUS,
            "headline_tau": HEAD, "headline_tau_q75": HEAD_Q75, "tau_definitions": TAU_DOC,
            "budget_arms": sorted(BUDGET_ARMS),
            "candidate_pool": "source-test U target-test",
            "recalibration": "first headline_m records of the recalibration pool (never candidates)",
            "fdp_convention": "FDP = 0 when nothing accepted (mean over splits = FDR); fdp_if_any excludes such splits; "
                              "pooled.fdp_pooled = total false accepts / total accepts over the 20 splits",
            "a3_minus_a2": "decisions A3 adds over A2 (A2's accept/reject sets are subsets of A3's, asserted)",
            "a4s_granularity": "p_min = 1/(n_target+1); any non-empty BH selection among n_f flagged candidates has "
                               ">= ceil(n_f p_min / q) members (asserted)",
            "summary": "mean and 2.5-97.5 percentile over 20 grouped splits of one dataset (split-to-split variability)",
            "runtime_min": runtime, "selftest": selftest}
    dump({"meta": meta, "results": results}, "R7_pipeline.json")
    write_md(results, selftest, runtime)
    print(f"done in {runtime:.1f} min", flush=True)


def report():
    """Rebuild the markdown (and refresh the deterministic self-test in the JSON meta) from
    an existing results/R7_pipeline.json without re-running the 20 x 6 x 2 splits."""
    import json
    J = json.load(open(f"{RES}/R7_pipeline.json"))
    selftest = TLm._selftest()
    J["meta"]["selftest"] = selftest
    dump(J, "R7_pipeline.json")
    write_md(J["results"], selftest, J["meta"]["runtime_min"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
