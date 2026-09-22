"""trustlayer.py -- a small, model-agnostic trust layer implemented as ONE decision procedure.

Answers Reviewer 2, comment 4 ("four parallel methods" vs. one framework) and comment 2
(conformal coverage guarantee vs. clinical safety guarantee).

The procedure
-------------
Every step is a method of a single object, and every experimental arm in
exp_R7_pipeline.py is an ablation of that one object (not a separate method):

    fit          wraps any predictor with fit/predict (or a pre-fitted one) and fits a
                 novelty model s(x) = mean Euclidean distance to the ``novelty_k`` nearest
                 standardised TRAIN inputs.
    calibrate    source-stratum records: residual |y - mu(x)|, mu(x), y, novelty s(x) and
                 an optional Mondrian group label (e.g. a missing-measurement pattern).
    shift_check  conformal novelty p-value of each candidate against the calibration
                 novelty scores (Bates et al. 2023, Ann. Stat. 51:149-178); flag = p < level.
    recalibrate  target-stratum records from a small, explicitly budgeted set of target
                 measurements (never the candidates themselves).
    interval     Mondrian split-conformal interval: in-domain candidates use source-stratum
                 quantiles; flagged candidates use target-stratum quantiles if the stratum
                 has enough records, else an infinite interval ('not certified').
    decide       accept / reject / measure / not_certified against a specification
                 y >= tau (direction 'ge') or y <= tau (direction 'le').
    select_fdr   conformal selection (Jin & Candes 2023, JMLR 24(244)) with
                 Benjamini-Hochberg (jointly over all candidates, or per stratum): an
                 accepted set whose false-discovery rate is <= q under exchangeability of
                 each candidate with its stratum's records.

How the four manuscript components map onto the one procedure (and what is evaluated)
-------------------------------------------------------------------------------------
    (1) distribution-free UQ      -> calibrate + interval.
                                      Evaluated end-to-end in exp_R7_pipeline.py.
    (2) shift robustness          -> shift_check + recalibrate (target stratum).
                                      Evaluated end-to-end in exp_R7_pipeline.py.
    (3) cost/risk-aware experiments-> only partly: decide() returns a static 'measure'
                                      label per candidate and the recalibration budget is
                                      a fixed, explicit cost; select_fdr bounds the error
                                      rate of the accepted set. This module does NOT
                                      implement a sequential acquisition policy (which
                                      candidate to measure next, when to stop), so it does
                                      not by itself demonstrate the sequential experiment
                                      design of manuscript Sec. 4.3.
    (4) incomplete modalities     -> Mondrian strata keyed by an arbitrary group label
                                      (e.g. which measurements are missing) and a
                                      NaN-tolerant novelty model; the predictor itself
                                      must accept NaN (e.g. HistGradientBoosting) or be
                                      wrapped in an imputer by the caller. Implemented and
                                      unit-checked (_selftest, part 4: routing and
                                      fallback) but NOT exercised by exp_R7_pipeline.py
                                      (groups=None there); its evaluation belongs to the
                                      missing-measurement-pattern experiment.
So R7 demonstrates components (1)-(2) and the accept/reject/measure/not_certified rule
as one object; it is not evidence for (3) or (4).

What is (and is not) guaranteed -- stated once, used everywhere
---------------------------------------------------------------
G1 (coverage). If a candidate is exchangeable with the calibration records of the
   stratum used for it, P(Y in C(X)) >= 1 - alpha, marginally over calibration and
   candidate (split conformal; Vovk et al. 2005; Lei et al. 2018). Consequently,
   for decide(): P(accept AND specification violated) <= P(Y not in C) <= alpha
   (an 'accept' requires the whole interval to satisfy the specification, so a
   false accept implies miscoverage). This does NOT bound the false-accept rate
   AMONG ACCEPTED candidates, P(violated | accepted) = P(accept AND violated) /
   P(accept), which can exceed alpha whenever acceptances are rare or concentrated
   where the residual distribution is worse than average (see _selftest, part 3).
G2 (FDR). select_fdr() bounds E[#false accepts / max(#accepted, 1)] <= q if each
   candidate is exchangeable with the calibration records of its stratum
   (Jin & Candes 2023). With scope='joint' the bound is for the accepted set as a
   whole (not for a subgroup such as the shifted candidates); with scope='stratum' it
   is per stratum (not for the union). Routing candidates to strata with the novelty
   flag while target records are taken from the whole target pool makes that
   exchangeability approximate, not exact; the experiment reports the realised FDP.
G3 (shift check). For a candidate exchangeable with the calibration inputs,
   P(p <= t) <= t for all t (Bates et al. 2023), so an in-domain candidate is flagged
   with probability <= novelty_level. There is NO guarantee that a shifted candidate
   is flagged (power depends on how far the shift moves the inputs).
Correct routing is not sufficient either: a shifted candidate routed to the target
   stratum is covered by G1 (a bound on P(accept AND violate)) and, under joint BH, by
   G2 for the accepted set as a whole; neither bounds the error rate among the accepted
   shifted candidates. Only select_fdr(scope='stratum') targets that subgroup, and with
   n target records its p-values cannot fall below 1/(n+1), so BH inside the stratum
   selects either nothing or at least ceil(n_stratum / ((n+1) q)) candidates.
None of G1-G3 is a clinical safety or reliability guarantee. They are statements about
a measured property under a statistical exchangeability assumption; safety of a
biomedical material additionally requires biological evidence (ISO 10993
biocompatibility, in-vitro and in-vivo testing), which no conformal statement replaces.
"""
from __future__ import annotations

from collections import Counter, OrderedDict

import numpy as np

ACCEPT, REJECT, MEASURE, NOT_CERTIFIED = "accept", "reject", "measure", "not_certified"
SOURCE, TARGET = "source", "target"
_ALL = "__all__"          # sentinel label: "no Mondrian group given"


# =============================================================== primitives
def conformal_quantile(scores, alpha):
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score,
    +inf when that rank exceeds n. Identical to common.conformal_q (asserted in the
    experiment script) so that the trust layer follows the binding protocol while
    remaining importable on its own."""
    s = np.asarray(scores, float)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(s)[k - 1])


def min_calibration_size(alpha):
    """Smallest number of calibration records for which conformal_quantile is finite.
    Equals ceil(1/alpha) - 1 (= 9 at alpha = 0.10). Computed with the same arithmetic as
    conformal_quantile so the threshold and the quantile can never disagree."""
    n = 1
    while not np.isfinite(conformal_quantile(np.zeros(n), alpha)):
        n += 1
    return n


def conformal_novelty_pvalues(cal_scores, test_scores):
    """Conformal outlier p-values (Bates, Candes, Lei, Romano & Sesia 2023, Ann. Stat.):
        p_j = (1 + #{i : s_i >= s_j}) / (n + 1),   larger s = more novel.
    Deterministic (conservative, ties counted against the candidate) version of their
    marginal p-value; super-uniform for a candidate exchangeable with the calibration
    inputs."""
    c = np.sort(np.asarray(cal_scores, float))
    t = np.asarray(test_scores, float)
    n = len(c)
    ge = n - np.searchsorted(c, t, side="left")
    return (1.0 + ge) / (n + 1.0)


def benjamini_hochberg(p, q):
    """Benjamini-Hochberg step-up at level q; returns a boolean selection mask."""
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    ps = np.sort(p)
    below = ps <= q * np.arange(1, m + 1) / m
    if not below.any():
        return np.zeros(m, bool)
    k = int(np.max(np.nonzero(below)[0])) + 1
    return p <= q * k / m


def selection_pvalues_closed_form(mu_cal, y_cal, mu_test, tau, direction="ge"):
    """Conformal-selection p-values (Jin & Candes 2023) in closed form.

    direction 'ge' (acceptable iff y >= tau):
        p_j = (1 + #{i : y_i <= tau and mu_i >= mu_j}) / (n + 1)
    direction 'le' (acceptable iff y <= tau): the mirror image
        p_j = (1 + #{i : y_i >= tau and mu_i <= mu_j}) / (n + 1)

    Derivation ('ge'). Jin & Candes use a monotone score V(x, y) (V(x, y) <= V(x, y') for
    y <= y'); we take their 'clipped' score V(x, y) = M * 1{y > tau} + tau - mu(x), with
    M larger than the range of mu so that M + tau - mu_i > tau - mu_j for all i, j.
    Calibration scores V_i = V(X_i, Y_i); the candidate score is evaluated at the
    boundary of the null, V^_j = V(X_j, tau) = tau - mu_j. Their p-value with the
    tie-randomisation U_j set to 1 (deterministic, conservative) is
        p_j = (1 + sum_i 1{V_i <= V^_j}) / (n + 1).
    If y_i > tau, V_i = M + tau - mu_i > tau - mu_j, so it never counts. If y_i <= tau,
    V_i <= V^_j  <=>  mu_i >= mu_j. Hence the closed form above. (Verified numerically
    against the generic score in _selftest, part 1.)
    Null hypothesis tested: H_j: Y_j <= tau. A false accept under our specification is
    Y_j < tau, a subset of H_j, so the FDR bound for H_j also bounds the false-accept
    FDR (conservative only through calibration labels exactly equal to tau)."""
    mu_cal = np.asarray(mu_cal, float)
    y_cal = np.asarray(y_cal, float)
    mu_test = np.asarray(mu_test, float)
    n = len(mu_cal)
    if direction == "ge":
        a = np.sort(mu_cal[y_cal <= tau])
        cnt = len(a) - np.searchsorted(a, mu_test, side="left")          # mu_i >= mu_j
    elif direction == "le":
        a = np.sort(mu_cal[y_cal >= tau])
        cnt = np.searchsorted(a, mu_test, side="right")                   # mu_i <= mu_j
    else:
        raise ValueError(direction)
    return (1.0 + cnt) / (n + 1.0)


def selection_pvalues_generic(mu_cal, y_cal, mu_test, tau, direction="ge"):
    """Same p-values computed literally from the paper's clipped score (for verification)."""
    mu_cal = np.asarray(mu_cal, float)
    y_cal = np.asarray(y_cal, float)
    mu_test = np.asarray(mu_test, float)
    sgn = 1.0 if direction == "ge" else -1.0          # 'le' = 'ge' on (-y, -tau, -mu)
    mc, yc, mt, t = sgn * mu_cal, sgn * y_cal, sgn * mu_test, sgn * tau
    M = 2.0 * (np.max(np.abs(np.r_[mc, mt])) + abs(t)) + 1.0
    V = M * (yc > t) + t - mc
    Vhat = M * (t > t) + t - mt
    return (1.0 + (V[None, :] <= Vhat[:, None]).sum(1)) / (len(mc) + 1.0)


# =============================================================== novelty model
class KNNNovelty:
    """s(x) = mean Euclidean distance from x to its k nearest TRAIN inputs after
    standardising each feature with the training mean and standard deviation
    (zero-variance features keep scale 1).

    Missing values: standardised with nan-aware moments and then set to 0, i.e. imputed
    at the training mean, so a missing coordinate contributes no distance. Scores of
    records with different missingness patterns are therefore not comparable, which is
    why shift_check() can compare a candidate only with calibration records of the same
    Mondrian group.

    Exchangeability: training inputs are never scored (a training point would be its own
    nearest neighbour); calibration records and candidates are both outside the training
    set, so under no shift their scores are exchangeable, as the conformal p-value needs."""

    def __init__(self, k=10):
        self.k = int(k)

    def fit(self, X):
        from sklearn.neighbors import NearestNeighbors
        X = np.asarray(X, float)
        with np.errstate(all="ignore"):
            mean = np.nanmean(X, 0)
            sd = np.nanstd(X, 0)
        mean[~np.isfinite(mean)] = 0.0
        sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
        self.mean_, self.scale_ = mean, sd
        self.k_ = min(self.k, len(X))
        self.nn_ = NearestNeighbors(n_neighbors=self.k_).fit(self._z(X))
        return self

    def _z(self, X):
        Z = (np.asarray(X, float) - self.mean_) / self.scale_
        return np.where(np.isfinite(Z), Z, 0.0)

    def score(self, X):
        d, _ = self.nn_.kneighbors(self._z(X), n_neighbors=self.k_)
        return d.mean(1)


# =============================================================== records
def _labels(groups, n):
    """Object array of hashable group labels (lists/arrays become tuples); _ALL if None."""
    out = np.empty(n, dtype=object)
    if groups is None:
        out[:] = _ALL
        return out
    groups = list(groups)
    if len(groups) != n:
        raise ValueError("groups must have one label per record")
    for i, g in enumerate(groups):
        out[i] = tuple(np.asarray(g).ravel().tolist()) if isinstance(g, (list, tuple, np.ndarray)) else g
    return out


class _Stratum:
    """Calibration records of one stratum (source or target)."""

    def __init__(self, name):
        self.name = name
        self.resid = np.empty(0)
        self.mu = np.empty(0)
        self.y = np.empty(0)
        self.nov = np.empty(0)
        self.groups = np.empty(0, dtype=object)

    def __len__(self):
        return len(self.resid)

    def add(self, resid, mu, y, nov, groups):
        self.resid = np.r_[self.resid, resid]
        self.mu = np.r_[self.mu, mu]
        self.y = np.r_[self.y, y]
        self.nov = np.r_[self.nov, nov]
        self.groups = np.concatenate([self.groups, groups])

    def index(self, g):
        if g is _ALL or (isinstance(g, str) and g == _ALL):
            return np.arange(len(self))
        return np.array([i for i, h in enumerate(self.groups) if h == g], dtype=int)


# =============================================================== the layer
class TrustLayer:
    """One decision procedure: predictor -> conformal interval -> shift check ->
    (budgeted) target recalibration -> accept / reject / measure / not_certified,
    with an optional FDR-controlled accept set.

    Parameters
    ----------
    predictor : any object with fit(X, y) and predict(X) (scikit-learn style). It is
        never modified by the layer beyond fit(); use fit(..., refit=False) to wrap an
        already-trained model without retraining it.
    alpha : miscoverage level of the intervals (protocol default 0.10).
    novelty_k : number of nearest training inputs in the novelty score (10; fixed a
        priori, not tuned).
    novelty_level : a candidate is flagged when its conformal novelty p-value is
        < novelty_level (0.05; fixed a priori).
    multiplicity : 'none' (default) or 'bh'. See shift_check().
    novelty_transform : optional callable mapping raw inputs to the space in which
        novelty is measured (e.g. a network embedding); identity if None.

    Stratum sizes. A stratum (or Mondrian group within it) with fewer than
    n_min = ceil(1/alpha) - 1 records (9 at alpha = 0.10) cannot produce a finite
    split-conformal quantile. Such a group falls back to its parent stratum (all records
    of the stratum, any group); if the parent is also too small the interval is
    infinite and the decision is 'not_certified'. The guarantee after a fallback is
    marginal over the parent stratum, not conditional on the group.
    """

    def __init__(self, predictor, alpha=0.1, novelty_k=10, novelty_level=0.05,
                 multiplicity="none", novelty_transform=None):
        self.predictor = predictor
        self.alpha = float(alpha)
        self.novelty_k = int(novelty_k)
        self.novelty_level = float(novelty_level)
        self.multiplicity = multiplicity
        self.novelty_transform = novelty_transform
        self.n_min = min_calibration_size(self.alpha)
        # smallest calibration size at which a novelty p-value can fall below the level
        self.n_min_novelty = 1
        while 1.0 / (self.n_min_novelty + 1) >= self.novelty_level:
            self.n_min_novelty += 1
        self._strata = {SOURCE: _Stratum(SOURCE), TARGET: _Stratum(TARGET)}
        self.novelty_ = None

    # ------------------------------------------------------------------ fitting
    def _nov_in(self, X):
        return self.novelty_transform(X) if self.novelty_transform is not None else X

    def fit(self, X_train, y_train, refit=True):
        """Fit the predictor (unless refit=False) and the novelty model on the TRAIN
        inputs only. Calibration and candidate records must be disjoint from X_train."""
        if refit:
            self.predictor.fit(X_train, y_train)
        self.novelty_ = KNNNovelty(self.novelty_k).fit(self._nov_in(X_train))
        return self

    def predict(self, X):
        return np.asarray(self.predictor.predict(X), float).ravel()

    def novelty_score(self, X):
        return self.novelty_.score(self._nov_in(X))

    def _records(self, X, y, groups):
        y = np.asarray(y, float).ravel()
        mu = self.predict(X)
        return np.abs(y - mu), mu, y, self.novelty_score(X), _labels(groups, len(y))

    def calibrate(self, X_cal, y_cal, groups=None):
        """(Re)set the source stratum from held-out source calibration records."""
        self._strata[SOURCE] = _Stratum(SOURCE)
        self._strata[SOURCE].add(*self._records(X_cal, y_cal, groups))
        return self

    def recalibrate(self, X_t, y_t, groups=None, reset=False):
        """Add target-stratum records (measured target-domain samples). These must be a
        separately budgeted draw, never the candidates being decided. reset=True clears
        earlier target records first."""
        if reset:
            self.reset_target()
        self._strata[TARGET].add(*self._records(X_t, y_t, groups))
        return self

    def reset_target(self):
        self._strata[TARGET] = _Stratum(TARGET)
        return self

    def state(self):
        """Stratum sizes and pooled quantiles (for reporting)."""
        out = {}
        for k, s in self._strata.items():
            out[k] = {"n": int(len(s)),
                      "q": conformal_quantile(s.resid, self.alpha) if len(s) >= self.n_min else float("inf"),
                      "groups": dict(Counter(str(g) for g in s.groups))}
        return out

    # ------------------------------------------------------------------ shift check
    def shift_check(self, X, groups=None, level=None, multiplicity=None):
        """Conformal novelty p-values of candidates against the SOURCE calibration novelty
        scores, and flags.

        Mondrian novelty: if group labels are given and a group has at least
        n_min_novelty calibration records (the smallest n for which 1/(n+1) < level,
        i.e. 20 at level 0.05), a candidate is compared only with calibration records of
        its own group (needed when missingness changes the scale of the score); else with
        all calibration records.

        Multiplicity. Default 'none': each candidate is a separate routing decision and is
        flagged iff p < level. Then P(flag | candidate exchangeable with calibration)
        <= level for every candidate individually (G3). We deliberately do not correct for
        the number of candidates: a false flag only routes an in-domain candidate to the
        target stratum or to measurement (it costs a measurement but does not weaken any
        other candidate's guarantee), whereas a missed shifted candidate is decided with
        source quantiles, the dangerous error; a multiplicity correction would trade the
        cheap error for the dangerous one. 'bh' applies Benjamini-Hochberg at the level to
        the candidate batch (FDR control of the flagged set; valid because conformal
        outlier p-values are PRDS, Bates et al. 2023) and is offered for comparison."""
        S = self._strata[SOURCE]
        if len(S) == 0:
            raise RuntimeError("calibrate() before shift_check()")
        level = self.novelty_level if level is None else level
        multiplicity = self.multiplicity if multiplicity is None else multiplicity
        s = self.novelty_score(X)
        g = _labels(groups, len(s))
        p = np.empty(len(s))
        for lab in OrderedDict.fromkeys(g.tolist()):
            sel = np.array([h == lab for h in g])
            idx = S.index(lab)
            ref = S.nov[idx] if len(idx) >= self.n_min_novelty else S.nov
            p[sel] = conformal_novelty_pvalues(ref, s[sel])
        if multiplicity == "bh":
            flag = benjamini_hochberg(p, level)
        elif multiplicity == "none":
            flag = p < level
        else:
            raise ValueError(multiplicity)
        return p, flag

    # ------------------------------------------------------------------ routing
    def _route(self, flag, g):
        """Stratum records used for one candidate: (index array or None, label).
        in-domain -> source; flagged -> target (never source). Group -> parent -> none."""
        S = self._strata[TARGET if flag else SOURCE]
        if not (isinstance(g, str) and g == _ALL):
            idx = S.index(g)
            if len(idx) >= self.n_min:
                return idx, f"{S.name}:{g}"
        if len(S) >= self.n_min:
            return np.arange(len(S)), f"{S.name}:all"
        return None, "none"

    def _flags(self, X, flags, groups):
        if flags is None:
            return self.shift_check(X, groups)[1]
        f = np.asarray(flags, bool).ravel()
        return f

    def interval(self, X, flags=None, groups=None, return_details=False):
        """Mondrian split-conformal intervals mu(x) -/+ q.
        flags: boolean array (True = shift-flagged). None -> computed by shift_check().
        Passing all-False flags gives plain source-calibrated split conformal (no shift
        check); passing the shift-check flags before any recalibrate() gives
        'not certified' (infinite) intervals for every flagged candidate."""
        mu = self.predict(X)
        n = len(mu)
        f = self._flags(X, flags, groups)
        g = _labels(groups, n)
        q = np.full(n, np.inf)
        used = np.empty(n, dtype=object)
        cache = {}
        for j in range(n):
            key = (bool(f[j]), g[j])
            if key not in cache:
                idx, lab = self._route(*key)
                S = self._strata[TARGET if key[0] else SOURCE]
                cache[key] = (conformal_quantile(S.resid[idx], self.alpha) if idx is not None else np.inf, lab)
            q[j], used[j] = cache[key]
        lo, hi = mu - q, mu + q
        if return_details:
            return lo, hi, {"mu": mu, "q": q, "flag": f, "stratum": used}
        return lo, hi

    def decide(self, X, tau, direction="ge", flags=None, groups=None, return_details=False):
        """Per-candidate decision against the specification (acceptable iff y >= tau for
        direction 'ge', y <= tau for 'le'):
            not_certified  interval infinite (flagged and no usable target stratum);
            accept         the whole interval satisfies the specification;
            reject         the whole interval violates it;
            measure        the interval straddles tau (decide by experiment).
        The interval is the two-sided 1 - alpha interval (the layer's single uncertainty
        object); a one-sided bound would accept more at the same alpha but is not used so
        that the same interval serves every decision and every reported coverage.
        Guarantee: P(accept AND violated) <= alpha for exchangeable candidates (G1); no
        bound on the violation rate among accepted candidates."""
        lo, hi, det = self.interval(X, flags, groups, return_details=True)
        fin = np.isfinite(lo) & np.isfinite(hi)
        if direction == "ge":
            acc, rej = fin & (lo >= tau), fin & (hi < tau)
        elif direction == "le":
            acc, rej = fin & (hi <= tau), fin & (lo > tau)
        else:
            raise ValueError(direction)
        dec = np.full(len(lo), MEASURE, dtype=object)
        dec[acc], dec[rej], dec[~fin] = ACCEPT, REJECT, NOT_CERTIFIED
        dec = dec.astype(str)
        if return_details:
            det.update(lo=lo, hi=hi)
            return dec, det
        return dec

    # ------------------------------------------------------------------ FDR selection
    def selection_pvalues(self, X, tau, direction="ge", flags=None, groups=None, return_strata=False):
        """Conformal-selection p-values; each candidate is compared with the records of
        the stratum it is routed to (same routing and fallback as interval()); a
        candidate with no usable stratum gets p = 1 (never selected)."""
        mu = self.predict(X)
        n = len(mu)
        f = self._flags(X, flags, groups)
        g = _labels(groups, n)
        p = np.ones(n)
        used = np.empty(n, dtype=object)
        keys = OrderedDict()
        for j in range(n):
            keys.setdefault((bool(f[j]), g[j]), []).append(j)
        for key, js in keys.items():
            idx, lab = self._route(*key)
            js = np.asarray(js)
            used[js] = lab
            if idx is None:
                continue
            S = self._strata[TARGET if key[0] else SOURCE]
            p[js] = selection_pvalues_closed_form(S.mu[idx], S.y[idx], mu[js], tau, direction)
        return (p, used) if return_strata else p

    def select_fdr(self, X, tau, q=0.1, direction="ge", flags=None, groups=None, scope="joint",
                   return_pvalues=False):
        """Conformal selection (Jin & Candes 2023) with Benjamini-Hochberg at level q, using
        stratum-wise conformal p-values. Returns a boolean mask of selected (accepted)
        candidates.

        scope='joint' (default): one BH over ALL candidates, so one q governs the whole
        accepted set. Guarantee (G2): FDR = E[#{selected, spec violated} / max(#selected, 1)]
        <= q when each candidate is exchangeable with the records of its stratum (the
        Jin-Candes argument for a candidate in one stratum conditions on the other strata,
        whose p-values do not depend on it). This bounds the error of the accepted set as a
        whole, NOT of a subgroup: if the accepted set is dominated by in-domain candidates,
        the false-accept proportion among the few accepted shifted candidates can exceed q.

        scope='stratum': a separate BH at level q inside each routed stratum (e.g. source
        vs target, or per Mondrian group). Bounds the FDR within each stratum under the
        same exchangeability, but not the FDR of the union (with k strata under the global
        null the union's FDR can approach k*q).

        Caveat for both: the novelty routing (candidates routed by flag, target records
        taken from the whole target pool) is not symmetric between candidates and target
        records, so the exchangeability is an approximation; report the realised FDP."""
        p, used = self.selection_pvalues(X, tau, direction, flags, groups, return_strata=True)
        if scope == "joint":
            sel = benjamini_hochberg(p, q)
        elif scope == "stratum":
            sel = np.zeros(len(p), bool)
            for lab in OrderedDict.fromkeys(used.tolist()):
                js = np.array([u == lab for u in used])
                sel[js] = benjamini_hochberg(p[js], q)
        else:
            raise ValueError(scope)
        return (sel, p) if return_pvalues else sel


# =============================================================== self-test
class _Oracle:
    """Pre-fitted predictor mu(x) = 10 x_0 (used with refit=False)."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return 10.0 * np.asarray(X)[:, 0]


def _selftest(reps=300, seed=0):
    """Numerical checks of the claims in the docstrings (all on synthetic, exchangeable
    data where the assumptions hold by construction):
      1. closed-form selection p-values == the paper's generic clipped-score p-values
         (random data with ties, both directions);
      2. marginal coverage >= 1 - alpha and novelty-flag rate <= level for in-domain
         candidates; flag rate for shifted candidates (power, no guarantee);
      3. the coverage-vs-FDR gap: in data where 10 % of inputs (x_0 >= 0.9) fail
         catastrophically with probability 0.3, interval-based 'accept' keeps
         P(accept AND violate) <= alpha but its false-accept rate AMONG accepted is far
         above alpha; conformal selection keeps FDR <= q;
      3b. homoscedastic exchangeable data: conformal-selection FDR is close to q (the bound
         is tight), while interval-accept is very conservative there;
      4. Mondrian routing: a group with < n_min records falls back to its parent."""
    rng = np.random.default_rng(seed)
    out = {}
    # ---- 1
    maxdiff = 0.0
    for _ in range(200):
        n, m = rng.integers(5, 60), rng.integers(1, 40)
        mu_c = np.round(rng.normal(size=n), 1); y_c = np.round(mu_c + rng.normal(size=n), 1)
        mu_t = np.round(rng.normal(size=m), 1); tau = float(np.round(rng.normal(), 1))
        for d in ("ge", "le"):
            a = selection_pvalues_closed_form(mu_c, y_c, mu_t, tau, d)
            b = selection_pvalues_generic(mu_c, y_c, mu_t, tau, d)
            maxdiff = max(maxdiff, float(np.max(np.abs(a - b))))
    out["selection_pvalue_closed_vs_generic_maxabsdiff"] = maxdiff

    # ---- 2 and 3
    def draw(n, shifted=False):
        X = rng.uniform(0, 1, size=(n, 2))
        if shifted:
            X[:, 1] = rng.uniform(1.5, 2.5, size=n)
        fail = (X[:, 0] >= 0.9) & (rng.uniform(size=n) < 0.3)
        y = 10 * X[:, 0] + rng.normal(0, 0.2, n) - 5.0 * fail
        return X, y

    tau, qfdr = 8.5, 0.1
    cov, flag_in, flag_out, fa_int, fdp_int, n_int, fdp_sel, n_sel, fa_uncond_sel = ([] for _ in range(9))
    for _ in range(reps):
        tl = TrustLayer(_Oracle(), alpha=0.1)
        Xtr, ytr = draw(400); tl.fit(Xtr, ytr, refit=False)
        Xc, yc = draw(400); tl.calibrate(Xc, yc)
        Xt, yt = draw(400)
        Xs, _ = draw(200, shifted=True)
        lo, hi = tl.interval(Xt, flags=np.zeros(len(yt), bool))
        cov.append(np.mean((yt >= lo) & (yt <= hi)))
        flag_in.append(tl.shift_check(Xt)[1].mean())
        flag_out.append(tl.shift_check(Xs)[1].mean())
        dec = tl.decide(Xt, tau, flags=np.zeros(len(yt), bool))
        acc = dec == ACCEPT; bad = yt < tau
        fa_int.append(np.mean(acc & bad)); n_int.append(acc.sum())
        fdp_int.append((acc & bad).sum() / max(acc.sum(), 1))
        sel = tl.select_fdr(Xt, tau, qfdr, flags=np.zeros(len(yt), bool))
        fdp_sel.append((sel & bad).sum() / max(sel.sum(), 1)); n_sel.append(sel.sum())
        fa_uncond_sel.append(np.mean(sel & bad))
    out["exchangeable_coverage_mean"] = float(np.mean(cov))
    out["novelty_flag_rate_in_domain_mean"] = float(np.mean(flag_in))
    out["novelty_flag_rate_shifted_mean"] = float(np.mean(flag_out))
    out["gap_demo"] = {
        "setup": "x~U(0,1)^2, y=10x0+N(0,0.2^2), minus 5 with prob 0.3 if x0>=0.9; oracle mu=10x0; "
                 "n_cal=400, 400 candidates, spec y>=8.5, alpha=0.1, q=0.1, reps=%d" % reps,
        "interval_accept": {"P(accept and violate)": float(np.mean(fa_int)),
                            "FDR_among_accepted": float(np.mean(fdp_int)),
                            "mean_n_accepted": float(np.mean(n_int))},
        "conformal_selection": {"P(accept and violate)": float(np.mean(fa_uncond_sel)),
                                "FDR_among_accepted": float(np.mean(fdp_sel)),
                                "mean_n_accepted": float(np.mean(n_sel))}}

    # ---- 3b: homoscedastic exchangeable data -- conformal-selection FDR should be close to
    # q (not pi0*q: the p-values are valid jointly with the random null event, Jin & Candes)
    fd, ns, fdi = [], [], []
    for _ in range(reps):
        tl = TrustLayer(_Oracle(), alpha=0.1)
        X = rng.uniform(0, 1, size=(900, 2)); yy = 10 * X[:, 0] + rng.normal(0, 1, 900)
        tl.fit(X[:300], yy[:300], refit=False); tl.calibrate(X[300:600], yy[300:600])
        z = np.zeros(300, bool); yt = yy[600:]
        sel = tl.select_fdr(X[600:], 5.0, qfdr, flags=z)
        fd.append((sel & (yt < 5.0)).sum() / max(sel.sum(), 1)); ns.append(sel.sum())
        acc = tl.decide(X[600:], 5.0, flags=z) == ACCEPT
        fdi.append((acc & (yt < 5.0)).sum() / max(acc.sum(), 1))
    out["homoscedastic_demo"] = {
        "setup": "y=10x0+N(0,1), oracle mu, n_cal=300, 300 candidates, spec y>=5, q=0.1, reps=%d" % reps,
        "selection_FDR": float(np.mean(fd)), "selection_mean_n": float(np.mean(ns)),
        "interval_accept_FDR": float(np.mean(fdi))}

    # ---- 4
    tl = TrustLayer(_Oracle(), alpha=0.1)
    Xtr, ytr = draw(200); tl.fit(Xtr, ytr, refit=False)
    Xc, yc = draw(200); gc = np.where(np.arange(200) < 5, "rare", "common")
    tl.calibrate(Xc, yc, groups=gc)
    _, _, det = tl.interval(Xc[:3], flags=np.zeros(3, bool), groups=["rare", "common", "unseen"], return_details=True)
    _, _, det2 = tl.interval(Xc[:1], flags=np.ones(1, bool), return_details=True)
    out["mondrian_routing"] = {"rare(<n_min)": det["stratum"][0], "common": det["stratum"][1],
                               "unseen": det["stratum"][2], "flagged_no_target": det2["stratum"][0],
                               "flagged_no_target_q": float(det2["q"][0]), "n_min": tl.n_min,
                               "n_min_novelty": tl.n_min_novelty}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(_selftest(), indent=1, default=str))
