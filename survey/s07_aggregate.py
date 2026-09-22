"""Aggregate the BMEAT Figure-2 systematic survey.

Reads revision/survey/sample_A*.json, codes_A_A*.json, codes_B_A*.json and the
cached identification counts, then writes survey_results.json,
survey_results.md and survey_coded_papers.csv under revision/survey/.

Protocol: revision/audit/A6_survey_protocol.md (frozen).
"""
import csv
import json
import math
import os
import sys
from collections import Counter, OrderedDict, defaultdict

import numpy as np

SURVEY = r"D:\claude\pajinsen\BMEAT\revision\survey"
AREAS = ["A1_imaging", "A2_property", "A3_sensors", "A4_tissue", "A5_autonomous"]
SHORT = {"A1_imaging": "A1", "A2_property": "A2", "A3_sensors": "A3",
         "A4_tissue": "A4", "A5_autonomous": "A5"}
LABEL = {"A1_imaging": "A1 imaging-based characterisation",
         "A2_property": "A2 property prediction / inverse design",
         "A3_sensors": "A3 biosensors and bioelectronics",
         "A4_tissue": "A4 tissue engineering, scaffolds, hydrogels",
         "A5_autonomous": "A5 autonomous and sequential discovery"}
FINAL_VERSION = {"A1_imaging": "v1", "A2_property": "v1", "A3_sensors": "v2",
                 "A4_tissue": "v2", "A5_autonomous": "v1"}
BOOT = 2000
SEED = 20260919

# ---------------------------------------------------------------- adjudication
# Decided by the adjudicator (this agent) after re-reading the cached JATS full
# text of every paper below. "final" is the code that enters the final data set
# for that paper in EVERY area in which it was sampled.
ADJUDICATION = [
    {
        "pmcid": "PMC10161767", "areas": ["A1_imaging"], "item": "U",
        "coder_A": 0, "coder_B": 1, "final": 1, "type": "A_vs_B",
        "decision": "U1",
        "reason": ("Coder B is right. The Results text of the paper states "
                   "'Table 1 summarizes the performance metrics of various algorithms under two "
                   "phases together with the corresponding standard deviations in parentheses', "
                   "and the Methods state the models are 'trained five times' over random weight "
                   "initialisations and dataset splits. That is the SD of a performance metric over "
                   "repeated runs/splits, which section 7 lists under U1. Coder A applied the 'trained "
                   "5 times and averaged with no spread' exclusion but missed the Table 1 spread, which "
                   "sits in <floats-group> and was not reached by the regex screen."),
    },
    {
        "pmcid": "PMC12796348", "areas": ["A1_imaging"], "item": "EP",
        "coder_A": 0, "coder_B": 1, "final": 0, "type": "A_vs_B",
        "decision": "EP0",
        "reason": ("Coder A is right. The paper itself calls the swine work a 'retrospective validation "
                   "study'; the specimens were collected and frozen before the model existed, and the "
                   "Discussion lists 'larger-scale prospective swine studies' as future work. The reported "
                   "concordance (30/32 samples; Pearson r = 0.87) is between the MIDAS assay readout and "
                   "reference assays, not between a model prediction and a newly produced measurement. "
                   "Section 7 requires samples produced or measured AFTER the model was fixed."),
    },
    {
        "pmcid": "PMC11538249", "areas": ["A1_imaging"], "item": "S",
        "coder_A": 1, "coder_B": 0, "final": 1, "type": "A_vs_B",
        "decision": "S1 (natural)",
        "reason": ("Coder A is right on the frozen codebook. Section 7 lists 'group-, cluster- or "
                   "scaffold-based split' under the S1 natural sub-code without any additional "
                   "requirement that the paper frame the grouping factor as the shift under study. The "
                   "Methods state 'the data obtained from the same object was used exclusively for either "
                   "training or testing, not both ... the objects used for training were different from "
                   "those used for validation and testing', i.e. an animal-level grouped split, and "
                   "segmentation performance on the held-out animals is reported (Tables 1-2). Coder B "
                   "applied a stricter rule that the frozen protocol does not contain."),
    },
    {
        "pmcid": "PMC9513834", "areas": ["A2_property"], "item": "EP",
        "coder_A": 1, "coder_B": 0, "final": 1, "type": "A_vs_B",
        "decision": "EP1",
        "reason": ("Coder A is right. Section 7 gives 'synthesising the predicted optimum' as the leading "
                   "EP example. The Results state 'Using this design as a guide, we synthesized aligned "
                   "nanofibers using a standard rotating drum electrospinning method, producing nanofibers "
                   "with four distinct diameters (including the diameters that were projected to be ideal "
                   "by our radar charts)', and the measured axonal alignment at ~200 nm 'fits the estimation "
                   "in the radar charts'. New material was made on a different platform after the "
                   "GPR-derived maps were fixed and compared with them. Coder B's stated reason addresses "
                   "U and E (hold-out CV, one lab), not the EP definition. The comparison is qualitative, "
                   "which section 7 does not forbid; confidence is medium."),
    },
    {
        "pmcid": "PMC11170757", "areas": ["A1_imaging", "A4_tissue"], "item": "EP",
        "coder_A": 1, "coder_B": 0, "final": 0, "type": "A_vs_B",
        "decision": "EP0",
        "reason": ("Coder B is right. The Introduction gives the chronology explicitly: (i) seven PolyHIPE "
                   "groups were fabricated, (ii) four users blind-quantified their pores and windows, "
                   "(iii) only afterwards was the Pore D2 YOLOv5 detector developed. The test-group "
                   "scaffolds and their manual ground truth therefore pre-date the model, so section 7's "
                   "requirement of samples 'produced or measured AFTER the model was fixed' is not met. "
                   "The comparison is automated versus manual measurement on the same SEM images, and the "
                   "Conclusions state 'it is not applicable to assume one way as a gold standard'. The "
                   "single-verifier pilot (protocol section 10.2) also recorded EP=0 for this paper."),
    },
    {
        "pmcid": "PMC7076403", "areas": ["A2_property", "A4_tissue"], "item": "EP",
        "coder_A": "1 in A2, 0 in A4", "coder_B": 1, "final": 1,
        "type": "A_vs_B_and_A_cross_area",
        "decision": "EP1",
        "reason": ("Coder B and coder A's A2 coding are right; coder A's A4 coding was inconsistent with "
                   "its own A2 coding of the same paper. The Results state 'The optimized 3D bioprinted "
                   "scaffold formulation, resulting in the highest factor of response ... was thus "
                   "synthesized incorporating 14% w/v PPF and 16% w/v PF127. The optimized 3D bioprinted "
                   "scaffold displayed a controlled release of simvastatin over a 20-day duration ... with "
                   "significant correlation to the predicted release kinetics ascertained using ANN', and "
                   "Figure 10's legend is 'Correlation of in vitro simvastatin release analysis of the "
                   "optimized 3D bioprinted scaffold with predicted release kinetics using ANN modeling'. "
                   "That is the predicted optimum synthesised and its measured behaviour compared with the "
                   "model prediction. Agreement is reported qualitatively, so confidence is medium."),
    },
    {
        "pmcid": "PMC11921027", "areas": ["A1_imaging", "A4_tissue"], "item": "E",
        "coder_A": "1 in A1, 0 in A4", "coder_B": 0, "final": 0,
        "type": "A_cross_area",
        "decision": "E0",
        "reason": ("Coder A's A4 coding and coder B are right. What the paper calls its 'external data "
                   "sets' are AFM images the authors acquired themselves on their own JPK Nanowizard4; no "
                   "other laboratory, instrument, cohort, site, or public/literature dataset is involved. "
                   "Section 7's E1 exclusions route 'a separately collected batch from the same lab and "
                   "set-up' away from E1. The different collagen source (rat-tail tendon, reformed fibrils) "
                   "is a distribution shift and is already captured as S1 natural, which both coders gave."),
    },
    {
        "pmcid": "PMC8575943", "areas": ["A1_imaging", "A4_tissue"], "item": "EP",
        "coder_A": "1 in A1, 0 in A4", "coder_B": "1 in A1, 0 in A4",
        "final": 0, "type": "cross_area_both_coders",
        "decision": "EP0",
        "reason": ("Both coders coded this paper EP=1 in its A1 record and EP=0 in its A4 record, so the "
                   "paper needed a single adjudicated code. EP=0. The substrate-stiffening data set is part "
                   "of the study's own Fig. 2 experiments; the Results say it 'was collected at a later "
                   "time, under different microscope settings' relative to the training images, but it was "
                   "collected before the model was applied and the model is run on it retrospectively and "
                   "compared with a manual analysis. Moreover the model was not fixed for that set: 'the "
                   "experimental images were normalized by a different set of values than the training set "
                   "... optimizing these normalization parameters until the model best matched the manual "
                   "analysis'. Section 7 requires samples produced or measured after the model was fixed. "
                   "The changed microscope settings are a shift and stay coded S1 natural in both areas."),
    },
]

# Single-coder borderline re-read by the adjudicator (not a disagreement; the
# code is NOT changed, but it is the only U3 in the survey so it is disclosed).
FLAGGED_SINGLE_CODER = [
    {
        "pmcid": "PMC10265106", "areas": ["A4_tissue"], "item": "U",
        "coder_A": 3, "coder_B": None, "final": 3, "upheld": True,
        "reason": ("Only U3 in the whole survey; coder A flagged it borderline with low confidence and it "
                   "is outside coder B's 50% subset, so it is not an adjudicable disagreement. On re-reading, "
                   "the main text does report a coverage-style assessment of the GPR's own predictive "
                   "intervals: 'the rationality of the GPR model in this research was confirmed by comparing "
                   "the confidence interval and the evaluated scores. As demonstrated in Figure S5 "
                   "(Supporting Information), all evaluated scores were within the 95% confidence interval'. "
                   "That is empirical coverage against a nominal level, which section 7 lists under U3, and "
                   "is different from the pilot's U3 false positive (an untested 'well-calibrated' claim). "
                   "Two weaknesses are recorded and the code is left at U3: the supporting plot is in the SI "
                   "(SI_dependency = true) and the main text does not state whether the compared scores are "
                   "the held-out 2.5 w/v% test set or all 40 points. A stricter adjudicator would return U2, "
                   "which would make U3 = 0/153 overall; both figures are reported."),
    },
]


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def kappa(a, b, levels, weights=None):
    """Cohen's kappa (unweighted, or with a weight matrix w[i][j] in [0,1] where
    1 = full agreement). Returns None when chance agreement is 1."""
    n = len(a)
    if n == 0:
        return None
    idx = {v: i for i, v in enumerate(levels)}
    m = np.zeros((len(levels), len(levels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    m /= n
    ra, rb = m.sum(1), m.sum(0)
    if weights is None:
        w = np.eye(len(levels))
    else:
        w = weights
    po = float((m * w).sum())
    pe = float((np.outer(ra, rb) * w).sum())
    if abs(1 - pe) < 1e-12:
        return None
    return (po - pe) / (1 - pe)


def linear_weights(levels):
    k = len(levels)
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            w[i, j] = 1 - abs(i - j) / (k - 1)
    return w


def agreement_block(pairs, levels, weights=None, rng=None, groups=None):
    """pairs: list of (a, b). groups: clustering key per pair for the bootstrap."""
    n = len(pairs)
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    po = sum(1 for x, y in zip(a, b) if x == y) / n if n else None
    k = kappa(a, b, levels, weights)
    out = {"n": n, "percent_agreement": po, "kappa": k}
    if len(levels) == 2:
        c11 = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
        c10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        c01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        c00 = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
        out["table"] = {"both_1": c11, "A1_B0": c10, "A0_B1": c01, "both_0": c00}
        out["pabak"] = 2 * po - 1 if po is not None else None
        out["ppos"] = (2 * c11 / (2 * c11 + c10 + c01)) if (2 * c11 + c10 + c01) else None
        out["pneg"] = (2 * c00 / (2 * c00 + c10 + c01)) if (2 * c00 + c10 + c01) else None
        out["n_positive_A"] = c11 + c10
        out["n_positive_B"] = c11 + c01
    # bootstrap over papers
    if rng is not None and n > 1:
        if groups is None:
            groups = list(range(n))
        by_g = defaultdict(list)
        for pr, g in zip(pairs, groups):
            by_g[g].append(pr)
        keys = list(by_g)
        ks, ps = [], []
        for _ in range(BOOT):
            draw = rng.integers(0, len(keys), len(keys))
            samp = [pr for i in draw for pr in by_g[keys[i]]]
            sa = [p[0] for p in samp]
            sb = [p[1] for p in samp]
            ps.append(sum(1 for x, y in zip(sa, sb) if x == y) / len(samp))
            kv = kappa(sa, sb, levels, weights)
            if kv is not None:
                ks.append(kv)
        out["percent_agreement_ci"] = [float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5))]
        if len(ks) >= 50:
            out["kappa_ci"] = [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))]
            out["kappa_ci_valid_draws"] = len(ks)
        else:
            out["kappa_ci"] = None
            out["kappa_ci_valid_draws"] = len(ks)
    return out


# ------------------------------------------------------------------- data load
def load(path):
    with open(os.path.join(SURVEY, path), encoding="utf-8") as f:
        return json.load(f)


def clist(o):
    return o["codes"] if isinstance(o, dict) else o


def to_int(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        s = v.strip().upper()
        if len(s) == 2 and s[0] in "UESP" and s[1].isdigit():
            return int(s[1])
        if s.startswith("EP") and s[2:].isdigit():
            return int(s[2:])
        if s.isdigit():
            return int(s)
        raise ValueError(v)
    return int(v)


def codes_of(rec):
    return {k: to_int(rec.get(k)) for k in ("U", "E", "EP", "S")}


def main():
    rng = np.random.default_rng(SEED)
    counts = {v: load(os.path.join("cache", f"counts_{v}.json")) for v in ("v1", "v2")}

    samples, A, B = {}, {}, {}
    for a in AREAS:
        samples[a] = load(f"sample_{a}.json")
        A[a] = {r["pmcid"]: r for r in clist(load(f"codes_A_{a}.json"))}
        B[a] = {r["pmcid"]: r for r in clist(load(f"codes_B_{a}.json"))}

    # ---------------------------------------------------------- 1. reliability
    items = OrderedDict()
    items["U_4level"] = ("U", [0, 1, 2, 3], None)
    items["U_4level_linear_weighted"] = ("U", [0, 1, 2, 3], linear_weights([0, 1, 2, 3]))
    items["U_ge1"] = ("U>=1", [0, 1], None)
    items["U_ge2"] = ("U>=2", [0, 1], None)
    items["U_ge3"] = ("U>=3", [0, 1], None)
    items["E1"] = ("E", [0, 1], None)
    items["EP"] = ("EP", [0, 1], None)
    items["S1"] = ("S", [0, 1], None)

    def val(c, key):
        if key == "U":
            return c["U"]
        if key == "U>=1":
            return int(c["U"] >= 1)
        if key == "U>=2":
            return int(c["U"] >= 2)
        if key == "U>=3":
            return int(c["U"] >= 3)
        return int(c[key] == 1)

    overlap_rows = []   # (area, pmcid, codes_A, codes_B)
    for a in AREAS:
        for p in B[a]:
            overlap_rows.append((a, p, codes_of(A[a][p]), codes_of(B[a][p])))

    reliability = {"design": {
        "dual_coded_area_records": len(overlap_rows),
        "dual_coded_unique_papers": len({r[1] for r in overlap_rows}),
        "coded_area_records_total": sum(len(A[a]) for a in AREAS),
        "coded_unique_papers_total": len({p for a in AREAS for p in A[a]}),
        "bootstrap_draws": BOOT, "bootstrap_seed": SEED,
        "bootstrap_unit": "paper (a paper dual-coded in two areas is resampled as one cluster)",
        "note": ("Statistics are computed on the raw coder-A vs coder-B codes, before adjudication. "
                 "Four papers sit in two area subsets, so 85 dual-coded area-records cover 81 unique papers."),
    }, "overall": {}, "by_area": {}}

    for name, (key, levels, w) in items.items():
        pairs = [(val(ca, key), val(cb, key)) for _, _, ca, cb in overlap_rows]
        groups = [p for _, p, _, _ in overlap_rows]
        reliability["overall"][name] = agreement_block(pairs, levels, w, rng, groups)
    for a in AREAS:
        rows = [r for r in overlap_rows if r[0] == a]
        reliability["by_area"][SHORT[a]] = {}
        for name, (key, levels, w) in items.items():
            pairs = [(val(ca, key), val(cb, key)) for _, _, ca, cb in rows]
            reliability["by_area"][SHORT[a]][name] = agreement_block(
                pairs, levels, w, rng, [p for _, p, _, _ in rows])

    # disagreement inventory (raw, before adjudication)
    disagreements = []
    for a, p, ca, cb in overlap_rows:
        for k in ("U", "E", "EP", "S"):
            if ca[k] != cb[k]:
                disagreements.append({"area": SHORT[a], "pmcid": p, "item": k,
                                      "coder_A": ca[k], "coder_B": cb[k]})

    # coder-A cross-area inconsistencies (same paper, two area records)
    cross_area = []
    by_paper = defaultdict(dict)
    for a in AREAS:
        for p, r in A[a].items():
            by_paper[p][a] = codes_of(r)
    for p, d in by_paper.items():
        if len(d) > 1:
            for k in ("U", "E", "EP", "S"):
                vals = {a: c[k] for a, c in d.items()}
                if len(set(vals.values())) > 1:
                    cross_area.append({"pmcid": p, "item": k,
                                       "coder_A_by_area": {SHORT[a]: v for a, v in vals.items()}})

    # ----------------------------------------------------------- 2/3. final set
    adj_by_paper = defaultdict(dict)
    for d in ADJUDICATION:
        adj_by_paper[d["pmcid"]][d["item"]] = d["final"]

    final = {}       # pmcid -> codes + provenance
    meta = {}        # pmcid -> bibliographic fields
    areas_of = defaultdict(list)
    for a in AREAS:
        for p, r in A[a].items():
            areas_of[p].append(a)
            if p not in meta or not meta[p].get("title"):
                meta[p] = {"doi": r.get("doi"), "year": r.get("year"),
                           "journal": r.get("journal"), "title": r.get("title")}
    # fill bibliography gaps from the sample files
    for a in AREAS:
        for r in samples[a]:
            p = r.get("pmcid")
            if p in meta:
                for k in ("doi", "year", "journal", "title"):
                    if not meta[p].get(k):
                        meta[p][k] = r.get(k)

    for p, areas in areas_of.items():
        base = codes_of(A[areas[0]][p])
        # a cross-area inconsistency that is not adjudicated should not exist;
        # assert it, so a silent mismatch cannot slip through
        for a in areas[1:]:
            other = codes_of(A[a][p])
            for k in ("U", "E", "EP", "S"):
                if other[k] != base[k] and k not in adj_by_paper.get(p, {}):
                    raise SystemExit(f"unadjudicated cross-area mismatch {p} {k}")
        codes = dict(base)
        adjudicated = []
        for k, v in adj_by_paper.get(p, {}).items():
            if codes[k] != v or True:
                codes[k] = v
            adjudicated.append(k)
        final[p] = {"codes": codes, "areas": [SHORT[x] for x in areas],
                    "adjudicated": sorted(adjudicated),
                    "dual_coded_in": sorted(SHORT[x] for x in areas if p in B[x])}

    # ------------------------------------------------------------ 4. estimates
    def props(pmcids):
        n = len(pmcids)
        out = {"n": n}
        defs = OrderedDict([
            ("U_ge1", lambda c: c["U"] >= 1),
            ("U_ge2", lambda c: c["U"] >= 2),
            ("U3", lambda c: c["U"] >= 3),
            ("E1", lambda c: c["E"] == 1),
            ("EP", lambda c: c["EP"] == 1),
            ("S1", lambda c: c["S"] == 1),
            ("U_ge2_and_E1_and_S1", lambda c: c["U"] >= 2 and c["E"] == 1 and c["S"] == 1),
        ])
        for name, f in defs.items():
            k = sum(1 for p in pmcids if f(final[p]["codes"]))
            lo, hi = wilson(k, n)
            out[name] = {"k": k, "n": n, "proportion": (k / n) if n else None,
                         "wilson95": [lo, hi]}
        return out

    estimates = {"by_area": {}, "overall_unique_papers": None,
                 "overall_area_records": None}
    for a in AREAS:
        estimates["by_area"][SHORT[a]] = props(sorted(A[a].keys()))
    estimates["overall_unique_papers"] = props(sorted(final.keys()))
    rec_ids = [p for a in AREAS for p in A[a]]
    n_rec = len(rec_ids)
    est_rec = {"n": n_rec}
    for name, f in [("U_ge1", lambda c: c["U"] >= 1), ("U_ge2", lambda c: c["U"] >= 2),
                    ("U3", lambda c: c["U"] >= 3), ("E1", lambda c: c["E"] == 1),
                    ("EP", lambda c: c["EP"] == 1), ("S1", lambda c: c["S"] == 1),
                    ("U_ge2_and_E1_and_S1",
                     lambda c: c["U"] >= 2 and c["E"] == 1 and c["S"] == 1)]:
        k = sum(1 for p in rec_ids if f(final[p]["codes"]))
        lo, hi = wilson(k, n_rec)
        est_rec[name] = {"k": k, "n": n_rec, "proportion": k / n_rec, "wilson95": [lo, hi]}
    estimates["overall_area_records"] = est_rec

    # U3 sensitivity: drop the single low-confidence U3
    sens_ids = sorted(final.keys())
    k_u3_strict = sum(1 for p in sens_ids
                      if final[p]["codes"]["U"] >= 3 and p != "PMC10265106")
    lo, hi = wilson(k_u3_strict, len(sens_ids))
    estimates["U3_sensitivity_strict"] = {
        "description": ("U3 count if the single low-confidence U3 (PMC10265106, coverage check "
                        "reported in the main text but plotted in Supporting Fig. S5, held-out status "
                        "not stated) is demoted to U2."),
        "k": k_u3_strict, "n": len(sens_ids),
        "proportion": k_u3_strict / len(sens_ids), "wilson95": [lo, hi]}

    # S1 sub-codes where coder A recorded them (A1, A2 only)
    s1_types = {}
    for a in AREAS:
        c = Counter()
        n_s1 = 0
        for p, r in A[a].items():
            if codes_of(r)["S"] == 1 or (p in adj_by_paper and adj_by_paper[p].get("S") == 1):
                pass
        for p, r in A[a].items():
            if final[p]["codes"]["S"] == 1:
                n_s1 += 1
                t = r.get("S1_type")
                c[t if t else "not_recorded"] += 1
        s1_types[SHORT[a]] = {"n_S1": n_s1, "sub_codes": dict(c)}
    estimates["S1_sub_codes"] = {
        "note": ("Coder A recorded the S1 sub-code for A1 and A2 only; for A3-A5 the field is absent, "
                 "so the synthetic share cannot be reported for those areas."),
        "by_area": s1_types}

    # year band (secondary, protocol section 4)
    bands = {}
    for a in AREAS:
        b = {"2020_2022": [], "2023_2025": []}
        for p, r in A[a].items():
            y = meta[p].get("year")
            if y is None:
                continue
            b["2020_2022" if int(y) <= 2022 else "2023_2025"].append(p)
        bands[SHORT[a]] = {k: props(v) for k, v in b.items()}
    allb = {"2020_2022": [], "2023_2025": []}
    for p in final:
        y = meta[p].get("year")
        if y is None:
            continue
        allb["2020_2022" if int(y) <= 2022 else "2023_2025"].append(p)
    estimates["by_year_band"] = {"overall_unique_papers": {k: props(v) for k, v in allb.items()},
                                 "by_area_records": bands}

    # ---------------------------------------------------------------- 5. PRISMA
    prisma = {}
    for a in AREAS:
        v = FINAL_VERSION[a]
        c = counts[v]["areas"][a]
        s = samples[a]
        excl = Counter()
        excl_stage = defaultdict(Counter)
        n_ft = 0
        for r in s:
            stage = r.get("stage") or r.get("stage_reached") or "unknown"
            elig = r.get("decision") in ("eligible", "include")
            if stage == "full_text":
                n_ft += 1
            if not elig:
                reason = r.get("exclusion_reason") or "unspecified"
                excl[reason] += 1
                excl_stage[stage][reason] += 1
        n_elig = sum(1 for r in s if r.get("decision") in ("eligible", "include"))
        prisma[SHORT[a]] = {
            "area": LABEL[a], "query_version": v,
            "identified_epmc_topic_all_types": c["epmc_topic_2020_2025_all_types"],
            "removed_tagged_reviews": c["epmc_tagged_review"],
            "removed_preprints": c["epmc_preprint"],
            "research_only_records": c["epmc_research_only"],
            "open_access_frame": c["epmc_oa_frame"],
            "frame_by_year": c["epmc_oa_frame_by_year"],
            "screened_in_random_order": len(s),
            "frame_exhausted": len(s) == c["epmc_oa_frame"],
            "full_text_retrieved": n_ft,
            "excluded_total": len(s) - n_elig,
            "excluded_by_reason": dict(excl),
            "excluded_by_stage_and_reason": {k: dict(vv) for k, vv in excl_stage.items()},
            "eligible": n_elig,
            "coded_coder_A": len(A[a]),
            "dual_coded_coder_B": len(B[a]),
        }
    prisma["_totals"] = {
        "research_only_area_records": sum(counts[FINAL_VERSION[a]]["areas"][a]["epmc_research_only"]
                                          for a in AREAS),
        "oa_frame_area_records": sum(counts[FINAL_VERSION[a]]["areas"][a]["epmc_oa_frame"]
                                     for a in AREAS),
        "screened_area_records": sum(len(samples[a]) for a in AREAS),
        "eligible_area_records": sum(prisma[SHORT[a]]["eligible"] for a in AREAS),
        "coded_area_records": sum(len(A[a]) for a in AREAS),
        "coded_unique_papers": len(final),
        "papers_in_two_areas": sum(1 for p in final if len(final[p]["areas"]) > 1),
        "dual_coded_area_records": sum(len(B[a]) for a in AREAS),
    }

    # ------------------------------------------------------------------ output
    out = OrderedDict()
    out["meta"] = {
        "title": "Systematic survey behind the revised Figure 2 (BMEMat Perspective, R2 comment 13)",
        "protocol": "revision/audit/A6_survey_protocol.md (frozen)",
        "source": "Europe PMC REST, TITLE_ABS search, 2020-2025, English, research articles, open access",
        "frozen_query_version_per_area": FINAL_VERSION,
        "frame_seed": "numpy.random.default_rng(20260919 + area_index)",
        "generated_by": "revision/survey/s07_aggregate.py",
        "coders": {"A": "primary LLM coder, all eligible papers",
                   "B": "second LLM coder, blind 50% subset",
                   "adjudicator": "third reader; re-read the cached JATS full text of every disputed paper"},
    }
    out["prisma"] = prisma
    out["reliability"] = reliability
    out["disagreements_raw"] = disagreements
    out["coder_A_cross_area_inconsistencies"] = cross_area
    out["adjudication"] = ADJUDICATION
    out["flagged_single_coder_borderline"] = FLAGGED_SINGLE_CODER
    out["estimates"] = estimates
    out["limitations"] = LIMITATIONS

    with open(os.path.join(SURVEY, "survey_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # CSV, one row per coded area-record
    csv_path = os.path.join(SURVEY, "survey_coded_papers.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["area", "pmcid", "doi", "year", "journal", "title",
                    "U", "E", "EP", "S", "adjudicated", "adjudicated_items",
                    "dual_coded", "also_coded_in_areas"])
        for a in AREAS:
            for p in sorted(A[a].keys()):
                fc = final[p]
                m = meta[p]
                others = [x for x in fc["areas"] if x != SHORT[a]]
                w.writerow([SHORT[a], p, m.get("doi") or "", m.get("year") or "",
                            m.get("journal") or "", m.get("title") or "",
                            fc["codes"]["U"], fc["codes"]["E"], fc["codes"]["EP"],
                            fc["codes"]["S"],
                            "yes" if fc["adjudicated"] else "no",
                            ";".join(fc["adjudicated"]),
                            "yes" if SHORT[a] in fc["dual_coded_in"] else "no",
                            ";".join(others)])

    write_md(out, samples, A, B, meta, final)
    print(json.dumps({"coded_area_records": out["prisma"]["_totals"]["coded_area_records"],
                      "unique_papers": len(final),
                      "disagreements": len(disagreements),
                      "cross_area_inconsistencies": len(cross_area)}, indent=1))


LIMITATIONS = [
    {"id": "coverage",
     "text": ("Europe PMC coverage. With the same boolean logic, only 23-28% of the article DOIs that "
              "OpenAlex returns are present in the Europe PMC research-only set (A1 26%, 363/1380; A2 26%, "
              "455/1755; A3 23%, 1105/4790; A4 28%, 771/2778); in the other direction 92-96% of Europe PMC "
              "DOIs are in OpenAlex. Materials and engineering journals that Europe PMC does not index "
              "(Chemical Engineering Journal, Nano Energy, Advanced Functional Materials, IEEE Sensors "
              "Journal, Sensors and Actuators A/B, Int. J. Bioprinting and others) are therefore absent, "
              "and part of the gap is OpenAlex noise that the pilot could not separate. The proportions "
              "reported here therefore describe the Europe PMC open-access literature, not the whole "
              "field.")},
    {"id": "oa_only",
     "text": ("Open-access only. The sampling frame is the open-access subset, because full text is needed "
              "for coding. It over-represents open-access biomedical and multidisciplinary venues "
              "(Scientific Reports, Advanced Science, MDPI and Frontiers titles). The non-PMC sensitivity "
              "sample of protocol section 6.4 was not drawn.")},
    {"id": "keyword_queries",
     "text": ("Keyword queries. Identification is a title/abstract boolean search, frozen per area after one "
              "refinement round (A1 v1, A2 v1, A3 v2, A4 v2, A5 v1). Papers that describe their ML or their "
              "material in other words are missed, and title/abstract precision was low in some areas, so "
              "the screen-until-quota sample is a random sample of what the query retrieves, not of the "
              "field. Areas are non-exclusive frames: 16 papers were sampled in two areas.")},
    {"id": "llm_coding",
     "text": ("LLM-assisted coding. Both coders are large language models working from a regex candidate "
              "screen plus an evaluation digest over the parsed JATS full text, with a verbatim quote "
              "required for every positive code. One coder covered all eligible papers, a second, "
              "independent coder a blind 50% subset, and a third reader adjudicated every disagreement "
              "against the full text. The human-versus-LLM blind check of protocol section 9 (20 papers "
              "coded by a domain-expert author) has not been done, so errors shared by both coders would "
              "not be detected here.")},
    {"id": "main_text_only",
     "text": ("Main article only. The construct is 'reported in the main article'; supplementary files were "
              "not screened. Evaluation detail that lives only in the SI is therefore coded absent, which "
              "biases the proportions downwards to an unknown degree.")},
    {"id": "a5_census",
     "text": ("A5 is a census, not a 40-paper sample. The whole A5 open-access frame (145 records) was "
              "screened and yielded only 9 eligible papers, so the A5 confidence intervals are wide and the "
              "pre-specified frame extension of protocol section 6.3 was not run.")},
    {"id": "pooling",
     "text": ("Pooling. Per-area proportions are unweighted. The pooled figure is computed over the 153 "
              "unique papers (each paper once); the area-record version (169) is given for comparison. "
              "Neither is weighted by frame size, so the pooled value is descriptive.")},
    {"id": "power",
     "text": ("Precision. With about 40 papers per area, a 0/40 observation has a 95% Wilson upper bound of "
              "8.8%, which supports or refutes 'rarely reported' (<10%) per area but is not enough for fine "
              "between-area contrasts unless the differences are large.")},
]


def pct(d, nd=1):
    if d is None or d.get("proportion") is None:
        return "-"
    lo, hi = d["wilson95"]
    return f"{d['k']}/{d['n']} = {100*d['proportion']:.{nd}f}% ({100*lo:.{nd}f}-{100*hi:.{nd}f})"


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def ci(x, nd=2):
    return "-" if not x else f"{x[0]:.{nd}f} to {x[1]:.{nd}f}"


def write_md(out, samples, A, B, meta, final):
    P = out["prisma"]
    R = out["reliability"]
    E = out["estimates"]
    L = []
    w = L.append
    w("# Systematic literature survey for Figure 2")
    w("")
    w("Response to Reviewer 2, comment 13 (\"The literature survey needs to be more systematic; "
      "otherwise, Figure 2 may give the impression of 'selective sampling'\").")
    w("")
    w("Protocol: `revision/audit/A6_survey_protocol.md`, frozen before the run. "
      "Source: Europe PMC REST title/abstract search, 2020-2025, English, original research articles, "
      "open access. Frozen query version per area: A1 v1, A2 v1, A3 v2, A4 v2, A5 v1. "
      "Frames were permuted with `numpy.random.default_rng(20260919 + area_index)` and screened in that "
      "order until 40 eligible papers were reached (A5: the whole frame).")
    w("")
    t = P["_totals"]
    w(f"**Sample.** {t['oa_frame_area_records']} open-access records in the five frames; "
      f"{t['screened_area_records']} screened in the frozen random order; "
      f"{t['coded_area_records']} eligible and coded, covering "
      f"{t['coded_unique_papers']} unique papers "
      f"({t['papers_in_two_areas']} papers fell in two areas). Half of the coded records "
      f"({t['dual_coded_area_records']} area-records, "
      f"{R['design']['dual_coded_unique_papers']} unique papers) were coded a second time, "
      "independently and blind, by a different coder, and every disagreement was adjudicated against "
      "the full text by a third reader.")
    w("")

    # headline
    w("## 1. Headline proportions (final, adjudicated codes)")
    w("")
    w("Proportion of coded papers that report each practice, with 95% Wilson confidence intervals.")
    w("")
    w("| Area | n | U>=1 (any UQ) | U>=2 (per-prediction) | U3 (calibrated) | E1 (external) | "
      "EP (prospective) | S1 (shift test) | U>=2 & E1 & S1 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for a in AREAS:
        s = SHORT[a]
        d = E["by_area"][s]
        w(f"| {LABEL[a]} | {d['n']} | {pct(d['U_ge1'])} | {pct(d['U_ge2'])} | {pct(d['U3'])} | "
          f"{pct(d['E1'])} | {pct(d['EP'])} | {pct(d['S1'])} | {pct(d['U_ge2_and_E1_and_S1'])} |")
    d = E["overall_unique_papers"]
    w(f"| **All areas (unique papers)** | **{d['n']}** | **{pct(d['U_ge1'])}** | **{pct(d['U_ge2'])}** | "
      f"**{pct(d['U3'])}** | **{pct(d['E1'])}** | **{pct(d['EP'])}** | **{pct(d['S1'])}** | "
      f"**{pct(d['U_ge2_and_E1_and_S1'])}** |")
    d = E["overall_area_records"]
    w(f"| All areas (area-records) | {d['n']} | {pct(d['U_ge1'])} | {pct(d['U_ge2'])} | {pct(d['U3'])} | "
      f"{pct(d['E1'])} | {pct(d['EP'])} | {pct(d['S1'])} | {pct(d['U_ge2_and_E1_and_S1'])} |")
    w("")
    s = E["U3_sensitivity_strict"]
    w(f"**U3 sensitivity.** The single U3 in the survey (PMC10265106) was coded by one coder with low "
      f"confidence and flagged borderline: the main text reports that all evaluated scores fell inside the "
      f"GPR's 95% interval, but the plot is in Supporting Fig. S5 and the held-out status of those scores "
      f"is not stated. It is kept as U3. If it is demoted to U2, U3 becomes {pct(s)} overall.")
    w("")
    w("**Codes.** U0 none; U1 spread of model *performance* (SD/SE/CI over folds, seeds or repeats); "
      "U2 per-prediction uncertainty reported or used operationally; U3 U2 plus an empirical calibration "
      "or coverage assessment. E1 evaluation on data from a source independent of the training data's "
      "generating process (another lab, instrument, cohort, public dataset, or simulation to experiment). "
      "EP prospective validation: new samples produced or measured after the model was fixed and compared "
      "with its predictions. S1 an explicit distribution-shift evaluation with performance reported.")
    w("")

    # S1 subcodes
    sc = E["S1_sub_codes"]
    w("### 1.1 S1 sub-codes")
    w("")
    w(sc["note"])
    w("")
    w("| Area | S1 papers | sub-codes |")
    w("|---|---|---|")
    for a in AREAS:
        s2 = SHORT[a]
        d2 = sc["by_area"][s2]
        w(f"| {s2} | {d2['n_S1']} | " +
          ", ".join(f"{k} {v}" for k, v in sorted(d2["sub_codes"].items())) + " |")
    w("")

    # year band
    w("### 1.2 Secondary analysis by year band")
    w("")
    yb = E["by_year_band"]["overall_unique_papers"]
    w("| Band | n | U>=1 | U>=2 | U3 | E1 | EP | S1 |")
    w("|---|---|---|---|---|---|---|---|")
    for k, lab in (("2020_2022", "2020-2022"), ("2023_2025", "2023-2025")):
        d2 = yb[k]
        w(f"| {lab} | {d2['n']} | {pct(d2['U_ge1'])} | {pct(d2['U_ge2'])} | {pct(d2['U3'])} | "
          f"{pct(d2['E1'])} | {pct(d2['EP'])} | {pct(d2['S1'])} |")
    w("")

    # Figure 2 mapping
    w("### 1.3 Mapping onto Figure 2")
    w("")
    w("Protocol section 7 fixes the wording rule in advance: \"reported\" if >= 50%, \"limited\" if "
      "10-49%, \"rarely\" if < 10%, with the CI shown. Applying it to the adjudicated codes:")
    w("")
    w("| Area | Uncertainty (U>=2) | External validation (E1) | Shift testing (S1) |")
    w("|---|---|---|---|")

    def band(d6):
        p = d6["proportion"]
        return "reported" if p >= 0.5 else ("limited" if p >= 0.10 else "rarely")
    for a in AREAS:
        s7 = SHORT[a]
        d6 = E["by_area"][s7]
        w(f"| {LABEL[a]} | {band(d6['U_ge2'])} ({100*d6['U_ge2']['proportion']:.0f}%) | "
          f"{band(d6['E1'])} ({100*d6['E1']['proportion']:.0f}%) | "
          f"{band(d6['S1'])} ({100*d6['S1']['proportion']:.0f}%) |")
    w("")
    w("The aggregate claim in the submitted Figure 2 legend - that calibrated uncertainty and "
      "distribution-shift testing are \"reported rarely or only in limited form\" - survives the "
      "systematic sample. Three nuances that the qualitative wording hides should be carried into the "
      "revised figure and text. (i) Per-prediction uncertainty is not uniformly rare: it is the norm in A5 "
      f"({pct(E['by_area']['A5']['U_ge2'])}), because Gaussian-process surrogates are intrinsic to "
      "Bayesian optimisation and active learning, while it is absent in A2 and A3. (ii) Having "
      "per-prediction uncertainty and *assessing* its calibration are different things: U3 is "
      f"{pct(E['overall_unique_papers']['U3'])} across the whole sample, and the one case is borderline. "
      "(iii) Shift testing is not uniformly rare either - it reaches "
      f"{pct(E['by_area']['A1']['S1'])} in imaging, where sim-to-real, acquisition-condition and "
      "grouped-split evaluations are common. External validation is the practice that is genuinely rare "
      f"everywhere ({pct(E['overall_unique_papers']['E1'])}), and the three trust practices almost never "
      f"co-occur: {pct(E['overall_unique_papers']['U_ge2_and_E1_and_S1'])} of papers report U>=2, E1 and "
      "S1 together.")
    w("")

    # PRISMA
    w("## 2. PRISMA-style flow")
    w("")
    w("| Stage | A1 | A2 | A3 | A4 | A5 |")
    w("|---|---|---|---|---|---|")
    rows = [("Identified, Europe PMC topic query, all types", "identified_epmc_topic_all_types"),
            ("Removed: tagged reviews", "removed_tagged_reviews"),
            ("Removed: preprints (SRC:PPR)", "removed_preprints"),
            ("Research-only records", "research_only_records"),
            ("Open-access sampling frame", "open_access_frame"),
            ("Screened in random order", "screened_in_random_order"),
            ("Full texts retrieved", "full_text_retrieved"),
            ("Excluded", "excluded_total"),
            ("Eligible", "eligible"),
            ("Coded (coder A)", "coded_coder_A"),
            ("Dual coded (coder B)", "dual_coded_coder_B")]
    for lab, key in rows:
        w(f"| {lab} | " + " | ".join(str(P[SHORT[a]][key]) for a in AREAS) + " |")
    w("")
    w("Exclusion reasons (`NOT_RESEARCH` review, perspective, resource without model evaluation; "
      "`NO_ML` no machine-learning model; `NOT_BIOMED` no biomedical material or device; "
      "`WRONG_AREA` eligible but not this area):")
    w("")
    reasons = ["NOT_BIOMED", "NO_ML", "WRONG_AREA", "NOT_RESEARCH", "unspecified"]
    present = [r for r in reasons
               if any(P[SHORT[a]]["excluded_by_reason"].get(r) for a in AREAS)]
    w("| Exclusion reason | " + " | ".join(SHORT[a] for a in AREAS) + " |")
    w("|---|" + "---|" * len(AREAS))
    for r in present:
        w(f"| {r} | " + " | ".join(str(P[SHORT[a]]["excluded_by_reason"].get(r, 0))
                                   for a in AREAS) + " |")
    w("")
    w(f"A5's frame was exhausted: all {P['A5']['screened_in_random_order']} open-access records were "
      f"screened and only {P['A5']['eligible']} were eligible, so A5 is a census of its frame rather than "
      "a 40-paper sample. A4 reached the quota after "
      f"{P['A4']['screened_in_random_order']} of {P['A4']['open_access_frame']} records.")
    w("")

    # reliability
    w("## 3. Inter-coder reliability (before adjudication)")
    w("")
    d3 = R["design"]
    w(f"Coder A coded all {d3['coded_area_records_total']} eligible area-records. Coder B independently "
      f"coded a blind 50% subset: {d3['dual_coded_area_records']} area-records covering "
      f"{d3['dual_coded_unique_papers']} unique papers. Cohen's kappa; 95% CIs from "
      f"{d3['bootstrap_draws']} bootstrap draws resampling papers (seed {d3['bootstrap_seed']}). "
      "Because several items are rare, raw agreement, PABAK and positive/negative specific agreement are "
      "reported as well (the kappa paradox).")
    w("")
    w("| Item | n | % agreement (95% CI) | Cohen's kappa (95% CI) | PABAK | pos. spec. agr. | "
      "neg. spec. agr. | A+ / B+ |")
    w("|---|---|---|---|---|---|---|---|")
    names = [("U_4level", "U, 4 levels (unweighted)"),
             ("U_4level_linear_weighted", "U, 4 levels (linearly weighted)"),
             ("U_ge1", "U >= 1"), ("U_ge2", "U >= 2"), ("U_ge3", "U >= 3 (U3)"),
             ("E1", "E1"), ("EP", "EP"), ("S1", "S1")]
    for key, lab in names:
        d4 = R["overall"][key]
        w(f"| {lab} | {d4['n']} | {100*d4['percent_agreement']:.1f}% "
          f"({100*d4['percent_agreement_ci'][0]:.1f}-{100*d4['percent_agreement_ci'][1]:.1f}) | "
          f"{fmt(d4['kappa'])} ({ci(d4.get('kappa_ci'))}) | "
          f"{fmt(d4.get('pabak'))} | {fmt(d4.get('ppos'))} | {fmt(d4.get('pneg'))} | "
          f"{d4.get('n_positive_A', '-')} / {d4.get('n_positive_B', '-')} |")
    w("")
    w("Per area (percent agreement, Cohen's kappa):")
    w("")
    w("| Item | " + " | ".join(SHORT[a] for a in AREAS) + " |")
    w("|---|" + "---|" * len(AREAS))
    for key, lab in names:
        cells = []
        for a in AREAS:
            d4 = R["by_area"][SHORT[a]][key]
            cells.append(f"{100*d4['percent_agreement']:.0f}% / {fmt(d4['kappa'], 2)}")
        w(f"| {lab} | " + " | ".join(cells) + " |")
    w("")
    w("`-` for kappa means chance agreement is 1 and kappa is undefined: both coders used a single "
      "category (this is the case for U3 everywhere, and for several items within single areas). "
      "Percent agreement is the informative statistic there.")
    w("")
    w("Per-area bootstrap confidence intervals for both statistics are in `survey_results.json` "
      "under `reliability.by_area`; they are omitted from the table above only for readability.")
    w("")
    n_dis = len(out["disagreements_raw"])
    ep_k = R["overall"]["EP"]["kappa"]
    w(f"**Disagreements.** {n_dis} item-level disagreements over "
      f"{d3['dual_coded_area_records']} dual-coded area-records x 4 items = "
      f"{4*d3['dual_coded_area_records']} decisions, i.e. "
      f"{100*n_dis/(4*d3['dual_coded_area_records']):.1f}% of decisions. "
      "Every one is adjudicated in section 4. The protocol's threshold (section 9: revise the item "
      f"definition and recode if kappa < 0.6) is met by every item for which kappa is defined; EP is the "
      f"weakest at kappa = {ep_k:.2f} (95% CI "
      f"{R['overall']['EP']['kappa_ci'][0]:.2f} to {R['overall']['EP']['kappa_ci'][1]:.2f}), and four of "
      "the six disagreements, plus all three cross-area inconsistencies, are EP or E calls. EP is the "
      "hardest item in this codebook: it turns on whether the samples post-date the fixing of the model, "
      "which papers often do not state. The protocol's human-versus-LLM blind check (section 9) has not "
      "been run, so these figures bound LLM-to-LLM agreement only, not accuracy.")
    w("")

    # adjudication
    w("## 4. Adjudication")
    w("")
    w("The adjudicator re-read the cached Europe PMC JATS full text of every disputed paper. "
      "Final codes = coder A, except where adjudicated below; the adjudicated code applies to the paper "
      "in every area in which it was sampled.")
    w("")
    for d5 in out["adjudication"]:
        m = meta.get(d5["pmcid"], {})
        w(f"**{d5['pmcid']}** ({', '.join(SHORT[a] for a in d5['areas'])}) - item **{d5['item']}**: "
          f"coder A {d5['coder_A']}, coder B {d5['coder_B']} -> **{d5['decision']}**  ")
        w(f"*{m.get('title','')}* ({m.get('journal','')}, {m.get('year','')})  ")
        w(d5["reason"])
        w("")
    w("Eight items over seven papers were adjudicated: six coder-A-versus-coder-B disagreements, plus "
      "two items (PMC11921027 E, PMC8575943 EP) where the coders agreed within each area but the same "
      "paper had been given different codes in its two area records. PMC7076403 was both. Sixteen papers "
      "were sampled in two areas, and three of them carried such a cross-area inconsistency "
      "(for PMC8575943 both coders made the same one, EP=1 in its A1 record and EP=0 in its A4 record). "
      "The protocol requires a paper to be coded once, so these were adjudicated too. Their existence is "
      "itself a finding: the area context in which a paper is read can move a borderline call, and all "
      "three sit on the EP/E boundary.")
    w("")
    for d5 in out["flagged_single_coder_borderline"]:
        w(f"**{d5['pmcid']}** ({', '.join(SHORT[a] for a in d5['areas'])}) - item **{d5['item']}**, "
          f"single-coder borderline, code upheld at U{d5['final']}.  ")
        w(d5["reason"])
        w("")

    # limitations
    w("## 5. Limitations")
    w("")
    for l in out["limitations"]:
        w(f"- **{l['id']}.** {l['text']}")
    w("")
    w("## 6. Files")
    w("")
    w("- `survey_results.json` - every number in this note, machine readable.")
    w("- `survey_coded_papers.csv` - one row per coded area-record (169 rows, 153 unique papers) with "
      "area, PMCID, DOI, year, journal, title, final U/E/EP/S codes and the adjudication flag. "
      "This is the Supporting Information table that replaces the hand-picked Table 6.")
    w("- `sample_A*.json` - the screening decision and reason for every record examined.")
    w("- `codes_A_A*.json`, `codes_B_A*.json` - both coders' codes with verbatim evidence quotes.")
    w("")

    with open(os.path.join(SURVEY, "survey_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
