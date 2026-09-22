"""Step 5 - pilot summary: T/A precision, hit rates, pattern precision,
regex-vs-verified agreement at paper level.  Prints markdown tables used in
A6_survey_protocol.md and writes pilot/summary.json.
"""
import collections
import csv
import json
import math
import sys
from pathlib import Path

V = sys.argv[1] if len(sys.argv) > 1 else "v10"
SUF = "" if V == "v10" else "_" + V

HERE = Path(__file__).resolve().parent
P = HERE / "pilot"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


out = {}
# ---------------------------------------------------------------- T/A screen
ta = list(csv.DictReader(open(P / "ta_labels.csv", encoding="utf-8")))
print("## Title/abstract precision (first 12 records of each random frame)")
print("| Area | v1 incl./12 | v2 incl./12 | reasons for exclusion (v1+v2, first 12) |")
print("|---|---|---|---|")
out["ta"] = {}
for a in ["A1_imaging", "A2_property", "A3_sensors", "A4_tissue", "A5_autonomous"]:
    row = {}
    reasons = collections.Counter()
    for v in ("v1", "v2"):
        rs = [r for r in ta if r["area"] == a and r["version"] == v and int(r["rank"]) <= 12]
        k = sum(r["decision"].startswith("include") for r in rs)
        row[v] = (k, len(rs))
        reasons.update(r["reason"] for r in rs if r["decision"] == "exclude")
    out["ta"][a] = {"v1": row["v1"], "v2": row["v2"], "reasons": dict(reasons)}
    f = lambda t: f"{t[0]}/{t[1]}" if t[1] else "n/a"
    print(f"| {a} | {f(row['v1'])} | {f(row['v2'])} | {dict(reasons)} |")
allr = collections.Counter(r["reason"] for r in ta if r["decision"] == "exclude")
print("all screened records:", len(ta), "| exclusion reasons:", dict(allr))

# ---------------------------------------------------------------- hits
papers = list(csv.DictReader(open(P / f"screen_papers{SUF}.csv", encoding="utf-8")))
ver = list(csv.DictReader(open(P / f"hit_verification{SUF}.csv", encoding="utf-8")))
codes = {r["pmcid"]: r for r in csv.DictReader(open(P / "paper_codes_verified.csv", encoding="utf-8"))}
n = len(papers)
print(f"\n## Regex screen: {n} papers, {sum(int(p['n_sentences']) for p in papers)} sentences, {len(ver)} candidate hits")
print("| Pattern | hits | papers hit | TP hits | hit precision | papers with >=1 TP |")
print("|---|---|---|---|---|---|")
out["patterns"] = {}
for pat in sorted({v["pattern"] for v in ver}):
    hs = [v for v in ver if v["pattern"] == pat]
    tp = [v for v in hs if v["verdict"] == "TP"]
    pp = {v["pmcid"] for v in hs}
    ptp = {v["pmcid"] for v in tp}
    out["patterns"][pat] = {"hits": len(hs), "papers": len(pp), "tp": len(tp), "papers_tp": len(ptp)}
    print(f"| {pat} | {len(hs)} | {len(pp)} | {len(tp)} | {len(tp)/len(hs):.0%} | {len(ptp)}/{len(pp)} |")
fp = collections.Counter(v["fp_category"] for v in ver if v["verdict"] == "FP")
print("FP categories:", fp.most_common())
out["fp_categories"] = dict(fp)
sec = collections.Counter((v["section"], v["verdict"]) for v in ver)
print("hits by section x verdict:", dict(sec))
out["section_verdict"] = {f"{a}|{b}": c for (a, b), c in sec.items()}

# ---------------------------------------------------------------- paper level
print("\n## Paper-level: regex-only code vs verified code")
agree = collections.Counter()
rows = []
for p in papers:
    c = codes[p["pmcid"]]
    rows.append((p["area"], p["pmcid"], p["regex_U"], c["U"], p["regex_E1"], c["E1"], p["regex_EP"], c["EP"],
                 p["regex_S1"], c["S1"]))
    agree["U"] += p["regex_U"] == c["U"]
    agree["E1"] += int(p["regex_E1"]) == int(c["E1"])
    agree["EP"] += int(p["regex_EP"]) == int(c["EP"])
    agree["S1"] += int(p["regex_S1"]) == int(c["S1"])
print("| Area | PMCID | U regex→verified | E1 | EP | S1 |")
print("|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]}→{r[3]} | {r[4]}→{r[5]} | {r[6]}→{r[7]} | {r[8]}→{r[9]} |")
print("regex-only agreement with verified code:", {k: f"{v}/{n}" for k, v in agree.items()})
out["paper_agreement"] = {k: [v, n] for k, v in agree.items()}


def conf(rx, ver_):
    tp = sum(1 for a, b in zip(rx, ver_) if a and b)
    fp_ = sum(1 for a, b in zip(rx, ver_) if a and not b)
    fn = sum(1 for a, b in zip(rx, ver_) if (not a) and b)
    tn = sum(1 for a, b in zip(rx, ver_) if (not a) and (not b))
    return tp, fp_, fn, tn


print("\n## Paper-level confusion (regex flag vs verified), TP/FP/FN/TN")
out["paper_confusion"] = {}
for name, rxf, vf in [
    ("U>=1", lambda p: p["regex_U"] != "U0", lambda c: c["U"] != "U0"),
    ("U>=2", lambda p: p["regex_U"] in ("U2", "U3"), lambda c: c["U"] in ("U2", "U3")),
    ("U3", lambda p: p["regex_U"] == "U3", lambda c: c["U"] == "U3"),
    ("E1", lambda p: p["regex_E1"] == "1", lambda c: c["E1"] == "1"),
    ("EP", lambda p: p["regex_EP"] == "1", lambda c: c["EP"] == "1"),
    ("S1", lambda p: p["regex_S1"] == "1", lambda c: c["S1"] == "1"),
]:
    t = conf([rxf(p) for p in papers], [vf(codes[p["pmcid"]]) for p in papers])
    out["paper_confusion"][name] = t
    print(f"{name}: TP={t[0]} FP={t[1]} FN={t[2]} TN={t[3]}")

print("\n## Verified prevalence in pilot (n per area = 5; descriptive only)")
out["prevalence"] = {}
for a in ["A1_imaging", "A2_property", "A3_sensors", "A4_tissue", "A5_autonomous", "ALL"]:
    cs = [c for c in codes.values() if a == "ALL" or c["area"] == a]
    k = {"U>=1": sum(c["U"] != "U0" for c in cs), "U>=2": sum(c["U"] in ("U2", "U3") for c in cs),
         "U3": sum(c["U"] == "U3" for c in cs), "E1": sum(c["E1"] == "1" for c in cs),
         "EP": sum(c["EP"] == "1" for c in cs), "S1": sum(c["S1"] == "1" for c in cs)}
    out["prevalence"][a] = {kk: [vv, len(cs)] for kk, vv in k.items()}
    ci = {kk: wilson(vv, len(cs)) for kk, vv in k.items()}
    print(a, {kk: f"{vv}/{len(cs)} (95% CI {ci[kk][0]:.2f}-{ci[kk][1]:.2f})" for kk, vv in k.items()})

print("\n## Text available to the screen")
for p in papers:
    pass
print("median sentences/paper:", sorted(int(p["n_sentences"]) for p in papers)[n // 2],
      "| papers with >=1 supplementary file:", sum(int(p["n_supplementary_files"]) > 0 for p in papers), "/", n)
(P / f"summary{SUF}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
