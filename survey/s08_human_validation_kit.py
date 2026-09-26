"""Human-validation kit for the LLM-coded survey (the check the manuscript says was not done).

    python s08_human_validation_kit.py make      # writes validation/ (rater sheet, key, codebook)
    python s08_human_validation_kit.py score validation/rater_filled.csv

make   draws 30 of the 153 unique papers with a fixed seed: every paper with U >= 2 or E1 (rare
       positives, otherwise kappa is undefined), then S1-positive and EP-positive papers to reach
       18 positives, then 12 random papers coded 0 on those items. Positives are over-sampled on
       purpose, so the resulting agreement describes the coding of rare items, not the survey's
       prevalences; both raw agreement and Cohen's kappa are reported by ``score``.
score  compares a human-filled sheet with the (hidden) LLM key: raw agreement, Cohen's kappa,
       PABAK and positive/negative specific agreement per item.
"""
import csv
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "validation")
ITEMS = ["U", "E", "EP", "S"]
SEED = 20260926


def load():
    rows = list(csv.DictReader(open(os.path.join(HERE, "survey_coded_papers.csv"), encoding="utf-8")))
    seen, uniq = set(), []
    for r in rows:                       # a paper coded in two areas keeps its first row
        if r["pmcid"] not in seen:
            seen.add(r["pmcid"]); uniq.append(r)
    return uniq


def codebook():
    """The frozen protocol's own definitions (audit/A6_survey_protocol.md), verbatim."""
    txt = open(os.path.join(HERE, "..", "audit", "A6_survey_protocol.md"), encoding="utf-8").read().splitlines()
    tab = [ln for ln in txt if re.match(r"\| \*\*(U0|U1|U2|U3|E0|E1|EP|S0|S1)\*\*", ln)]
    head = ("# Coding sheet instructions\n\nFor each paper, read the full text (open access, link in the sheet) and "
            "fill the six code columns (U_ge1, U_ge2, U3, E1, EP, S1) with 0 or 1. Do not look at the key file.\n\n"
            "* `U_ge1` = 1 if the paper reports any uncertainty (code U1, U2 or U3 below)\n"
            "* `U_ge2` = 1 if U2 or U3; `U3` = 1 if U3\n* `E1`, `EP`, `S1` as defined below\n\n"
            "| Code | Definition | Counts | Does not count |\n|---|---|---|---|\n")
    return head + "\n".join(tab) + "\n"


def make():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    P = load()
    take = []
    def add(pool, k):
        pool = [p for p in pool if p not in take]
        rng.shuffle(pool)
        take.extend(pool[:k])
    add([p for p in P if int(p["U"]) >= 2 or p["E"] == "1"], 18)
    add([p for p in P if p["S"] == "1"], 18 - len(take))
    add([p for p in P if p["EP"] == "1"], 18 - len(take))
    add([p for p in P if int(p["U"]) == 0 and p["E"] == "0" and p["S"] == "0" and p["EP"] == "0"], 30 - len(take))
    rng.shuffle(take)
    with open(os.path.join(OUT, "rater_sheet.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "area", "pmcid", "link", "title", "U_ge1", "U_ge2", "U3", "E1", "EP", "S1", "note"])
        for i, p in enumerate(take, 1):
            w.writerow([i, p["area"], p["pmcid"], f"https://pmc.ncbi.nlm.nih.gov/articles/{p['pmcid']}/",
                        p["title"], "", "", "", "", "", "", ""])
    with open(os.path.join(OUT, "llm_key_DO_NOT_SHARE_WITH_RATER.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pmcid", "U_ge1", "U_ge2", "U3", "E1", "EP", "S1"])
        for i, p in enumerate(take, 1):
            u = int(p["U"])
            w.writerow([i, p["pmcid"], int(u >= 1), int(u >= 2), int(u >= 3), p["E"], p["EP"], p["S"]])
    open(os.path.join(OUT, "CODEBOOK.md"), "w", encoding="utf-8").write(codebook())
    print(f"{len(take)} papers -> {OUT}")


def kappa_stats(a, b):
    n = len(a)
    tp = sum(x == y == 1 for x, y in zip(a, b)); tn = sum(x == y == 0 for x, y in zip(a, b))
    fp = sum(x == 0 and y == 1 for x, y in zip(a, b)); fn = sum(x == 1 and y == 0 for x, y in zip(a, b))
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n ** 2
    kap = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return dict(n=n, agree=po, kappa=kap, pabak=2 * po - 1,
                pos_agree=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
                neg_agree=2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else float("nan"),
                llm_pos=tp + fp, human_pos=tp + fn)


def score(path):
    key = {r["id"]: r for r in csv.DictReader(open(os.path.join(OUT, "llm_key_DO_NOT_SHARE_WITH_RATER.csv"), encoding="utf-8"))}
    hum = {r["id"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}
    print(f"{'item':6} {'n':>3} {'agree':>6} {'kappa':>6} {'PABAK':>6} {'pos.ag':>6} {'neg.ag':>6}  llm+/human+")
    for it in ["U_ge1", "U_ge2", "U3", "E1", "EP", "S1"]:
        ids = [i for i in key if hum.get(i, {}).get(it, "").strip() in ("0", "1")]
        s = kappa_stats([int(key[i][it]) for i in ids], [int(hum[i][it]) for i in ids])
        print(f"{it:6} {s['n']:>3} {s['agree']:6.2f} {s['kappa']:6.2f} {s['pabak']:6.2f} "
              f"{s['pos_agree']:6.2f} {s['neg_agree']:6.2f}  {s['llm_pos']}/{s['human_pos']}")


if __name__ == "__main__":
    make() if sys.argv[1:2] == ["make"] else score(sys.argv[2])
