"""Step 2 - sampling frames: random order of each area's OA research-only set.

Each frame is sorted by Europe PMC id (a stable key) and then permuted with
numpy.random.default_rng(SEED + area_index).  Screening proceeds down this
order until the per-area quota of eligible full texts is met ("screen-until-
quota"), which yields a simple random sample of eligible OA papers.
Outputs frames/<version>/<area>_frame.csv (rank, ids, year, journal, title, abstract).
"""
import csv
import html
import re
from pathlib import Path

import numpy as np

import config as C
import epmc
import sys

V = sys.argv[1] if len(sys.argv) > 1 else C.QUERY_VERSION

OUT = Path(__file__).resolve().parent / "frames" / V
OUT.mkdir(parents=True, exist_ok=True)


def clean(s):
    s = html.unescape(s or "")
    return re.sub(r"<[^>]+>", "", s).strip()


for i, a in enumerate(C.AREAS):
    recs = epmc.all_records(C.area_query(a, oa=True, version=V), result_type="core")
    recs = sorted(recs, key=lambda r: r["id"])
    order = np.random.default_rng(C.SEED + i).permutation(len(recs))
    with open(OUT / f"{a}_frame.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "area", "epmc_id", "pmcid", "doi", "year", "journal", "pub_types", "title", "abstract"])
        for rank, j in enumerate(order, 1):
            r = recs[j]
            ji = r.get("journalInfo") or {}
            jt = (ji.get("journal") or {}).get("title", "")
            pts = "; ".join((r.get("pubTypeList") or {}).get("pubType", []))
            w.writerow([rank, a, r["id"], r.get("pmcid", ""), r.get("doi", ""), r.get("pubYear", ""),
                        jt, pts, clean(r.get("title")), clean(r.get("abstractText"))])
    print(a, len(recs), "records; with PMCID:", sum(1 for r in recs if r.get("pmcid")))
