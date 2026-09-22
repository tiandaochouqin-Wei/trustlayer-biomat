"""Step 6 - write the frozen per-area query strings and their counts.

Output: queries_final.txt (exact Europe PMC query strings, OpenAlex
translations) and prints a count table.  Counts come from the cache written by
s01_counts.py (run `python s01_counts.py v1` and `python s01_counts.py v2`).
"""
import json
from pathlib import Path

import config as C
import epmc
import openalex as O

HERE = Path(__file__).resolve().parent
cv = {v: json.loads((epmc.CACHE / f"counts_{v}.json").read_text(encoding="utf-8")) for v in ("v1", "v2")}
lines = [f"# Frozen survey queries (Europe PMC REST 'query' parameter), seed={C.SEED}, years={C.YEARS}", ""]
print("| Area | version | topic 2020-25 (all types) | tagged reviews | preprints | research-only | OA research-only frame | OpenAlex articles (OA) |")
print("|---|---|---|---|---|---|---|---|")
for a in C.AREAS:
    v = C.FINAL_VERSION[a]
    d = cv[v]["areas"][a]
    lines += [f"## {a}  (query version {v})", "",
              "Research-only query:", C.area_query(a, version=v), "",
              "OA sampling frame = research-only query AND " + C.OA_F, "",
              "OpenAlex title_and_abstract.search translation (coverage check; plus publication_year:2020-2025,type:article):",
              O.translate(C.area_core(a, v)), ""]
    ox = f"{d['openalex_articles']} ({d['openalex_articles_oa']})" if d.get("openalex_articles") is not None else "not retrieved"
    print(f"| {a} | {v} | {d['epmc_topic_2020_2025_all_types']} | {d['epmc_tagged_review']} | {d['epmc_preprint']} | "
          f"{d['epmc_research_only']} | {d['epmc_oa_frame']} | {ox} |")
(HERE / "queries_final.txt").write_text("\n".join(lines), encoding="utf-8")
print("written", HERE / "queries_final.txt")
