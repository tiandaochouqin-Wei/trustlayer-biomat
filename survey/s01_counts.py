"""Step 1 - identification counts (PRISMA 'records identified').

Outputs cache/counts.json and prints a markdown table.  Re-running uses the
cache; delete cache/search and cache/openalex to refresh.
"""
import collections
import itertools
import json

import config as C
import epmc
import openalex as O
import sys

V = sys.argv[1] if len(sys.argv) > 1 else C.QUERY_VERSION

out = {"version": None, "areas": {}, "overlap_research": {}, "overlap_oa": {}}
ids_research, ids_oa = {}, {}
out["version"] = V

for a in C.AREAS:
    core_yr = f"({C.area_core(a, V)}) AND {C.YEAR_F}"
    rec_res = epmc.all_records(C.area_query(a, version=V), result_type="lite")
    rec_oa = epmc.all_records(C.area_query(a, oa=True, version=V), result_type="core")
    ids_research[a] = {r["id"] for r in rec_res}
    ids_oa[a] = {r["id"] for r in rec_oa}
    by_year = collections.Counter(int(r["pubYear"]) for r in rec_oa if r.get("pubYear"))
    # OpenAlex cross-check
    oa_search = O.translate(C.area_core(a, V))
    try:
        ow = O.all_works(oa_search)
    except RuntimeError as e:  # OpenAlex daily budget / rate limit: record and continue
        print(f"[{a}] OpenAlex download unavailable: {e.__class__.__name__}")
        ow = None
    try:
        ox_n, ox_n_oa = O.count(oa_search), O.count(oa_search, oa=True)
    except RuntimeError:
        ox_n = ox_n_oa = None
    ox_dois = {w["doi"] for w in ow if w["doi"]} if ow is not None else set()
    ep_dois = {(r.get("doi") or "").lower() for r in rec_res if r.get("doi")}
    out["areas"][a] = {
        "epmc_query_research_only": C.area_query(a, version=V),
        "epmc_query_oa_frame": C.area_query(a, oa=True, version=V),
        "openalex_search": oa_search,
        "epmc_topic_2020_2025_all_types": epmc.hit_count(core_yr),
        "epmc_tagged_review": epmc.hit_count(core_yr + ' AND (PUB_TYPE:review OR PUB_TYPE:"review-article" OR PUB_TYPE:"systematic review" OR PUB_TYPE:"systematic-review")'),
        "epmc_preprint": epmc.hit_count(core_yr + " AND SRC:PPR"),
        "epmc_research_only": len(rec_res),
        "epmc_research_only_hitcount": epmc.hit_count(C.area_query(a, version=V)),
        "epmc_oa_frame": len(rec_oa),
        "epmc_oa_frame_by_year": dict(sorted(by_year.items())),
        "openalex_articles": ox_n,
        "openalex_articles_oa": ox_n_oa,
        "openalex_downloaded": ow is not None,
        "openalex_dois": len(ox_dois) if ow is not None else None,
        "epmc_research_dois": len(ep_dois),
        "doi_overlap": len(ox_dois & ep_dois) if ow is not None else None,
        "frac_openalex_dois_in_epmc": round(len(ox_dois & ep_dois) / max(1, len(ox_dois)), 3) if ow is not None else None,
        "frac_epmc_dois_in_openalex": round(len(ox_dois & ep_dois) / max(1, len(ep_dois)), 3) if ow is not None else None,
    }

for (a, b) in itertools.combinations(C.AREAS, 2):
    out["overlap_research"][f"{a}|{b}"] = len(ids_research[a] & ids_research[b])
    out["overlap_oa"][f"{a}|{b}"] = len(ids_oa[a] & ids_oa[b])
out["union_research"] = len(set().union(*ids_research.values()))
out["union_oa"] = len(set().union(*ids_oa.values()))
out["sum_research"] = sum(len(v) for v in ids_research.values())
out["sum_oa"] = sum(len(v) for v in ids_oa.values())

(C_PATH := epmc.CACHE / f"counts_{V}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

print("| Area | EPMC topic 2020-25 (all types) | tagged reviews | preprints | research-only | OA research-only (frame) | OpenAlex articles (OA) | OpenAlex DOIs found in EPMC |")
print("|---|---|---|---|---|---|---|---|")
for a, d in out["areas"].items():
    cov = (f"{d['doi_overlap']}/{d['openalex_dois']} = {d['frac_openalex_dois_in_epmc']:.0%}"
           if d["openalex_downloaded"] else "not retrieved (OpenAlex quota)")
    print(f"| {a} | {d['epmc_topic_2020_2025_all_types']} | {d['epmc_tagged_review']} | {d['epmc_preprint']} | "
          f"{d['epmc_research_only']} | {d['epmc_oa_frame']} | {d['openalex_articles']} ({d['openalex_articles_oa']}) | {cov} |")
print("union research-only:", out["union_research"], "of", out["sum_research"], "area-records;",
      "union OA frame:", out["union_oa"], "of", out["sum_oa"])
print("pairwise overlap (research-only):", out["overlap_research"])
print("OA frame by year:", {a: d["epmc_oa_frame_by_year"] for a, d in out["areas"].items()})
