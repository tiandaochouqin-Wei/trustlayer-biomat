"""Cached OpenAlex client (secondary source: coverage cross-check only).

The Europe PMC title/abstract queries in config.py are translated to OpenAlex
`title_and_abstract.search` syntax: OpenAlex stems words and does not support
`*` truncation, so each truncated stem is replaced by one full word whose stem
covers the same family (e.g. regenerat* -> regeneration).  No e-mail address is
sent (no `mailto` parameter).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "openalex"
CACHE.mkdir(parents=True, exist_ok=True)
BASE = "https://api.openalex.org/works"

# truncated stem -> full word whose Porter stem matches the family
EXPAND = {"regenerat": "regeneration", "biocompatib": "biocompatibility",
          "bioprint": "bioprinting", "bioelectronic": "bioelectronics"}


def translate(epmc_topic: str) -> str:
    """Europe PMC TITLE_ABS topic query -> OpenAlex boolean search string."""
    q = epmc_topic.replace("TITLE_ABS:", "")
    q = re.sub(r"\b([A-Za-z\-]+)\*", lambda m: EXPAND.get(m.group(1), m.group(1)), q)
    return q


def _key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:20]


def _get(params: dict, tries: int = 4):
    for i in range(tries):
        try:
            r = requests.get(BASE, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and "budget" in r.text.lower():
                break  # daily free budget exhausted; resets at midnight UTC
            if r.status_code == 429:
                time.sleep(1.5)  # >5 boolean operators: limited to 1 request/s
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"OpenAlex GET failed: {params}")


def filter_string(search: str, years=(2020, 2025), article_only=True, oa=None) -> str:
    f = f"title_and_abstract.search:{search},publication_year:{years[0]}-{years[1]}"
    if article_only:
        f += ",type:article"
    if oa is not None:
        f += f",is_oa:{'true' if oa else 'false'}"
    return f


def count(search: str, **kw) -> int:
    flt = filter_string(search, **kw)
    f = CACHE / f"count_{_key(flt)}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["count"]
    d = _get({"filter": flt, "per_page": 1, "select": "id"})
    n = int(d["meta"]["count"])
    f.write_text(json.dumps({"filter": flt, "count": n,
                             "retrieved": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
    return n


def all_works(search: str, **kw) -> list[dict]:
    flt = filter_string(search, **kw)
    f = CACHE / f"works_{_key(flt)}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["works"]
    out, cursor = [], "*"
    while cursor:
        d = _get({"filter": flt, "per_page": 200, "cursor": cursor,
                  "select": "id,doi,title,publication_year,type,open_access,primary_location"})
        res = d.get("results", [])
        for w in res:
            src = ((w.get("primary_location") or {}).get("source") or {})
            out.append({"id": w["id"], "doi": (w.get("doi") or "").lower().replace("https://doi.org/", ""),
                        "title": w.get("title"), "year": w.get("publication_year"),
                        "is_oa": (w.get("open_access") or {}).get("is_oa"),
                        "source": src.get("display_name")})
        cursor = d["meta"].get("next_cursor") if res else None
        time.sleep(0.2)
    f.write_text(json.dumps({"filter": flt, "n": len(out),
                             "retrieved": time.strftime("%Y-%m-%d %H:%M:%S"), "works": out},
                            ensure_ascii=False), encoding="utf-8")
    return out
