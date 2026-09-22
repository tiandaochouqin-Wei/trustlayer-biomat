"""Minimal, cached Europe PMC REST client used by the survey scripts.

All responses are cached under revision/survey/cache/ so that every count and
sample reported in A6_survey_protocol.md can be reproduced offline.  No e-mail
address or other personal identifier is sent with any request.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
(CACHE / "search").mkdir(exist_ok=True)
(CACHE / "fulltext").mkdir(exist_ok=True)

_S = requests.Session()
_S.headers["User-Agent"] = "BMEMat-revision-survey/0.1 (research script; python-requests)"


def _key(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:20]


def _get(url: str, params: dict | None = None, tries: int = 4, timeout: int = 90):
    for i in range(tries):
        try:
            r = _S.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (404, 400):
                return r
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET failed after {tries} tries: {url} {params}")


def hit_count(query: str, use_cache: bool = True) -> int:
    """Number of Europe PMC records matching `query` (cached)."""
    f = CACHE / "search" / f"count_{_key(query)}.json"
    if use_cache and f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["hitCount"]
    r = _get(f"{BASE}/search", {"query": query, "format": "json", "pageSize": 1,
                                 "resultType": "idlist"})
    n = int(r.json()["hitCount"])
    f.write_text(json.dumps({"query": query, "hitCount": n,
                             "retrieved": time.strftime("%Y-%m-%d %H:%M:%S")},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    return n


def all_records(query: str, result_type: str = "core", page_size: int = 1000,
                use_cache: bool = True) -> list[dict]:
    """All records for `query` via cursorMark paging (cached as one JSON file)."""
    f = CACHE / "search" / f"records_{result_type}_{_key(query)}.json"
    if use_cache and f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    cursor, out = "*", []
    while True:
        r = _get(f"{BASE}/search", {"query": query, "format": "json",
                                     "pageSize": page_size, "resultType": result_type,
                                     "cursorMark": cursor})
        d = r.json()
        res = d.get("resultList", {}).get("result", [])
        out.extend(res)
        nxt = d.get("nextCursorMark")
        if not res or not nxt or nxt == cursor:
            break
        cursor = nxt
        time.sleep(0.3)
    slim = []
    keep = ("id", "source", "pmid", "pmcid", "doi", "title", "authorString", "pubYear",
            "journalInfo", "abstractText", "pubTypeList", "isOpenAccess", "inEPMC",
            "inPMC", "hasPDF", "journalTitle", "keywordList")
    for x in out:
        slim.append({k: x.get(k) for k in keep if k in x})
    f.write_text(json.dumps({"query": query, "n": len(slim),
                             "retrieved": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "records": slim}, ensure_ascii=False), encoding="utf-8")
    return slim


def fulltext_xml(pmcid: str, use_cache: bool = True) -> str | None:
    """JATS full-text XML for an open-access PMC article (cached)."""
    f = CACHE / "fulltext" / f"{pmcid}.xml"
    if use_cache and f.exists():
        return f.read_text(encoding="utf-8")
    r = _get(f"{BASE}/{pmcid}/fullTextXML")
    if r.status_code != 200 or not r.text.strip().startswith("<"):
        return None
    f.write_text(r.text, encoding="utf-8")
    time.sleep(0.3)
    return r.text
