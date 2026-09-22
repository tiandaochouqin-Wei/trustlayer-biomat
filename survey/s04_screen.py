"""Step 4 - regex screen of JATS full texts.

For every pilot paper: parse the Europe PMC JATS XML into paragraphs tagged
with a section class (ABSTRACT / INTRO / METHODS / RESULTS / RESULTS_DISCUSSION /
DISCUSSION / CONCLUSION / FIG / TABLE / APPENDIX / BODY_OTHER), replace in-text
citations by [CIT], split into sentences and apply regexes.PATTERNS.
Reference lists, acknowledgements, declarations and supplementary-file stubs
are not screened.

Outputs
  pilot/screen_hits.csv    one row per (sentence, pattern) candidate
  pilot/screen_papers.csv  one row per paper: text size, hit counts, regex-only codes
"""
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import regexes as R

HERE = Path(__file__).resolve().parent
FT = HERE / "cache" / "fulltext"
PILOT = HERE / "pilot"
OWN = R.OWN_WORK | {"APPENDIX"}

ABBR = r"(?<!\bFig)(?<!\bFigs)(?<!\bEq)(?<!\bEqs)(?<!\bRef)(?<!\bRefs)(?<!\bet al)(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bNo)(?<!\bapprox)(?<!\bca)(?<!\bSupp)(?<!\bTab)"
SPLIT = re.compile(ABBR + r"(?<=[.!?])\s+(?=[A-Z(\[])")


def text_of(el) -> str:
    parts = [el.text or ""]
    for ch in el:
        tag = ch.tag
        if tag == "xref" and ch.get("ref-type") == "bibr":
            parts.append("[CIT]")
        elif tag in ("fig", "table-wrap", "supplementary-material"):
            pass  # handled separately
        else:
            parts.append(text_of(ch))
        parts.append(ch.tail or "")
    return "".join(parts)


def norm(s: str) -> str:
    s = re.sub(r"(\[CIT\][\s,–-]*)+", "[CIT] ", s)
    return " ".join(s.split())


def walk(el, cls, title_path, out, top=False):
    for ch in el:
        tag = ch.tag
        if tag == "sec":
            title = norm(text_of(ch.find("title"))) if ch.find("title") is not None else ""
            st = ch.get("sec-type") or ""
            if st in R.SKIP_SECTYPE or (title and R.SKIP_SEC.search(title)):
                continue
            if top or cls in ("BODY_OTHER", None):
                c = R.classify_section(title, st)
                if c == "BODY_OTHER" and cls not in (None, "BODY_OTHER"):
                    c = cls
            else:
                c = cls
            walk(ch, c, title_path + [title], out)
        elif tag == "p":
            t = norm(text_of(ch))
            if t:
                out.append((cls or "BODY_OTHER", " > ".join(x for x in title_path if x), t))
            # figures/tables nested inside paragraphs
            walk_floats(ch, title_path, out)
        elif tag in ("fig", "table-wrap", "fig-group", "table-wrap-group"):
            walk_floats_one(ch, title_path, out)
        elif tag in ("list", "list-item", "boxed-text", "disp-quote", "statement"):
            walk(ch, cls, title_path, out)
        elif tag in ("title", "label", "supplementary-material", "ref-list", "ack", "fn-group"):
            continue
        else:
            walk(ch, cls, title_path, out)


def walk_floats(el, title_path, out):
    for ch in el.iter():
        if ch is el:
            continue
        if ch.tag in ("fig", "table-wrap"):
            walk_floats_one(ch, title_path, out)


def walk_floats_one(ch, title_path, out):
    if ch.tag in ("fig-group", "table-wrap-group"):
        for sub in ch:
            if sub.tag in ("fig", "table-wrap"):
                walk_floats_one(sub, title_path, out)
        return
    if ch.tag == "fig":
        cap = ch.find("caption")
        if cap is not None:
            t = norm(" ".join(norm(text_of(x)) for x in cap))
            if t:
                out.append(("FIG", " > ".join(x for x in title_path if x), t))
    elif ch.tag == "table-wrap":
        pieces = []
        cap = ch.find("caption")
        if cap is not None:
            pieces.append(norm(" ".join(norm(text_of(x)) for x in cap)))
        for tr in ch.iter("tr"):
            cells = [norm(text_of(c)) for c in tr if c.tag in ("td", "th")]
            if any(cells):
                pieces.append(" | ".join(cells) + ".")
        foot = ch.find("table-wrap-foot")
        if foot is not None:
            pieces.append(norm(text_of(foot)))
        t = " ".join(p for p in pieces if p)
        if t:
            out.append(("TABLE", " > ".join(x for x in title_path if x), t))


def paragraphs(xml_text: str):
    root = ET.fromstring(xml_text)
    out = []
    for ab in root.findall(".//front/article-meta/abstract"):
        if ab.get("abstract-type") in ("graphical", "teaser", "toc"):
            continue
        t = norm(text_of(ab))
        if t:
            out.append(("ABSTRACT", "Abstract", t))
    body = root.find(".//body")
    if body is not None:
        walk(body, None, [], out, top=True)
    back = root.find(".//back")
    if back is not None:
        for app in back.iter("app"):
            title = app.findtext("title") or "Appendix"
            if R.SKIP_SEC.search(title):
                continue
            walk(app, "APPENDIX", [title], out)
    n_suppl = len(root.findall(".//supplementary-material"))
    return out, n_suppl


def sentences(par: str):
    return [s.strip() for s in SPLIT.split(par) if s.strip()]


def main(sample_csv=PILOT / "pilot_sample.csv", version="v10"):
    screen = R.screen_sentence if version == "v10" else R.screen_sentence_v11
    suffix = "" if version == "v10" else "_" + version
    rows = [r for r in csv.DictReader(open(sample_csv, encoding="utf-8"))
            if r.get("fulltext_status", "screened") == "screened"]
    hits, papers = [], []
    for r in rows:
        pmcid = r["pmcid"]
        pars, n_suppl = paragraphs((FT / f"{pmcid}.xml").read_text(encoding="utf-8"))
        n_sent = 0
        sec_words = {}
        for (cls, title, par) in pars:
            sec_words[cls] = sec_words.get(cls, 0) + len(par.split())
            for s in sentences(par):
                n_sent += 1
                for (pid, fam, lvl, mt) in screen(s):
                    hits.append({"area": r["area"], "pmcid": pmcid, "section": cls, "section_title": title[:80],
                                 "own_work_section": int(cls in OWN), "pattern": pid, "family": fam,
                                 "level": lvl, "match": mt, "has_cit": int("[CIT]" in s),
                                 "future_or_negation": int(bool(R.FUTURE_NEG.search(s))),
                                 "sentence": s[:700]})
        ph = [h for h in hits if h["pmcid"] == pmcid]
        own = [h for h in ph if h["own_work_section"]]
        u_lvl = max([h["level"] for h in own if h["family"] == "U"], default=0)
        papers.append({"area": r["area"], "pmcid": pmcid, "n_paragraphs": len(pars), "n_sentences": n_sent,
                       "words_methods": sec_words.get("METHODS", 0),
                       "words_results": sec_words.get("RESULTS", 0) + sec_words.get("RESULTS_DISCUSSION", 0),
                       "words_body_other": sec_words.get("BODY_OTHER", 0),
                       "n_supplementary_files": n_suppl,
                       "hits_total": len(ph), "hits_own_work": len(own),
                       "hits_U1": sum(h["pattern"] == "U1_spread" for h in ph),
                       "hits_U2": sum(h["family"] == "U" and h["level"] == 2 for h in ph),
                       "hits_U3": sum(h["family"] == "U" and h["level"] == 3 for h in ph),
                       "hits_E1": sum(h["family"] == "E" for h in ph),
                       "hits_EP": sum(h["family"] == "EP" for h in ph),
                       "hits_S1": sum(h["family"] == "S" for h in ph),
                       "regex_U": f"U{u_lvl}",
                       "regex_E1": int(any(h["family"] == "E" for h in own)),
                       "regex_EP": int(any(h["family"] == "EP" for h in own)),
                       "regex_S1": int(any(h["family"] == "S" for h in own))})
    with open(PILOT / f"screen_hits{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hits[0].keys()))
        w.writeheader()
        for i, h in enumerate(hits):
            w.writerow(h)
    with open(PILOT / f"screen_papers{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(papers[0].keys()))
        w.writeheader()
        w.writerows(papers)
    return hits, papers


if __name__ == "__main__":
    hits, papers = main(version=(sys.argv[1] if len(sys.argv) > 1 else "v10"))
    import collections
    print(len(papers), "papers;", len(hits), "candidate (sentence, pattern) hits")
    print(collections.Counter(h["pattern"] for h in hits).most_common())
    print(collections.Counter(h["section"] for h in hits).most_common())
    for p in papers:
        print(p["area"][:3], p["pmcid"], p["n_sentences"], "sent;", p["hits_total"], "hits;",
              p["regex_U"], "E1" if p["regex_E1"] else "--", "EP" if p["regex_EP"] else "--",
              "S1" if p["regex_S1"] else "--", "| BODY_OTHER words", p["words_body_other"])
