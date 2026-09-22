"""Step 3 - fetch Europe PMC JATS full text for the pilot sample (cached)."""
import csv
from pathlib import Path

import epmc

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "pilot" / "pilot_sample.csv", encoding="utf-8")))
ok = 0
for r in rows:
    x = epmc.fulltext_xml(r["pmcid"])
    n = len(x) if x else 0
    ok += bool(x)
    print(f"{r['area']:15s} {r['pmcid']:12s} {'OK ' if x else 'MISSING'} {n:>8d} chars")
print(f"{ok}/{len(rows)} full texts available")
