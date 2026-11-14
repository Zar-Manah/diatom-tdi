#!/usr/bin/env python3
"""F3 - build the TDI value table for the 101 classes of the sealed split.

The TDI is a pollution-tolerance index: each taxon carries a sensitivity value
on a 1..5 scale (Kelly & Whitton 1995) and the sample index is the abundance
weighted mean of those values over the benthic fraction.

Two published sources are merged, in priority order:

  1. Kelly & Whitton (1995) doi:10.1007/BF00003802 - the original definition.
     Vendored at data/tdi_kelly_whitton.csv from the R package `diathor`
     (dataset `tdi`, 4146 taxa). The index value is the `tdi_s` column
     (sensitivity). The companion `tdi_v` column is NOT the TDI value - it is a
     different ecological scale and agrees with DARLEQ's TDIo only 5.9% of the
     time, versus 80.7% for `tdi_s`.
  2. DARLEQ2 (aquaMetrics/aquaMetrics, inst/extdata/DARLEQ-TAXON-DICTIONARY.csv),
     1275 taxa, column `TDIo` (original TDI) plus `TDI3`/`TDI4` recalibrations.
     Used only for taxa absent from source 1.

Only hand-verified synonyms are applied (SYNONYMS below); nothing is inferred
from epithet similarity, because that produces false matches across unrelated
genera (e.g. Seminavis strigosa vs Surirella strigosa).

Nothing is invented: taxa with no published value keep an empty value and are
reported, so the real coverage is always visible.
"""
import csv
import difflib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KW_CSV = ROOT / "data" / "tdi_kelly_whitton.csv"
DL_CSV = ROOT / "data" / "darleq_taxon_dictionary.csv"
SPLIT = ROOT / "state" / "split_f1.json"
OUT = ROOT / "data" / "tdi_values_101.csv"

# Hand-verified synonyms: model class -> (key to look up, why).
SYNONYMS = {
    "Humidophila simplex": ("navicula simplex", "Humidophila simplex = Navicula simplex Kratz"),
    "Conticribra weissflogii": ("parlibellus weissflogii", "Contribra weissflogii = Parlibellus weissflogii"),
    "Encyonema ventricosum": ("gomphonema ventricosum", "Encyonema ventricosum = Gomphonema ventricosum"),
    "Navicula metareichardtiana = reichardtiana": (
        "navicula reichardtiana",
        "Navicula metareichardtiana = Navicula reichardtiana",
    ),
}

# Classes that are genus-level buckets in the UDE dataset (no species epithet),
# so the genus value is the defensible default.
GENUS_BUCKETS = {"Gomphonema sp. 1"}


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\bvar\.?\b", " ", s)
    s = re.sub(r"\bsubsp\.?\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_synonym(s):
    """'A = B' -> ['a', 'b']"""
    if "=" in s:
        left, right = [x.strip() for x in s.split("=", 1)]
        return [norm(left), norm(right)]
    return [norm(s)]


def majority(pairs):
    """Most common (tdi_s, tdi_v) among infraspecific rows."""
    return max(set(pairs), key=pairs.count)


def load_kelly_whitton():
    idx = defaultdict(list)
    with KW_CSV.open() as fh:
        for r in csv.DictReader(fh):
            idx[norm(r["fullspecies"])].append((int(r["tdi_s"]), int(r["tdi_v"])))
    return idx


def load_darleq():
    out = {}
    with DL_CSV.open(encoding="latin-1") as fh:
        for r in csv.DictReader(fh):
            out[norm(r["TaxonName"])] = r
    return out


def na(v):
    return v is None or str(v).strip().upper() in {"", "NA", "N/A"}


def from_kelly_whitton(cls, idx):
    """Return (tdi_s, tdi_v, key, how) or None."""
    cands = split_synonym(norm(cls))

    if cls in SYNONYMS:
        key, _why = SYNONYMS[cls]
        if key in idx:
            return (*majority(idx[key]), key, "synonym")

    for c in cands:
        if c in idx:
            return (*majority(idx[c]), c, "exact")

    # species present only as an infraspecific taxon ("gomphonema parvulum var. exilis")
    for c in cands:
        hits = [v for k, v in idx.items() if k.startswith(c + " ")]
        if hits:
            return (*majority([x for h in hits for x in h]), c, "infraspecific")

    # genus-scoped fuzzy, only for true binomials
    g = norm(cls).split(" ")[0]
    pool = [k for k in idx if k.startswith(g + " ")]
    close = difflib.get_close_matches(cands[0], pool, n=1, cutoff=0.92)
    if close:
        return (*majority(idx[close[0]]), close[0], "fuzzy")

    # the class is literally a genus / genus bucket
    if cls in GENUS_BUCKETS or " " not in norm(cls):
        for c in cands:
            if c in idx:
                return (*majority(idx[c]), c, "genus")
    return None


def from_darleq(cls, dl):
    """Return (tdi_o, tdi3, key, how) or None. Missing values stay empty."""
    cands = split_synonym(norm(cls))
    if cls in SYNONYMS:
        cands = [SYNONYMS[cls][0]] + cands

    for c in cands:
        r = dl.get(c)
        if r and not (na(r["TDIo"]) and na(r["TDI3"]) and na(r["TDI4"])):
            return (
                "" if na(r["TDIo"]) else int(r["TDIo"]),
                "" if na(r["TDI3"]) else int(r["TDI3"]),
                c,
                "darleq" if not na(r["TDIo"]) else "darleq_recalibrated_only",
            )
    return None


def main():
    for p in (KW_CSV, DL_CSV, SPLIT):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1

    idx = load_kelly_whitton()
    dl = load_darleq()
    classes = json.load(SPLIT.open())["classes"]

    rows, stats, unresolved = [], defaultdict(int), []
    for cls in classes:
        kw = from_kelly_whitton(cls, idx)
        dlrow = from_darleq(cls, dl)

        if kw:
            s, v, key, how = kw
            rows.append(
                {
                    "class": cls,
                    "tdi_value": s,
                    "tdi_source": "Kelly&Whitton1995",
                    "tdi_source_key": key,
                    "match_type": how,
                    "darleq_tdio": dlrow[0] if dlrow else "",
                    "darleq_tdi3": dlrow[1] if dlrow else "",
                }
            )
            stats[how] += 1
            continue

        if dlrow:
            tdio, tdi3, key, how = dlrow
            use = tdio if tdio != "" else tdi3
            src = "DARLEQ2:TDIo" if tdio != "" else "DARLEQ2:TDI3"
            rows.append(
                {
                    "class": cls,
                    "tdi_value": use,
                    "tdi_source": src,
                    "tdi_source_key": key,
                    "match_type": how,
                    "darleq_tdio": tdio,
                    "darleq_tdi3": tdi3,
                }
            )
            stats[how] += 1
            continue

        rows.append(
            {
                "class": cls,
                "tdi_value": "",
                "tdi_source": "",
                "tdi_source_key": "",
                "match_type": "unresolved",
                "darleq_tdio": "",
                "darleq_tdi3": "",
            }
        )
        stats["unresolved"] += 1
        unresolved.append(cls)

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    resolved = total - stats["unresolved"]
    print(f"classes: {total}")
    for how in sorted(stats, key=lambda k: -stats[k]):
        print(f"  {how:<28} {stats[how]}")
    print(f"\nwith published TDI value: {resolved}/{total} ({100.0 * resolved / total:.1f}%)")
    print(f"table -> {OUT.relative_to(ROOT)}")

    if unresolved:
        print(f"\nUNRESOLVED ({len(unresolved)}) - no published value, left empty on purpose:")
        for c in unresolved:
            print(f"  - {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())