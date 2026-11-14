#!/usr/bin/env python3
"""F0 en el Mac: split top-N determinista desde el CSV de UDE.

Labels: species si existe si no, si no genus (taxón del paper: 542 sp + 69 géneros).
Salida: state/split_f1.json sellado con SHA256 — el mismo hash viaja dentro del
checkpoint; si no cuadra, train_f1.py aborta el resume.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

CSV_DEFAULT = (
    "data 🧬/data/"
    "UDE Diatoms in the Wild 2024/UDE Diatoms in the Wild 2024.csv"
)


def label_of(row: dict) -> str | None:
    sp = (row.get("species") or "").strip()
    ge = (row.get("genus") or "").strip()
    if sp and sp.lower() != "none":
        return sp
    if ge and ge.lower() != "none":
        return ge
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--out", default="data 🧬/state/split_f1.json")
    ap.add_argument("--min-per-class", type=int, default=100)
    ap.add_argument("--max-classes", type=int, default=101)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--images-prefix",
        default="UDE Diatoms in the Wild 2024/images",
        help="prefijo relativo tras unzip del zip UDE",
    )
    args = ap.parse_args()

    rows = []
    counts = Counter()
    genus_of = {}
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = label_of(row)
            fn = (row.get("cutout_filename") or "").strip()
            if not lab or not fn:
                continue
            counts[lab] += 1
            ge = (row.get("genus") or "").strip()
            if ge and ge.lower() != "none":
                genus_of[lab] = ge
            rows.append((fn, lab))

    eligible = [c for c, n in counts.items() if n >= args.min_per_class]
    eligible.sort(key=lambda c: (-counts[c], c))
    classes = eligible[: args.max_classes]
    if len(classes) < 10:
        raise SystemExit(f"too few classes >= {args.min_per_class}: {len(classes)}")
    c2i = {c: i for i, c in enumerate(classes)}
    prefix = args.images_prefix.strip("/")

    kept = [(f"{prefix}/{fn}", c2i[lab]) for fn, lab in rows if lab in c2i]
    by_class = defaultdict(list)
    for rel, yi in kept:
        by_class[yi].append(rel)

    rng = random.Random(args.seed)
    train, val, test = [], [], []
    for yi, rels in sorted(by_class.items()):
        rng.shuffle(rels)
        n = len(rels)
        n_test = max(1, int(n * args.test_frac))
        n_val = max(1, int(n * args.val_frac))
        test.extend((r, yi) for r in rels[:n_test])
        val.extend((r, yi) for r in rels[n_test : n_test + n_val])
        train.extend((r, yi) for r in rels[n_test + n_val :])

    split = dict(
        classes=classes,
        genus={c: genus_of.get(c, "") for c in classes},
        train=train,
        val=val,
        test=test,
        seed=args.seed,
        min_per_class=args.min_per_class,
        max_classes=args.max_classes,
        source="UDE Diatoms in the Wild 2024 — Zenodo 10410655 CC0 — md5 4b46497fb17693aa361e59a0fc91c8bf",
        counts={c: counts[c] for c in classes},
        images_prefix=prefix,
        total_raw_labels=len(counts),
    )
    blob = json.dumps(
        {k: split[k] for k in ("classes", "train", "val", "test", "seed",
                               "min_per_class", "max_classes", "images_prefix")},
        sort_keys=True,
    )
    split["hash"] = hashlib.sha256(blob.encode()).hexdigest()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split))
    print(
        f"classes={len(classes)} train={len(train)} val={len(val)} test={len(test)} "
        f"hash={split['hash'][:16]}"
    )
    print(f"top5={classes[:5]}")
    print(
        f"count min/median/max = {min(counts[c] for c in classes)}/"
        f"{sorted(counts[c] for c in classes)[len(classes)//2]}/"
        f"{max(counts[c] for c in classes)}"
    )
    print(f"saved -> {out}")

    st_path = out.parent / "state.json"
    try:
        st = json.loads(st_path.read_text())
    except Exception:
        st = {}
    st.update(
        phase="F1",
        status="split_ready",
        split_hash=split["hash"],
        classes=len(classes),
        train=len(train),
        val=len(val),
        test=len(test),
    )
    st_path.write_text(json.dumps(st, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
