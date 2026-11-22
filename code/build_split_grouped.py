#!/usr/bin/env python3
"""F0b - group-aware split, the split that should have been used from the start.

`build_split.py` shuffles *cutout filenames*, so cutouts of the same microscope
image - and even of the same slide - land in train and test. This script groups
by slide (the sampling unit encoded in the UDE filename) so a slide is either
fully in train, fully in val or fully in test.

Why it matters: with the random split, 45.6% of the test cutouts come from a
micrograph that is also in train, and 100% of the test slides are in train.

Output: state/split_grouped.json. It is NOT compatible with the current
checkpoints: those were trained on the leaky split, so they have already seen
the slides that this split holds out. Use it to retrain and re-measure.

    python3 code/build_split_grouped.py --min-per-class 100 --max-classes 101
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_DEFAULT = ROOT / "data" / "UDE Diatoms in the Wild 2024" / "UDE Diatoms in the Wild 2024.csv"


def slide_of(cutout_filename: str) -> str:
    """Slide/sampling unit encoded in the UDE filename (same rule as validate_tdi)."""
    stem = cutout_filename.split(".Chunk")[0].split(".x_")[0]
    tokens = [t for t in stem.split("_") if not re.match(r"^\d{8}$", t)]
    return "_".join(tokens[1:]) if len(tokens) > 1 else (tokens[0] if tokens else "?")


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
    ap.add_argument("--csv", default=str(CSV_DEFAULT))
    ap.add_argument("--out", default=str(ROOT / "state" / "split_grouped.json"))
    ap.add_argument("--min-per-class", type=int, default=100)
    ap.add_argument("--max-classes", type=int, default=101)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-by", choices=["slide", "image"], default="slide")
    args = ap.parse_args()

    rows = []
    counts = Counter()
    genus_of = {}
    with open(args.csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
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
    c2i = {c: i for i, c in enumerate(classes)}
    prefix = "UDE Diatoms in the Wild 2024/images"

    kept = [(f"{prefix}/{fn}", c2i[lab], slide_of(fn)) for fn, lab in rows if lab in c2i]

    # Group globally, not per class: the same slide carries many species, so a
    # per-class split would put slide X in test for one taxon and in train for
    # another, which is exactly the leak we are trying to remove.
    by_group = defaultdict(list)
    for rel, yi, group in kept:
        by_group[group].append((rel, yi))

    groups = sorted(by_group)
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    n_total = len(kept)
    target_test = n_total * args.test_frac
    target_val = n_total * args.val_frac
    train, val, test = [], [], []
    got_test = got_val = 0
    for g in groups:
        items = by_group[g]
        if got_test < target_test:
            test.extend(items)
            got_test += len(items)
        elif got_val < target_val:
            val.extend(items)
            got_val += len(items)
        else:
            train.extend(items)

    test_by_class = Counter(yi for _, yi in test)
    no_test = [classes[c] for c in range(len(classes)) if test_by_class.get(c, 0) == 0]
    thin_test = [classes[c] for c in range(len(classes)) if 0 < test_by_class.get(c, 0) < 5]

    split = dict(
        classes=classes,
        genus={c: genus_of.get(c, "") for c in classes},
        train=train,
        val=val,
        test=test,
        seed=args.seed,
        min_per_class=args.min_per_class,
        max_classes=args.max_classes,
        grouping="slide",
        source="UDE Diatoms in the Wild 2024 — Zenodo 10410655 CC0 — md5 4b46497fb17693aa361e59a0fc91c8bf",
        counts={c: counts[c] for c in classes},
        images_prefix=prefix,
        total_raw_labels=len(counts),
        classes_without_test=no_test,
        classes_thin_test=thin_test,
        test_cells_per_class={classes[c]: test_by_class.get(c, 0) for c in range(len(classes))},
        note=("Group-aware split: a slide is entirely in one split. The current "
              "checkpoints were trained on the random split and cannot be "
              "evaluated honestly on this one; retrain to use it."),
    )
    blob = json.dumps(
        {k: split[k] for k in ("classes", "train", "val", "test", "seed",
                               "min_per_class", "max_classes", "images_prefix",
                               "grouping")},
        sort_keys=True,
    )
    split["hash"] = hashlib.sha256(blob.encode()).hexdigest()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split))

    # leakage self-check
    train_slides = {slide_of(r) for r, _ in train}
    test_slides = {slide_of(r) for r, _ in test}
    print(f"classes={len(classes)} train={len(train)} val={len(val)} test={len(test)}")
    print(f"slides: train={len(train_slides)} test={len(test_slides)} "
          f"shared={len(train_slides & test_slides)}")
    print(f"classes without test coverage: {len(no_test)}")
    if no_test:
        print("  " + ", ".join(no_test[:12]) + (" ..." if len(no_test) > 12 else ""))
    print(f"classes with <5 test cells:    {len(thin_test)}")
    if thin_test:
        print("  " + ", ".join(thin_test[:12]) + (" ..." if len(thin_test) > 12 else ""))
    print(f"hash={split['hash'][:16]}")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
