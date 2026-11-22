#!/usr/bin/env python3
"""Audit: how much does the random split inflate the test accuracy?

The shipped checkpoint was trained with `build_split.py`, which shuffles cutout
filenames. As a result, many test cutouts are crops of a microscope image that
is also present in train. This script splits the sealed test set into

    seen   - the source micrograph (.Chunk file) also appears in train
    unseen - the source micrograph is exclusive to test

and reports the accuracy on each half. The gap is the part of the headline
accuracy that comes from having seen the same photograph during training.

It does not retrain or modify anything. Run:

    python3 code/audit_leakage.py [--ckpt checkpoints/f2_edited/best_edited.pt]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from tdi import DiatomClassifier  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_of(rel: str) -> str:
    return rel.split(".Chunk")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=None,
                    help="repeatable; defaults to the checkpoint shipped with the app")
    ap.add_argument("--split", default=str(ROOT / "state" / "split_f1.json"),
                    help="split to audit (default: the leaky sealed split_f1)")
    args = ap.parse_args()
    ckpts = [Path(c).expanduser() for c in (args.ckpt or ["checkpoints/f2_edited/best_edited.pt"])]
    ckpts = [c if c.is_absolute() else (ROOT / c).resolve() for c in ckpts]

    split_path = Path(args.split).expanduser()
    if not split_path.is_absolute():
        split_path = (ROOT / split_path).resolve()
    split = json.load(split_path.open())
    test = split["test"]
    train_images = {image_of(r) for r, _ in split["train"]}
    test_images = [image_of(r) for r, _ in test]
    truth = [y for _, y in test]
    seen = [img in train_images for img in test_images]

    results = []
    for ckpt in ckpts:
        ckpt_sha = sha256(ckpt)
        tag = ckpt.stem + "_" + ckpt_sha[:8]
        cache = ROOT / "reports" / f"_pred_cache_{tag}.json"
        pred = None
        if cache.exists():
            blob = json.loads(cache.read_text())
            pred = blob["pred"] if blob.get("ckpt_sha") == ckpt_sha else None
        if pred is None:
            clf = DiatomClassifier(ckpt=ckpt)
            pred, batch = [], []
            t0 = time.perf_counter()
            for rel, _ in test:
                p = ROOT / "data" / rel
                if not p.exists():
                    continue
                batch.append(Image.open(p).convert("RGB"))
                if len(batch) == 64:
                    pred.extend(clf.classify(batch, batch_size=64).argmax(1).tolist())
                    batch = []
            if batch:
                pred.extend(clf.classify(batch, batch_size=64).argmax(1).tolist())
            print(f"inference {time.perf_counter() - t0:.1f}s for {ckpt.name}")
            cache.write_text(json.dumps({"ckpt_sha": ckpt_sha, "pred": pred}))

        n_seen = sum(seen)
        n_unseen = len(seen) - n_seen
        idx_seen = [i for i, s in enumerate(seen) if s]
        idx_unseen = [i for i, s in enumerate(seen) if not s]

        def acc(idx):
            idx = list(idx)
            return sum(pred[i] == truth[i] for i in idx) / max(1, len(idx))

        a_all, a_seen, a_unseen = acc(range(len(seen))), acc(idx_seen), acc(idx_unseen)
        results.append({
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "checkpoint_sha256": ckpt_sha,
            "test_cells": len(truth),
            "cells_from_a_train_micrograph": n_seen,
            "pct_cells_from_a_train_micrograph": round(n_seen / len(seen) * 100, 2),
            "accuracy_all": round(a_all * 100, 2),
            "accuracy_micrograph_seen": round(a_seen * 100, 2),
            "accuracy_micrograph_unseen": round(a_unseen * 100, 2),
            "leak_gap_pp": round((a_seen - a_unseen) * 100, 2),
        })
        r = results[-1]
        print(f"{ckpt.name:<20} all {r['accuracy_all']}% | seen {r['accuracy_micrograph_seen']}% "
              f"| unseen {r['accuracy_micrograph_unseen']}% | gap {r['leak_gap_pp']:+.2f} pp")

    primary = results[0]
    result = {
        "method": ("test cutouts are split by whether their source micrograph (.Chunk) "
                   "also appears in train; the accuracy gap is the inflation caused by "
                   "build_split.py shuffling cutout filenames instead of grouping by slide"),
        "split": str(split_path.relative_to(ROOT)) + (
            " (random split, the one the shipped checkpoints used)"
            if split_path.name == "split_f1.json" else " (slide-grouped split)"
        ),
        "checkpoints": results,
        **{k: v for k, v in primary.items() if k != "checkpoint_sha256"},
        "checkpoint_sha256": primary["checkpoint_sha256"],
    }
    out = ROOT / "reports" / "auditoria_fuga.json"
    if split_path.name != "split_f1.json":
        # Never overwrite the audit of the leaky split with the clean one.
        out = ROOT / "reports" / f"auditoria_fuga_{split_path.stem}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\ntest cells {result['test_cells']} | "
          f"micrograph in train {result['cells_from_a_train_micrograph']} "
          f"({result['pct_cells_from_a_train_micrograph']}%)")
    print(f"report -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
