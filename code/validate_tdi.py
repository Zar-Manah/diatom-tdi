#!/usr/bin/env python3
"""F3 - validate the TDI layer against the sealed test split.

The point of this script is not to re-measure classifier accuracy (that is F2's
gate). It measures how much the classifier's error *biases the index*, because
that is the error the product actually inherits:

    TDI_true  = index computed from the human-verified cell labels
    TDI_pred  = index computed from what the model predicted
    bias      = TDI_pred - TDI_true, per site and overall

Test split: state/split_f1.json -> test (7375 cells, hash-locked).

Prediction caching
    Predictions are cached per checkpoint (file name carries the checkpoint
    SHA-256 prefix) and the SHA is stored inside the file. A cache from another
    checkpoint can never be reused, which is what produced the original
    mislabelled report.

    Run:  python3 code/validate_tdi.py [--ckpt checkpoints/f2/best.pt] [--fresh]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from tdi import DiatomClassifier, TdiTable, compute_tdi, Detection, TDI_BANDS  # noqa: E402

DATA = ROOT / "data" / "UDE Diatoms in the Wild 2024" / "images"
OUT = ROOT / "reports" / "validacion_tdi.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def site_of(filename: str) -> str:
    """Sampling site token: the UDE filenames encode it between project prefix and date.

    e.g. 'SFB_Kinzig21_SAL2_20210326_Naphrax_NH4Cl.Chunk1.x_1.y_2.png' -> 'SAL2'
    Falls back to the project prefix when no site token is present.
    """
    stem = filename.split(".Chunk")[0].split(".x_")[0]
    tokens = [t for t in stem.split("_") if not re.match(r"^\d{8}$", t)]
    # drop a leading project token
    return "_".join(tokens[1:]) if len(tokens) > 1 else (tokens[0] if tokens else "?")


def slide_of(filename: str) -> str:
    """Whole micrograph identity (site + date + treatment, no chunk/cutout).

    The grouped split assigns complete slides to one side, so the honest unit for
    error propagation is the slide, not the site: one site can contribute several
    slides and some of them may live in train.
    """
    return filename.split(".Chunk")[0].split(".x_")[0]


def tdi_from_counts(counts: dict[str, int], table: TdiTable) -> dict:
    dets: list[Detection] = []
    for name, n in counts.items():
        idx = table.classes.index(name)
        dets.extend(Detection(idx, 0.9, (0, 0, 1, 1)) for _ in range(n))
    r = compute_tdi(dets, table)
    return {
        "tdi": r.tdi,
        "tdi100": r.tdi100,
        "total_cells": r.total_cells,
        "valued_cells": r.valued_cells,
        "value_mass": r.value_mass,
        "quality": r.quality,
        "planktonic_fraction": r.planktonic_fraction,
    }


def merge(dd: dict[str, dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for d in dd.values():
        for k, v in d.items():
            out[k] += v
    return dict(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "f2_edited" / "best_edited.pt"))
    ap.add_argument("--split", default=str(ROOT / "state" / "split_f1.json"),
                    help="split to evaluate on (default: the sealed split_f1)")
    ap.add_argument("--group-by", choices=("site", "slide"), default="site",
                    help="aggregation unit for the index error (slide is the honest "
                         "unit for the grouped split)")
    ap.add_argument("--fresh", action="store_true", help="ignore any cached predictions")
    args = ap.parse_args()
    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_absolute():
        ckpt = (ROOT / ckpt).resolve()
    ckpt_sha = sha256(ckpt)
    tag = Path(ckpt).stem + "_" + ckpt_sha[:8]

    split_path = Path(args.split).expanduser()
    if not split_path.is_absolute():
        split_path = (ROOT / split_path).resolve()
    split = json.load(split_path.open())
    classes, test, hsh = split["classes"], split["test"], split["hash"]
    canonical = split_path.name == "split_f1.json"

    clf = DiatomClassifier(ckpt=ckpt)
    if clf.split_hash != hsh:
        print("ABORT: checkpoint split hash differs from the sealed split", file=sys.stderr)
        return 1
    if clf.classes != classes:
        print("ABORT: class order differs from the sealed split", file=sys.stderr)
        return 1

    table = TdiTable.load()
    if table.classes != clf.classes:
        print("ABORT: TDI table class order differs from the checkpoint", file=sys.stderr)
        return 1

    print(f"checkpoint {ckpt.name} sha {ckpt_sha[:12]} | ep{clf.epoch} | "
          f"device {clf.device} | split {split_path.name} | "
          f"{len(test)} test cells | group-by {args.group_by}")

    truth, sites = [], []
    group_of = slide_of if args.group_by == "slide" else site_of
    for rel, label in test:
        p = ROOT / "data" / rel
        if not p.exists():
            continue
        truth.append(int(label))
        sites.append(group_of(rel.split("/")[-1]))

    cache = ROOT / "reports" / f"_pred_cache_{tag}.json"
    pred = None
    if cache.exists() and not args.fresh:
        blob = json.loads(cache.read_text())
        if blob.get("ckpt_sha") == ckpt_sha and len(blob.get("pred", [])) == len(truth):
            pred = blob["pred"]
            print(f"predicciones desde cache {cache.name}")
    if pred is None:
        t0 = time.perf_counter()
        pred = []
        imgs: list[Image.Image] = []
        for rel, _ in test:
            p = ROOT / "data" / rel
            if not p.exists():
                continue
            imgs.append(Image.open(p).convert("RGB"))
            if len(imgs) == 64:
                pred.extend(clf.classify(imgs, batch_size=64).argmax(1).tolist())
                imgs = []
        if imgs:
            pred.extend(clf.classify(imgs, batch_size=64).argmax(1).tolist())
        print(f"inference {time.perf_counter() - t0:.1f}s for {len(truth)} cells")
        cache.write_text(json.dumps({"ckpt_sha": ckpt_sha, "pred": pred}))

    overall_acc = sum(p == t for p, t in zip(pred, truth)) / len(truth)
    print(f"top-1 accuracy on test: {overall_acc * 100:.2f}%")

    true_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pred_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s, t, p in zip(sites, truth, pred):
        true_counts[s][classes[t]] += 1
        pred_counts[s][classes[p]] += 1

    per_site = []
    for s in sorted(true_counts):
        a = tdi_from_counts(dict(true_counts[s]), table)
        b = tdi_from_counts(dict(pred_counts[s]), table)
        if a["tdi"] is None or b["tdi"] is None:
            continue
        per_site.append(
            {
                "site": s,
                "cells": a["total_cells"],
                "tdi_true": round(a["tdi"], 4),
                "tdi_pred": round(b["tdi"], 4),
                "tdi100_true": round(a["tdi100"], 1),
                "tdi100_pred": round(b["tdi100"], 1),
                "bias": round(b["tdi"] - a["tdi"], 4),
                "abs_error": round(abs(b["tdi"] - a["tdi"]), 4),
                "planktonic_true": round(a["planktonic_fraction"], 3),
                "quality_true": a["quality"],
                "quality_pred": b["quality"],
            }
        )

    tot_true = tdi_from_counts(merge(true_counts), table)
    tot_pred = tdi_from_counts(merge(pred_counts), table)

    biases = [x["bias"] for x in per_site]
    abs_err = [x["abs_error"] for x in per_site]
    mean_abs = sum(abs_err) / len(abs_err) if abs_err else float("nan")
    rmse = (sum(e * e for e in abs_err) / len(abs_err)) ** 0.5 if abs_err else float("nan")

    # correlation of the site-level index
    try:
        from scipy.stats import pearsonr, spearmanr

        xs = [x["tdi_true"] for x in per_site]
        ys = [x["tdi_pred"] for x in per_site]
        pear = float(pearsonr(xs, ys)[0])
        spear = float(spearmanr(xs, ys)[0])
    except Exception:  # pragma: no cover - scipy is optional
        pear = spear = float("nan")

    band_order = [b[1] for b in TDI_BANDS]

    def dist(a: str, b: str) -> int:
        return abs(band_order.index(a) - band_order.index(b))

    exact = sum(1 for x in per_site if x["quality_true"] == x["quality_pred"])
    within1 = sum(1 for x in per_site if dist(x["quality_true"], x["quality_pred"]) <= 1)

    report = {
        "checkpoint": str(Path(ckpt).relative_to(ROOT)),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": clf.epoch,
        "checkpoint_val_acc": clf.best_acc,
        "split": split_path.name,
        "group_by": args.group_by,
        "split_hash": hsh,
        "test_cells": len(truth),
        "top1_accuracy": round(overall_acc * 100, 2),
        "tdi_scale": "Kelly & Whitton 0..100 = 25*raw - 25 (diathor/R/diat_tdi.R)",
        "tdi_whole_test_true": round(tot_true["tdi"], 4) if tot_true["tdi"] else None,
        "tdi_whole_test_pred": round(tot_pred["tdi"], 4) if tot_pred["tdi"] else None,
        "tdi100_whole_test_true": round(tot_true["tdi100"], 1) if tot_true["tdi100"] else None,
        "tdi100_whole_test_pred": round(tot_pred["tdi100"], 1) if tot_pred["tdi100"] else None,
        "value_mass_whole_test": round(tot_true["value_mass"], 4),
        "planktonic_fraction_whole_test": round(tot_true["planktonic_fraction"], 4),
        "sites": len(per_site),
        "mean_abs_error_tdi": round(mean_abs, 4),
        "rmse_tdi": round(rmse, 4),
        "pearson_r": round(pear, 4),
        "spearman_rho": round(spear, 4),
        "r2": round(pear ** 2, 4),
        "bias_positive_negative": [sum(1 for b in biases if b > 0), sum(1 for b in biases if b < 0)],
        "band_exact": exact,
        "band_exact_pct": round(exact / len(per_site) * 100, 2),
        "band_within1": within1,
        "band_within1_pct": round(within1 / len(per_site) * 100, 2),
        "per_site": per_site,
    }
    OUT.parent.mkdir(exist_ok=True)
    blob = json.dumps(report, indent=2, ensure_ascii=False)
    if canonical:
        OUT.write_text(blob)
        # One report per checkpoint, so a later run of another checkpoint can never
        # silently replace the numbers of the shipped one (the original bug).
        (OUT.parent / f"validacion_tdi_{tag}.json").write_text(blob)
        written = OUT
    else:
        # A split other than the sealed one must never touch the canonical report
        # that the artifact gate pins to the shipped checkpoint.
        written = OUT.parent / f"validacion_{split_path.stem}_{tag}_{args.group_by}.json"
        written.write_text(blob)

    print(f"\nTDI (0..100) sobre el test completo - real {report['tdi100_whole_test_true']} "
          f"vs predicho {report['tdi100_whole_test_pred']}")
    print(f"TDI crudo (1..5)                 - real {report['tdi_whole_test_true']} "
          f"vs predicho {report['tdi_whole_test_pred']}")
    print(f"masa valorada {report['value_mass_whole_test'] * 100:.1f}% | "
          f"planctónicas {report['planktonic_fraction_whole_test'] * 100:.1f}%")
    print(f"sitios: {report['sites']} | MAE = {report['mean_abs_error_tdi']} "
          f"| RMSE = {report['rmse_tdi']} | r = {report['pearson_r']} | rho = {report['spearman_rho']}")
    print(f"sesgo: {report['bias_positive_negative'][0]} sitios suben, "
          f"{report['bias_positive_negative'][1]} bajan")
    print(f"banda de calidad exacta: {report['band_exact']}/{report['sites']} "
          f"({report['band_exact_pct']}%) | dentro de 1: {report['band_within1']}/{report['sites']} "
          f"({report['band_within1_pct']}%)")
    print(f"\ninforme -> {written.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
