#!/usr/bin/env python3
"""F3 - validate the TDI layer against the sealed test split.

The point of this script is not to re-measure classifier accuracy (that is F2's
gate). It measures how much the classifier's error *biases the index*, because
that is the error the product actually inherits:

    TDI_true  = index computed from the human-verified cell labels
    TDI_pred  = index computed from what the model predicted
    bias      = TDI_pred - TDI_true, per site and overall

Composition is also grouped by site, so the index can be checked for structure:
different sampling sites should not collapse onto one identical index.

Test split: state/split_f1.json -> test (7375 cells, hash-locked).
Output: reports/validacion_tdi.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from tdi import DiatomClassifier, TdiTable, compute_tdi, Detection  # noqa: E402

DATA = ROOT / "data" / "UDE Diatoms in the Wild 2024" / "images"
OUT = ROOT / "reports" / "validacion_tdi.json"


def site_of(filename: str) -> str:
    """Sampling site token: the UDE filenames encode it between project prefix and date.

    e.g. 'SFB_Kinzig21_SAL2_20210326_Naphrax_NH4Cl.Chunk1.x_1.y_2.png' -> 'SAL2'
    Falls back to the project prefix when no site token is present.
    """
    stem = filename.split(".Chunk")[0].split(".x_")[0]
    tokens = [t for t in stem.split("_") if not re.match(r"^\d{8}$", t)]
    # drop a leading project token
    return "_".join(tokens[1:]) if len(tokens) > 1 else (tokens[0] if tokens else "?")


def tdi_from_counts(counts: dict[str, int], table: TdiTable) -> dict:
    dets: list[Detection] = []
    for name, n in counts.items():
        idx = table.classes.index(name)
        dets.extend(Detection(idx, 0.9, (0, 0, 1, 1)) for _ in range(n))
    r = compute_tdi(dets, table)
    return {
        "tdi": r.tdi,
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
    split = json.load((ROOT / "state" / "split_f1.json").open())
    classes, test, hsh = split["classes"], split["test"], split["hash"]

    clf = DiatomClassifier()
    if clf.split_hash != hsh:
        print("ABORT: checkpoint split hash differs from the sealed split", file=sys.stderr)
        return 1
    if clf.classes != classes:
        print("ABORT: class order differs from the sealed split", file=sys.stderr)
        return 1

    table = TdiTable.load()
    print(f"checkpoint ep{clf.epoch} | device {clf.device} | {len(test)} test cells")

    imgs, truth, sites = [], [], []
    for rel, label in test:
        p = ROOT / "data" / rel
        if not p.exists():
            continue
        imgs.append(Image.open(p).convert("RGB"))
        truth.append(int(label))
        sites.append(site_of(rel.split("/")[-1]))

    cache = ROOT / "reports" / "_test_pred_cache.json"
    if cache.exists():
        pred = json.loads(cache.read_text())["pred"]
        assert len(pred) == len(truth), "cache desfasado"
        print(f"predicciones desde cache {cache.relative_to(ROOT)}")
        dt = 0.0
    else:
        t0 = time.perf_counter()
        probs = clf.classify(imgs, batch_size=64)
        dt = time.perf_counter() - t0
        pred = probs.argmax(1).tolist()
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps({"pred": pred}))

    overall_acc = sum(p == t for p, t in zip(pred, truth)) / len(truth)
    if dt:
        print(f"inference {dt:.1f}s for {len(truth)} cells ({dt / len(truth) * 1000:.0f} ms/cell)")
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
                "tdi_true": round(a["tdi"], 3),
                "tdi_pred": round(b["tdi"], 3),
                "bias": round(b["tdi"] - a["tdi"], 3),
                "abs_error": round(abs(b["tdi"] - a["tdi"]), 3),
                "class_switch_rate": round(
                    sum(pred_counts[s].values()) / max(1, sum(true_counts[s].values())), 4
                ),
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
    # bias direction consistency: does the model systematically shift the index?
    same_sign = sum(1 for b in biases if b > 0) , sum(1 for b in biases if b < 0)

    distinct_true = len({round(x["tdi_true"], 2) for x in per_site})
    distinct_pred = len({round(x["tdi_pred"], 2) for x in per_site})

    report = {
        "checkpoint_epoch": clf.epoch,
        "checkpoint_val_acc": clf.best_acc,
        "split_hash": hsh,
        "test_cells": len(truth),
        "top1_accuracy": round(overall_acc * 100, 2),
        "tdi_whole_test_true": round(tot_true["tdi"], 3) if tot_true["tdi"] else None,
        "tdi_whole_test_pred": round(tot_pred["tdi"], 3) if tot_pred["tdi"] else None,
        "value_mass_whole_test": round(tot_true["value_mass"], 4),
        "sites": len(per_site),
        "mean_abs_error_tdi": round(mean_abs, 3),
        "rmse_tdi": round(rmse, 3),
        "bias_positive_negative": list(same_sign),
        "distinct_index_values_true": distinct_true,
        "distinct_index_values_pred": distinct_pred,
        "per_site": sorted(per_site, key=lambda x: -x["cells"])[:40],
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\nTDI sobre el test completo - real {report['tdi_whole_test_true']} "
          f"vs predicho {report['tdi_whole_test_pred']}")
    print(f"masa valorada {report['value_mass_whole_test'] * 100:.1f}%")
    print(f"sitios: {report['sites']} | error medio |TDI| = {report['mean_abs_error_tdi']} "
          f"| RMSE = {report['rmse_tdi']}")
    print(f"sesgo: {same_sign[0]} sitios suben, {same_sign[1]} bajan")
    print(f"valores de índice distintos: real {distinct_true}, predicho {distinct_pred}")
    print(f"\ninforme -> {OUT.relative_to(ROOT)}")
    print("\nsitios (ordenados por nº de células):")
    print(f"  {'site':<34} {'n':>5} {'TDI real':>9} {'TDI pred':>9} {'sesgo':>7} {'clase':>8}->{'':<8}")
    for x in sorted(per_site, key=lambda x: -x["cells"])[:12]:
        print(f"  {x['site'][:33]:<34} {x['cells']:>5} {x['tdi_true']:>9.2f} "
              f"{x['tdi_pred']:>9.2f} {x['bias']:>+7.2f}  {x['quality_true']} -> {x['quality_pred']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())