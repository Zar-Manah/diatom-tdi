#!/usr/bin/env python3
"""F3 - verification suite for the TDI layer.

Run:  python3 code/test_tdi.py

Checks, in order:
  1. table integrity        - 101 classes, values in range, sources attributed
  2. index arithmetic       - abundance-weighted mean is exact on known inputs
  3. monotone response      - more eutrophic cells => higher index
  4. band mapping           - thresholds land in the right quality class
  5. unvalued handling      - taxa with no published value never get mass
  6. planktonic reporting   - planktonic fraction is separated, not folded in
  7. end-to-end on a real image - the CLI path returns a sane result

The classifier's own accuracy is NOT re-measured here: it is F2's gate
(77.23% test) and is out of scope for the TDI layer.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from tdi import (  # noqa: E402
    Detection,
    TdiTable,
    band_for,
    compute_tdi,
)

FAILURES: list[str] = []
PASSES = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSES
    if cond:
        PASSES += 1
        print(f"  PASS  {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def detections_from(counts: dict[str, int], table: TdiTable) -> list[Detection]:
    dets: list[Detection] = []
    for name, n in counts.items():
        idx = table.classes.index(name)
        for _ in range(n):
            dets.append(Detection(cls_idx=idx, conf=0.9, box=(0, 0, 1, 1)))
    return dets


def main() -> int:
    table = TdiTable.load()

    print("\n[1] table integrity")
    check("101 classes", len(table.classes) == 101, str(len(table.classes)))
    valued = [v for v in table.values if v.value is not None]
    check("majority has a published value", len(valued) >= 70, f"{len(valued)}/101")
    check(
        "values inside the published 1..5 scale",
        all(1.0 <= v.value <= 5.0 for v in valued),
    )
    check(
        "every value carries a source",
        all(v.source for v in valued),
        str([v.name for v in valued if not v.source][:3]),
    )
    check(
        "split classes and table classes agree",
        table.classes == json_classes(),
    )

    print("\n[2] index arithmetic (exact)")
    v = {t.name: t.value for t in table.values}
    olig = pick(v, 1)
    eutro = pick(v, 5)
    mid = pick(v, 3)

    r = compute_tdi(detections_from({olig: 10}, table), table)
    check("pure oligotrophic -> 1.0", abs(r.tdi - 1.0) < 1e-9, str(r.tdi))
    check("  band = Excellent", r.quality == "Excellent", r.quality)

    r = compute_tdi(detections_from({eutro: 10}, table), table)
    check("pure eutrophic -> 5.0", abs(r.tdi - 5.0) < 1e-9, str(r.tdi))
    check("  band = Bad", r.quality == "Bad", r.quality)
    check("  x10 form is 50.0", abs(r.tdi_x10 - 50.0) < 1e-9, str(r.tdi_x10))

    r = compute_tdi(detections_from({mid: 4}, table), table)
    check("mid taxon -> 3.0", abs(r.tdi - 3.0) < 1e-9, str(r.tdi))
    check("  x10 form is 30.0", abs(r.tdi_x10 - 30.0) < 1e-9, str(r.tdi_x10))
    check("  band = Fair", r.quality == "Fair", r.quality)

    r = compute_tdi(detections_from({olig: 2, eutro: 2}, table), table)
    check("equal mix -> midpoint 3.0", abs(r.tdi - 3.0) < 1e-9, str(r.tdi))

    r = compute_tdi(detections_from({olig: 1, eutro: 3}, table), table)
    check("abundance weighted -> 4.0", abs(r.tdi - 4.0) < 1e-9, str(r.tdi))

    print("\n[3] monotone response to eutrophy")
    series = []
    for k in range(0, 11):
        comp = {olig: 10 - k}
        if k:
            comp[eutro] = k
        series.append(compute_tdi(detections_from(comp, table), table).tdi)
    rising = all(b > a for a, b in zip(series, series[1:]))
    check("TDI increases strictly as eutrophic share grows", rising, str(series))
    check("range spans the full scale", series[0] == 1.0 and series[-1] == 5.0)

    print("\n[4] band mapping")
    # native 1..5 scale, Kelly & Whitton (1995) classes
    for tdi, want in [
        (1.0, "Excellent"),
        (1.9, "Excellent"),
        (2.1, "Good"),
        (2.9, "Good"),
        (3.2, "Fair"),
        (3.9, "Fair"),
        (4.2, "Poor"),
        (4.6, "Bad"),
        (5.0, "Bad"),
    ]:
        got, _ = band_for(tdi)
        check(f"band({tdi}) = {want}", got == want, got)

    print("\n[5] taxa with no published value")
    unvalued = [t.name for t in table.values if t.value is None]
    check("there are unvalued taxa to test with", len(unvalued) > 0)
    probe = unvalued[0]
    r = compute_tdi(detections_from({probe: 8, olig: 8}, table), table)
    check(
        "unvalued taxon excluded from the index",
        abs(r.tdi - 1.0) < 1e-9,
        str(r.tdi),
    )
    check("but still counted as cells", r.total_cells == 16, str(r.total_cells))
    check(
        "and reduces the reported value mass",
        abs(r.value_mass - 0.5) < 1e-9,
        str(r.value_mass),
    )
    r = compute_tdi(detections_from({probe: 8}, table), table)
    check("sample of only unvalued taxa -> no index", r.tdi is None, str(r.tdi))

    print("\n[6] planktonic reporting")
    plank = [t for t in table.classes if t.split(" ")[0].lower() in
             {"cyclotella", "stephanodiscus", "aulacoseira", "cyclostephanos",
              "thalassiosira", "melosira"} and table.values[table.classes.index(t)].value is not None]
    if plank:
        r = compute_tdi(detections_from({plank[0]: 5, olig: 5}, table), table)
        check("planktonic share reported", r.planktonic_fraction > 0.4, str(r.planktonic_fraction))
    else:
        print("  SKIP  no valued planktonic class available to test")

    print("\n[7] end-to-end on a real image")
    img = find_test_image()
    if img is None:
        print("  SKIP  no extracted UDE image available")
    else:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            out = fh.name
        proc = subprocess.run(
            [sys.executable, str(ROOT / "code" / "tdi.py"), "--image", str(img),
             "--json", "--conf", "0.3"],
            capture_output=True, text=True, cwd=ROOT,
        )
        check("CLI exits 0", proc.returncode == 0, proc.stderr[-300:])
        if proc.returncode == 0:
            import json as _json
            res = _json.loads(proc.stdout)
            check("returns counts", isinstance(res.get("counts"), dict))
            check("reports mass fraction", "value_mass" in res)
            check("reports planktonic fraction", "planktonic_fraction" in res)
            check("index or explicit None", res.get("tdi") is None or isinstance(res["tdi"], float))

    print(f"\n{'=' * 60}")
    if FAILURES:
        print(f"{PASSES} passed, {len(FAILURES)} FAILED")
        for f in FAILURES:
            print("  -", f)
        return 1
    print(f"all {PASSES} checks passed")
    return 0


def json_classes():
    import json
    return json.load((ROOT / "state" / "split_f1.json").open())["classes"]


def pick(vals: dict[str, float], target: float) -> str:
    for name, val in vals.items():
        if val == target:
            return name
    raise SystemExit(f"no taxon with value {target}")


def find_test_image() -> Path | None:
    d = ROOT / "data" / "UDE Diatoms in the Wild 2024" / "images"
    if not d.exists():
        return None
    imgs = sorted(d.glob("*.png"))
    return imgs[0] if imgs else None


if __name__ == "__main__":
    raise SystemExit(main())