#!/usr/bin/env python3
"""F3 - verification suite for the TDI layer.

Run:  python3 code/test_tdi.py

Checks, in order:
  1. table integrity        - 101 classes, s in 1..5, v in 1..3, sources attributed
  2. index arithmetic       - Kelly & Whitton formula is exact on known inputs
  3. indicator weighting    - the `v` weight changes the index as defined
  4. monotone response      - more eutrophic cells => higher index
  5. band mapping           - thresholds land in the right quality class
  6. unvalued handling      - taxa without value/weight never get mass
  7. planktonic exclusion   - planktonic cells are excluded, not folded in
  8. end-to-end on a real image - the CLI path returns a sane result

The classifier's own accuracy is NOT re-measured here: it is F2's gate and is
out of scope for the TDI layer.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from tdi import (  # noqa: E402
    PLANKTONIC_GENUS,
    Detection,
    TdiTable,
    band_for,
    compute_tdi,
    sweep_plan,
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
    usable = [v for v in valued if v.indicator is not None]
    check("majority has a published sensitivity", len(valued) >= 70, f"{len(valued)}/101")
    check("majority has a published indicator weight", len(usable) >= 70, f"{len(usable)}/101")
    check(
        "s inside the published 1..5 scale",
        all(1.0 <= v.value <= 5.0 for v in valued),
    )
    check(
        "v inside the published 1..3 scale",
        all(1.0 <= v.indicator <= 3.0 for v in usable),
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

    print("\n[2] index arithmetic (Kelly & Whitton formula, exact)")
    v = {t.name: (t.value, t.indicator) for t in table.values}
    olig = pick(v, 2)   # no benthic taxon in this 101-taxon set has s=1
    eutro = pick(v, 5)
    mid = pick(v, 3)

    r = compute_tdi(detections_from({olig: 10}, table), table)
    check("pure oligotrophic (s=2) -> raw 2.0", abs(r.tdi - 2.0) < 1e-9, str(r.tdi))
    check("  tdi100 = 25", abs(r.tdi100 - 25.0) < 1e-9, str(r.tdi100))
    check("  band = Excellent", r.quality == "Excellent", r.quality)

    r = compute_tdi(detections_from({eutro: 10}, table), table)
    check("pure eutrophic -> raw 5.0", abs(r.tdi - 5.0) < 1e-9, str(r.tdi))
    check("  tdi100 = 100", abs(r.tdi100 - 100.0) < 1e-9, str(r.tdi100))
    check("  band = Bad", r.quality == "Bad", r.quality)

    r = compute_tdi(detections_from({mid: 4}, table), table)
    check("mid sensitivity -> raw 3.0", abs(r.tdi - 3.0) < 1e-9, str(r.tdi))
    check("  tdi100 = 50", abs(r.tdi100 - 50.0) < 1e-9, str(r.tdi100))
    check("  band = Fair", r.quality == "Fair", r.quality)

    print("\n[3] indicator weight (v) is applied")
    # Two taxa with different sensitivity AND different indicator weight.
    a = next(n for n, (s, w) in v.items()
             if s == 2 and w == 1 and n.split(" ")[0].lower() not in PLANKTONIC_GENUS)
    b = next(n for n, (s, w) in v.items()
             if s == 5 and w == 3 and n.split(" ")[0].lower() not in PLANKTONIC_GENUS)
    r = compute_tdi(detections_from({a: 1, b: 1}, table), table)
    expect = (2 * 1 * 1 + 5 * 3 * 1) / (1 * 1 + 3 * 1)
    check(
        f"v-weighted mix of {a} + {b} -> {expect:.3f}",
        abs(r.tdi - expect) < 1e-9,
        str(r.tdi),
    )
    check(
        "unweighted mean (3.5) would be different",
        abs(r.tdi - 3.5) > 1e-6,
        str(r.tdi),
    )

    print("\n[4] monotone response to eutrophy")
    series = []
    for k in range(0, 11):
        comp = {olig: 10 - k}
        if k:
            comp[eutro] = k
        series.append(compute_tdi(detections_from(comp, table), table).tdi)
    rising = all(b > a for a, b in zip(series, series[1:]))
    check("TDI increases strictly as eutrophic share grows", rising, str(series))
    check("range spans the astatic scale available (2..5)", series[0] == 2.0 and series[-1] == 5.0)

    print("\n[5] band mapping (0..100 scale, Kelly & Whitton)")
    for tdi100, want in [
        (0.0, "Excellent"),
        (34.9, "Excellent"),
        (35.0, "Good"),
        (49.9, "Good"),
        (50.0, "Fair"),
        (64.9, "Fair"),
        (65.0, "Poor"),
        (84.9, "Poor"),
        (85.0, "Bad"),
        (100.0, "Bad"),
    ]:
        got, _ = band_for(tdi100)
        check(f"band({tdi100}) = {want}", got == want, got)

    print("\n[6] taxa without value or indicator weight")
    unvalued = [t.name for t in table.values
                if t.value is None and t.name.split(" ")[0].lower() not in PLANKTONIC_GENUS]
    check("there are unvalued benthic taxa to test with", len(unvalued) > 0)
    probe = unvalued[0]
    r = compute_tdi(detections_from({probe: 8, olig: 8}, table), table)
    check(
        "unvalued taxon excluded from the index",
        abs(r.tdi - 2.0) < 1e-9,
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

    no_v = [t.name for t in table.values if t.value is not None and t.indicator is None]
    if no_v:
        r = compute_tdi(detections_from({no_v[0]: 5, olig: 5}, table), table)
        check(
            "taxon with sensitivity but no indicator weight is excluded",
            abs(r.tdi - 2.0) < 1e-9 and abs(r.value_mass - 0.5) < 1e-9,
            f"{r.tdi} {r.value_mass}",
        )

    print("\n[7] planktonic exclusion")
    plank = [t.name for t in table.values
             if t.name.split(" ")[0].lower() in
             {"cyclotella", "stephanodiscus", "aulacoseira", "cyclostephanos",
              "thalassiosira", "melosira", "discostella", "conticribra"}
             and t.value is not None and t.indicator is not None]
    check("there are valued planktonic taxa to test with", len(plank) > 0, str(plank))
    if plank:
        p = plank[0]
        r = compute_tdi(detections_from({p: 5, olig: 5}, table), table)
        check(
            f"planktonic {p} does not contaminate the benthic index",
            abs(r.tdi - v[olig][0]) < 1e-9,
            str(r.tdi),
        )
        check("planktonic share reported", abs(r.planktonic_fraction - 0.5) < 1e-9,
              str(r.planktonic_fraction))
        check("planktonic cells do not count as valued mass",
              abs(r.value_mass - 0.5) < 1e-9, str(r.value_mass))


    print("\n[9] sweep covers the whole frame, never truncates")
    for w, h, mw in [(5000, 3000, 800), (5000, 3000, 1200), (4000, 3000, 800),
                     (1280, 720, 800), (1000, 1000, 800), (320, 240, 800)]:
        xs, ys, step = sweep_plan(w, h, 150, 0.5, mw)
        check(f"{w}x{h}/{mw}: last column reaches the right edge", xs[-1] + 150 >= w,
              f"{xs[-1]}+150 vs {w}")
        check(f"{w}x{h}/{mw}: last row reaches the bottom edge", ys[-1] + 150 >= h,
              f"{ys[-1]}+150 vs {h}")
        check(f"{w}x{h}/{mw}: within the window budget", len(xs) * len(ys) <= mw,
              str(len(xs) * len(ys)))
        check(f"{w}x{h}/{mw}: no duplicate origins", len(set(xs)) == len(xs) and len(set(ys)) == len(ys))

    print("\n[10] end-to-end on a real image")
    img = find_test_image()
    if img is None:
        print("  SKIP  no extracted UDE image available")
    else:
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
            check("reports the 0..100 index", "tdi100" in res)
            check(
                "index or explicit None",
                res.get("tdi") is None or isinstance(res["tdi"], float),
            )

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


def pick(vals: dict[str, tuple], target: float) -> str:
    for name, (val, ind) in vals.items():
        if val == target and ind is not None and name.split(" ")[0].lower() not in PLANKTONIC_GENUS:
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
