#!/usr/bin/env python3
"""Artifact consistency gate.

The audit found the same failure three times in different clothes: the code, the
shipped engine and the report drifted apart, and nothing complained. This script
is the complaint. It asserts that what the repository says is true matches what
the binaries and reports actually contain.

    python3 code/check_artifacts.py

Exit code 0 = everything consistent. Any FAIL is drift that must be fixed before
publishing.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYZ = ROOT / "zar_manah_diatom_tdi.pyz"
APP = ROOT / "Diatomeas.app" / "Contents" / "Resources"
CKPT = ROOT / "checkpoints" / "f2_edited" / "best_edited.pt"
SPLIT_F1_HASH = "b6a0cb94aab2dd544e88de5d9c0f1c81a78aa59217c3081baf125943874f898d"

# pyz member -> repository source it must be byte-identical to
PYZ_MEMBERS = {
    "tdi.py": ROOT / "code" / "tdi.py",
    "index.html": ROOT / "app" / "index.html",
    "__main__.py": ROOT / "packaging" / "__main__.py",
    "tdi_values_101.csv": ROOT / "data" / "tdi_values_101.csv",
    "morphometric_prior_101.json": ROOT / "data" / "morphometric_prior_101.json",
}

APP_FILES = ["code/tdi.py", "app/server.py", "app/index.html", "app/sample.png",
             "data/tdi_values_101.csv", "data/morphometric_prior_101.json"]

FAILS: list[str] = []
PASSES = 0


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(Path(p).read_bytes())


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSES
    if cond:
        PASSES += 1
        print(f"  PASS  {name}")
    else:
        FAILS.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def main() -> int:
    print("\n[1] single-file engine")
    if not PYZ.exists():
        check("pyz exists", False, str(PYZ))
        return finish()
    check("pyz exists", True)
    with zipfile.ZipFile(PYZ) as zf:
        names = set(zf.namelist())
        for member, src in PYZ_MEMBERS.items():
            check(f"{member} present", member in names)
            if member in names and src.exists():
                check(f"{member} identical to {src.relative_to(ROOT)}",
                      sha_bytes(zf.read(member)) == sha_file(src),
                      f"{sha_bytes(zf.read(member))[:12]} vs {sha_file(src)[:12]}")
        check("best.pt present", "best.pt" in names)
        if "best.pt" in names:
            check("best.pt is the shipped checkpoint",
                  sha_bytes(zf.read("best.pt")) == sha_file(CKPT),
                  sha_bytes(zf.read("best.pt"))[:12])
        manifest = json.loads(zf.read("BUILD.json")) if "BUILD.json" in names else {}
        for member, meta in manifest.get("assets", {}).items():
            if member in names:
                check(f"BUILD.json sha matches {member}",
                      sha_bytes(zf.read(member)) == meta["sha256"])
        check("manifest declares the published formula",
              "25" in manifest.get("tdi_formula", "") and "sum(a*s*v)" in manifest.get("tdi_formula", ""))

    print("\n[2] native app bundle carries the same code")
    if not APP.exists():
        check("Diatomeas.app resources", False, "bundle not found")
    else:
        for rel in APP_FILES:
            src, dst = ROOT / rel, APP / rel
            check(f"app {rel}", dst.exists() and sha_file(dst) == sha_file(src))
        app_ckpt = APP / "checkpoints" / "f2_edited" / "best_edited.pt"
        check("app checkpoint is the shipped one",
              app_ckpt.exists() and sha_file(app_ckpt) == sha_file(CKPT))
        stale = APP / "checkpoints" / "f2" / "best.pt"
        check("app has no misleading f2/best.pt", not stale.exists())
        check("app server reports tdi100",
              "tdi100" in (APP / "app" / "server.py").read_text())
        check("app server sweeps 1200 windows",
              "max_windows=1200" in (APP / "app" / "server.py").read_text())

    print("\n[3] reports belong to the shipped checkpoint")
    ship_sha = sha_file(CKPT)
    val = ROOT / "reports" / "validacion_tdi.json"
    if not val.exists():
        check("validacion_tdi.json exists", False)
    else:
        rep = json.loads(val.read_text())
        check("report exists", True)
        check("report checkpoint sha", rep.get("checkpoint_sha256") == ship_sha,
              f"{str(rep.get('checkpoint_sha256'))[:12]} vs {ship_sha[:12]}")
        check("report checkpoint path", rep.get("checkpoint", "").endswith("best_edited.pt"),
              rep.get("checkpoint", ""))
        split_hash = json.loads((ROOT / "state" / "split_f1.json").read_text())["hash"]
        check("report split hash matches state/split_f1.json",
              rep.get("split_hash") == split_hash)
        check("report has 294 groups", rep.get("sites") == 294, str(rep.get("sites")))
        check("report uses the 0..100 scale", "25*raw - 25" in rep.get("tdi_scale", ""))
        check("report publishes band agreement", "band_exact" in rep and "band_within1" in rep)

    fuga = ROOT / "reports" / "auditoria_fuga.json"
    if fuga.exists():
        blob = json.loads(fuga.read_text())
        shas = [c["checkpoint_sha256"] for c in blob.get("checkpoints", [])]
        check("leak report covers the shipped checkpoint", ship_sha in shas)
        check("leak report quantifies seen vs unseen",
              all("accuracy_micrograph_unseen" in c for c in blob.get("checkpoints", [])))
    else:
        check("auditoria_fuga.json exists", False)

    print("\n[4] splits")
    sg = json.loads((ROOT / "state" / "split_grouped.json").read_text())
    keys = ("classes", "train", "val", "test", "seed", "min_per_class",
            "max_classes", "images_prefix", "grouping")
    recomputed = hashlib.sha256(
        json.dumps({k: sg[k] for k in keys}, sort_keys=True).encode()
    ).hexdigest()
    check("split_grouped hash recomputes", recomputed == sg["hash"],
          f"{recomputed[:12]} vs {sg['hash'][:12]}")

    def slide_of(rel: str) -> str:
        stem = rel.split("/")[-1].split(".Chunk")[0].split(".x_")[0]
        tokens = [t for t in stem.split("_") if not re.match(r"^\d{8}$", t)]
        return "_".join(tokens[1:]) if len(tokens) > 1 else (tokens[0] if tokens else "?")

    train_slides = {slide_of(r) for r, _ in sg["train"]}
    test_slides = {slide_of(r) for r, _ in sg["test"]}
    check("split_grouped shares no slide", not (train_slides & test_slides),
          str(len(train_slides & test_slides)))

    sf = json.loads((ROOT / "state" / "split_f1.json").read_text())
    check("split_f1 untouched", sf["hash"] == SPLIT_F1_HASH, sf["hash"][:12])
    leak_slides = {slide_of(r) for r, _ in sf["test"]} & {slide_of(r) for r, _ in sf["train"]}
    check("split_f1 leak documented (should be 294 shared slides)", len(leak_slides) == 294,
          str(len(leak_slides)))

    print("\n[5] per-taxon table")
    rows = list(__import__("csv").DictReader((ROOT / "data" / "tdi_values_101.csv").open()))
    check("101 rows", len(rows) == 101, str(len(rows)))
    check("indicator_value column present", "indicator_value" in rows[0])
    with_s = [r for r in rows if r["tdi_value"].strip()]
    with_v = [r for r in with_s if r["indicator_value"].strip()]
    check("79 taxa with sensitivity", len(with_s) == 79, str(len(with_s)))
    check("74 taxa with s and v", len(with_v) == 74, str(len(with_v)))
    check("v only takes 1..3", {r["indicator_value"] for r in with_v} <= {"1", "2", "3"})

    return finish()


def finish() -> int:
    print("\n" + "=" * 60)
    if FAILS:
        print(f"{PASSES} passed, {len(FAILS)} FAILED")
        for f in FAILS:
            print("  -", f)
        return 1
    print(f"all {PASSES} consistency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
