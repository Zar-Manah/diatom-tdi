#!/usr/bin/env python3
"""Build the single-file engine (`.pyz`) for zar manah diatom tdi.

The zip carries the *same* `code/tdi.py` the tests exercise, so the shipped app
cannot compute a different index than the repository. A `BUILD.json` manifest is
written inside it with the SHA-256 of every payload, so a user can prove which
checkpoint and which per-taxon table they are running.

    python3 code/build_pyz.py [--ckpt checkpoints/f2_edited/best_edited.pt] [--out zar_manah_diatom_tdi.pyz]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "packaging"

# name inside the zip -> source path
PAYLOAD = {
    "__main__.py": STAGING / "__main__.py",
    "tdi.py": ROOT / "code" / "tdi.py",
    "index.html": ROOT / "app" / "index.html",
    "icon.svg": ROOT / "app" / "icon.svg",
    "sample.png": ROOT / "app" / "sample.png",
    "tdi_values_101.csv": ROOT / "data" / "tdi_values_101.csv",
    "morphometric_prior_101.json": ROOT / "data" / "morphometric_prior_101.json",
}

ZIP_DATE = (2026, 11, 14, 0, 0, 0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "f2_edited" / "best_edited.pt"))
    ap.add_argument("--ckpt-name", default="best.pt", help="name the checkpoint gets inside the zip")
    ap.add_argument("--out", default=str(ROOT / "zar_manah_diatom_tdi.pyz"))
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out = Path(args.out)
    missing = [str(p) for p in list(PAYLOAD.values()) + [ckpt] if not p.exists()]
    if missing:
        print("missing inputs:\n  " + "\n  ".join(missing))
        return 1

    entries = dict(PAYLOAD)
    entries[args.ckpt_name] = ckpt

    manifest = {
        "engine": "zar manah diatom tdi",
        "tdi_formula": "25 * sum(a*s*v)/sum(a*v) - 25  (Kelly & Whitton 1995; diathor/R/diat_tdi.R)",
        "bands": "<35 Excellent, 35-50 Good, 50-65 Fair, 65-85 Poor, >=85 Bad",
        "planktonic_excluded": True,
        "morphometric_prior": False,
        "assets": {name: {"sha256": sha256(p), "bytes": p.stat().st_size} for name, p in entries.items()},
    }

    tmp = out.with_suffix(out.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, path in entries.items():
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("BUILD.json", date_time=ZIP_DATE)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, json.dumps(manifest, indent=2))
    tmp.replace(out)

    print(f"built {out.name}  {out.stat().st_size / 1e6:.1f} MB")
    for name, meta in manifest["assets"].items():
        print(f"  {meta['bytes']:>10}  {name:<30} {meta['sha256'][:16]}")
    print(f"  sha256 {sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
