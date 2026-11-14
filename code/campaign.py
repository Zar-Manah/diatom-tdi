#!/usr/bin/env python3
"""Campaign driver MAX — grid + 9 corridas + ensemble, secuencial, resume-safe.

Orden:
  0. smoke: forward de tamaños en las 3 arquitecturas (falla temprana)
  1. grid 6 combos smoothing{0.05,0.1,0.15} x drop_path{0.0,0.2} @192, 4ep, fold0
  2. 9 corridas: 3 arqs x 3 folds (0,1,2), receta completa progresiva
  3. ensemble_eval.py

Cada paso: si /content/ckpt/<id>/DONE → skip; si last.pt → --resume.
Estado: /content/ckpt/campaign_status.json (lo baja el watch cada poll).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

CK = Path("/content/ckpt")
GRID = [
    (0.05, 0.0), (0.05, 0.2),
    (0.1, 0.0), (0.1, 0.2),
    (0.15, 0.0), (0.15, 0.2),
]
ARCHS = [
    "convnext_tiny",
    "efficientnetv2_rw_s",
    "swin_tiny_patch4_window7_224",
]
FOLDS = [0, 1, 2]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(CK / "campaign.log", "a") as f:
        f.write(line + "\n")


def status(state, current, detail=""):
    CK.joinpath("campaign_status.json").write_text(json.dumps(
        dict(state=state, current=current, detail=detail,
             updated=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=2))


def run_item(run_id: str, extra: list) -> int:
    ck_dir = CK / run_id
    ck_dir.mkdir(parents=True, exist_ok=True)
    if (ck_dir / "DONE").exists():
        log(f"skip {run_id} (DONE)")
        return 0
    cmd = [sys.executable, "/content/train_max.py", "--run-id", run_id] + extra
    last = ck_dir / "last.pt"
    if last.exists():
        cmd += ["--resume", str(last)]
        log(f"resume {run_id}")
    else:
        log(f"fresh {run_id}")
    status("running", run_id)
    with open(ck_dir / "run.out", "w") as logf:
        rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)
    if rc != 0 and not (ck_dir / "DONE").exists():
        log(f"FALLO {run_id} rc={rc} (últimas líneas):")
        try:
            tail = Path(ck_dir / "train.log").read_text()[-800:]
            log(tail)
        except Exception:
            pass
        return rc
    log(f"ok {run_id}")
    return 0


def smoke() -> bool:
    import torch
    import timm
    log("smoke: forward tamaños 192/320/448/512 en 3 arquitecturas")
    for name in ARCHS:
        for size in (192, 320, 448, 512):
            kw = dict(pretrained=False, num_classes=101)
            if name.startswith("swin"):
                kw["img_size"] = size
            try:
                m = timm.create_model(name, **kw).cuda().eval()
                x = torch.randn(2, 3, size, size).cuda()
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    y = m(x)
                assert y.shape == (2, 101)
                del m, x
                torch.cuda.empty_cache()
            except Exception as e:
                log(f"SMOKE FAIL {name}@{size}: {e}")
                return False
    log("smoke OK")
    return True


def pick_grid_winner():
    wpath = CK / "grid_winner.json"
    if wpath.exists():
        return json.loads(wpath.read_text())
    best = None
    for sm, dp in GRID:
        gid = f"grid_s{sm}_d{dp}"
        m = CK / gid / "metrics.json"
        if not m.exists():
            continue
        met = json.loads(m.read_text())
        if best is None or met["best_val_acc"] > best["best_val_acc"]:
            best = met
    if best is None:
        log("grid sin metrics → smoothing=0.1 dp default")
        best = dict(smoothing=0.1, drop_path=-1, best_val_acc=0)
    wpath.write_text(json.dumps(best, indent=2))
    log(f"grid winner: smooth={best['smoothing']} dp={best.get('drop_path')} "
        f"val={best['best_val_acc']:.2f}")
    return best


def main() -> int:
    if not Path("/content/split_max.json").exists():
        log("FATAL falta /content/split_max.json")
        status("blocked", "missing_split")
        return 2

    status("smoke", "smoke")
    if not smoke():
        status("blocked", "smoke")
        return 3

    log("=== GRID ===")
    for sm, dp in GRID:
        gid = f"grid_s{sm}_d{dp}"
        rc = run_item(gid, ["--mode", "grid", "--fold", "0",
                            "--model", "convnext_tiny",
                            "--smoothing", str(sm),
                            "--drop-path", str(dp)])
        if rc != 0:
            status("failed", gid)
            return rc
    winner = pick_grid_winner()
    sm = winner["smoothing"]
    dp = winner.get("drop_path", -1)

    log(f"=== 9 CORRIDAS (smooth={sm} dp={dp}) ===")
    for arch in ARCHS:
        for fold in FOLDS:
            rid = f"{arch.split('_')[0]}_f{fold}"
            rc = run_item(rid, ["--mode", "run", "--fold", str(fold),
                                "--model", arch, "--smoothing", str(sm),
                                "--drop-path", str(dp)])
            if rc != 0:
                status("failed", rid)
                return rc

    status("ensemble", "ensemble")
    log("=== ENSEMBLE ===")
    with open(CK / "ensemble.out", "w") as logf:
        rc = subprocess.call([sys.executable, "/content/ensemble_eval.py"],
                             stdout=logf, stderr=subprocess.STDOUT)
    if rc != 0:
        status("failed", "ensemble")
        return rc

    status("done", "campaign_done")
    (CK / "CAMPAIGN_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    log("CAMPAIGN DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
