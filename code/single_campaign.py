#!/usr/bin/env python3
"""Driver de UN modelo generalista MAX — grid +1 corrida. SIN ensemble, SIN k-fold.

Escribe los mismos artefactos que campaign.py (campaign_status.json,
campaign_pid.txt, CAMPAIGN_DONE) para reutilizar watch_max.py tal cual.
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
MODEL = "convnext_tiny"


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
        log(f"FALLO {run_id} rc={rc}")
        try:
            log(Path(ck_dir / "train.log").read_text()[-800:])
        except Exception:
            pass
        return rc
    log(f"ok {run_id}")
    return 0


def smoke() -> bool:
    import torch
    import timm
    log(f"smoke: {MODEL} @192/320/448/512")
    for size in (192, 320, 448, 512):
        try:
            m = timm.create_model(MODEL, pretrained=False, num_classes=101).cuda().eval()
            x = torch.randn(2, 3, size, size).cuda()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                y = m(x)
            assert y.shape == (2, 101)
            del m, x
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"SMOKE FAIL @{size}: {e}")
            return False
    log("smoke OK")
    return True


def pick_grid_winner():
    wpath = CK / "grid_winner.json"
    if wpath.exists():
        return json.loads(wpath.read_text())
    best = None
    for sm, dp in GRID:
        m = CK / f"grid_s{sm}_d{dp}" / "metrics.json"
        if not m.exists():
            continue
        met = json.loads(m.read_text())
        if best is None or met["best_val_acc"] > best["best_val_acc"]:
            best = met
    if best is None:
        best = dict(smoothing=0.1, drop_path=0.2, best_val_acc=0)
    wpath.write_text(json.dumps(best, indent=2))
    log(f"grid winner: smooth={best['smoothing']} dp={best.get('drop_path')} "
        f"val={best['best_val_acc']:.2f}")
    return best


def main() -> int:
    if not Path("/content/split_single.json").exists():
        log("FATAL falta /content/split_single.json")
        status("blocked", "missing_split")
        return 2

    status("smoke", "smoke")
    if not smoke():
        status("blocked", "smoke")
        return 3

    log("=== GRID (6 combos, UN modelo) ===")
    for sm, dp in GRID:
        gid = f"grid_s{sm}_d{dp}"
        rc = run_item(gid, ["--mode", "grid", "--fold", "0",
                            "--model", MODEL, "--smoothing", str(sm),
                            "--drop-path", str(dp)])
        if rc != 0:
            status("failed", gid)
            return rc
    winner = pick_grid_winner()

    log(f"=== MODELO ÚNICO {MODEL} (smooth={winner['smoothing']} "
        f"dp={winner.get('drop_path')}) ===")
    rc = run_item("single", ["--mode", "run", "--fold", "0",
                             "--model", MODEL,
                             "--smoothing", str(winner["smoothing"]),
                             "--drop-path", str(winner.get("drop_path", -1))])
    if rc != 0:
        status("failed", "single")
        return rc

    try:
        met = json.loads((CK / "single" / "metrics.json").read_text())
        (CK / "metrics_single.json").write_text(json.dumps(met, indent=2))
    except Exception as e:
        log(f"metrics copy fail: {e}")

    status("done", "single_done")
    (CK / "CAMPAIGN_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    log(f"UN MODELO DONE {json.dumps(met) if 'met' in dir() else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
