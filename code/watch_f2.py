#!/usr/bin/env python3
"""Monitor blindado F2 — descarga incremental al Mac. Sesión diat-f2."""
import json
import subprocess
import sys
import time
from pathlib import Path

SESSION = "diat-f2"
REMOTE = "/content/ckpt"
ROOT = Path("data 🧬")
LOCAL = ROOT / "checkpoints" / "f2"
STATE = ROOT / "state" / "state.json"
REPORTS = ROOT / "reports"
POLL = 180
FORCE_SEC = 1800

LOCAL.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def cli(args, tries=5, timeout=120):
    delay = 5
    err = ""
    for i in range(tries):
        try:
            r = subprocess.run(
                ["colab"] + args, capture_output=True, text=True, timeout=timeout
            )
            if r.returncode == 0:
                return r.stdout
            err = (r.stderr or r.stdout or "").strip()[:200]
            if "not found" in err.lower() or "no such file" in err.lower():
                return None
        except Exception as e:
            err = str(e)[:200]
        print(f"  cli retry {i+1}/{tries} in {delay}s: {err}", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 80)
    return None


def dl(remote, local: Path) -> bool:
    out = cli(["download", "-s", SESSION, remote, str(local)])
    return out is not None and local.exists()


def remote_ok(code_snippet: str) -> bool:
    delay = 5
    for i in range(3):
        try:
            r = subprocess.run(
                ["colab", "exec", "-s", SESSION, "--timeout", "30"],
                input=code_snippet,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(delay)
        delay *= 2
    return False


def pid_alive(pid: int) -> bool:
    code = (
        "import subprocess,sys\n"
        f"r=subprocess.run(['ps','-p','{pid}','-o','pid='],capture_output=True,text=True)\n"
        "sys.exit(0 if r.stdout.strip() else 1)\n"
    )
    return remote_ok(code)


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(**kw):
    st = load_state()
    st.update(kw)
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    STATE.write_text(json.dumps(st, indent=2))


def emit_event(kind: str, detail: str) -> None:
    try:
        path = Path.home() / ".config" / "opencode" / "sofia" / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": int(time.time() * 1000),
                        "kind": kind,
                        "detail": detail,
                    }
                )
                + "\n"
            )
    except Exception:
        pass


def main() -> int:
    print(f"watch F2 session={SESSION} local={LOCAL}", flush=True)
    local_epoch = int(load_state().get("epoch_f2") or 0)
    last_force = 0.0
    dead_streak = 0
    pid = load_state().get("pid")

    if not pid:
        tmp = LOCAL / "pid.txt"
        if dl(f"{REMOTE}/pid.txt", tmp):
            try:
                pid = int(tmp.read_text().strip())
                save_state(pid=pid)
            except Exception:
                pid = None

    while True:
        try:
            fresh = load_state()
            if fresh.get("pid") and fresh.get("pid") != pid:
                print(f"  pid actualizado {pid} -> {fresh.get('pid')}", flush=True)
                pid = fresh.get("pid")
                dead_streak = 0
            if fresh.get("epoch_f2") and int(fresh.get("epoch_f2")) > local_epoch:
                local_epoch = int(fresh.get("epoch_f2"))
        except Exception:
            pass

        log_remote = LOCAL / "train_log.csv"
        if dl(f"{REMOTE}/train_log.csv", log_remote):
            try:
                lines = [ln for ln in log_remote.read_text().splitlines() if ln.strip()]
                if len(lines) > 1:
                    epoch = int(lines[-1].split(",")[0])
                    if epoch > local_epoch:
                        dl(f"{REMOTE}/last.pt", LOCAL / "last.pt")
                        dl(f"{REMOTE}/best.pt", LOCAL / "best.pt")
                        local_epoch = epoch
                        last_force = time.time()
                        save_state(
                            epoch_f2=epoch,
                            last_ckpt_synced_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            status="training_f2",
                            session=SESSION,
                            log_tail_f2=lines[-1],
                        )
                        print(f"  synced epoch {epoch}", flush=True)
            except Exception as e:
                print(f"  log parse: {e}", flush=True)

        tmp_m = LOCAL / "metrics.json"
        if dl(f"{REMOTE}/metrics.json", tmp_m):
            try:
                m = json.loads(tmp_m.read_text())
                if m.get("finished_at"):
                    dl(f"{REMOTE}/best.pt", LOCAL / "best.pt")
                    dl(f"{REMOTE}/last.pt", LOCAL / "last.pt")
                    dl(f"{REMOTE}/train.log", LOCAL / "train.log")
                    dl(f"{REMOTE}/train_log.csv", LOCAL / "train_log.csv")
                    save_state(
                        status="f2_metrics_ready",
                        best_val_acc_f2=m.get("best_val_acc"),
                        test_acc_f2=m.get("test_acc"),
                        gate_pass_f2=m.get("gate_pass"),
                    )
                    print("FIN metrics synced", flush=True)
                    return 0
            except Exception:
                pass

        if time.time() - last_force > FORCE_SEC:
            dl(f"{REMOTE}/last.pt", LOCAL / "last.pt")
            last_force = time.time()

        if pid:
            if pid_alive(pid):
                dead_streak = 0
            else:
                dead_streak += 1
                print(f"  PID miss {dead_streak}/3 (pid={pid})", flush=True)
                if dead_streak >= 3:
                    save_state(status="vm_dead_f2", epoch_f2=local_epoch, pid=pid)
                    (REPORTS / f"DEAD_f2_at_epoch_{local_epoch}.md").write_text(
                        f"VM F2 dead at epoch {local_epoch} "
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                        f"Resume: fase nueva + last.pt de checkpoints/f2.\n"
                    )
                    emit_event(
                        "diat_f2_dead",
                        f"session={SESSION} epoch={local_epoch} "
                        f"last_pt=data 🧬/checkpoints/f2/last.pt",
                    )
                    msg = f"VM Colab F2 muerta en epoch {local_epoch} — resume"
                    try:
                        subprocess.run(
                            [
                                "osascript",
                                "-e",
                                f'display notification "{msg}" with title "SOF-IA F2"',
                            ],
                            timeout=10,
                            capture_output=True,
                        )
                    except Exception:
                        pass
                    print("DEAD confirmado — monitor sale. Resume desde last.pt local.", flush=True)
                    return 2

        time.sleep(POLL)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
