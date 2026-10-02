#!/usr/bin/env python3
"""Monitor blindado MAX — multi-corrida (grid + 9 runs + ensemble).

- Baja campaign_status + train_log del run activo cada poll (180 s).
- ckpt del run activo: al avanzar época + forzado 30 min.
- Driver/campeón muerto con VM viva → relanza campaign (resume).
- VM muerta (3× PID): 1 intento de sesión nueva + re-entry completo
  (pip, uploads con split de ckpt por partes, resume). Si new falla →
  event + espera (nunca bucle).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

SESSION_DEFAULT = "diat-max"
REMOTE = "/content/ckpt"
ROOT = Path("data 🧬")
CODE = ROOT / "code"
LOCAL = ROOT / "checkpoints" / "max"
STATE = ROOT / "state" / "state.json"
SPLIT_MAX = ROOT / "state" / "split_single.json"
REPORTS = ROOT / "reports"
POLL = 180
FORCE_SEC = 1800
PT_PART = 30_000_000

LOCAL.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def session() -> str:
    try:
        return json.loads(STATE.read_text()).get("session") or SESSION_DEFAULT
    except Exception:
        return SESSION_DEFAULT


def cli(args, tries=5, timeout=180, stdin=None):
    delay = 5
    err = ""
    for i in range(tries):
        try:
            r = subprocess.run(
                ["colab"] + args, capture_output=True, text=True,
                timeout=timeout, input=stdin,
            )
            if r.returncode == 0:
                return r.stdout
            err = (r.stderr or r.stdout or "").strip()[:200]
            if any(s in err.lower() for s in
                   ("not found", "no such file", "404", "401")):
                return None
        except Exception as e:
            err = str(e)[:200]
        print(f"  cli retry {i+1}/{tries} in {delay}s: {err}", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 80)
    return None


def dl(remote, local: Path) -> bool:
    out = cli(["download", "-s", session(), remote, str(local)])
    return out is not None and local.exists()


def remote_ok(code: str) -> bool:
    delay = 5
    for i in range(3):
        try:
            r = subprocess.run(
                ["colab", "exec", "-s", session(), "--timeout", "30"],
                input=code, capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(delay)
        delay *= 2
    return False


def pid_alive(pid: int) -> bool:
    # pgrep por la command real: ps -p miente con zombis <defunct> y PIDs reciclados
    return remote_ok(
        "import subprocess,sys\n"
        "r=subprocess.run(['pgrep','-af','python3 /content/campaign.py'],"
        "capture_output=True,text=True)\n"
        "sys.exit(0 if 'campaign.py' in r.stdout else 1)\n"
    )


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
        p = Path.home() / ".config" / "opencode" / "sofia" / "events.jsonl"
        with open(p, "a") as f:
            f.write(json.dumps({"ts": int(time.time() * 1000),
                                "kind": kind, "detail": detail}) + "\n")
    except Exception:
        pass


def launch_campaign_remote() -> bool:
    code = (
        "import subprocess,sys,os\n"
        "os.makedirs('/content/ckpt',exist_ok=True)\n"
        "logf=open('/content/ckpt/campaign_run.out','a')\n"
        "p=subprocess.Popen([sys.executable,'/content/single_campaign.py'],"
        "stdout=logf,stderr=subprocess.STDOUT,start_new_session=True)\n"
        "open('/content/ckpt/campaign_pid.txt','w').write(str(p.pid))\n"
        "print('PID:',p.pid)\n"
    )
    return remote_ok(code)


def split_upload(remote_dir: str, local_pt: Path, sname: str) -> bool:
    tmp = Path("/tmp") / f"pt_{local_pt.name}"
    data = local_pt.read_bytes()
    parts = [data[i:i + PT_PART] for i in range(0, len(data), PT_PART)]
    for i, chunk in enumerate(parts):
        part = Path("/tmp") / f"{local_pt.name}.p{i}"
        part.write_bytes(chunk)
        if cli(["upload", "-s", sname, str(part),
                f"{remote_dir}/{local_pt.name}.p{i}"]) is None:
            return False
    names = ",".join(f"{local_pt.name}.p{i}" for i in range(len(parts)))
    code = (
        "import pathlib\n"
        f"parts='{names}'.split(',')\n"
        f"out=pathlib.Path('{remote_dir}/{local_pt.name}')\n"
        "with open(out,'wb') as w:\n"
        "    for p in parts:\n"
        f"        w.write(open('{remote_dir}/'+p,'rb').read())\n"
        "        pathlib.Path('{remote_dir}/'+p).unlink()\n"
        "print('asm',out.stat().st_size)\n"
    )
    return remote_ok(code)


def reentry_new_session() -> bool:
    """1 intento de sesión nueva + subida mínima + relanzar campaign."""
    st = load_state()
    n = int(st.get("session_n") or 1) + 1
    sname = f"diat-max{n}"
    print(f"VM muerta → re-entry con sesión nueva {sname} (1 intento)", flush=True)
    out = subprocess.run(["colab", "new", "-s", sname, "--gpu", "T4"],
                         capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        err = (out.stderr or out.stdout or "")[:300]
        emit_event("diat_max_reentry_failed",
                   f"colab new {sname} falló: {err}")
        save_state(status="vm_dead_need_human", session=sname,
                   session_n=n, reentry_error=err)
        print("colab new falló → event + parada (espera orden)", flush=True)
        return False

    def u(local, remote):
        return cli(["upload", "-s", sname, str(local), remote]) is not None

    subprocess.run(["colab", "exec", "-s", sname, "--timeout", "300"],
                   input=("import subprocess,sys;"
                          "subprocess.run([sys.executable,'-m','pip','install',"
                          "'-q','timm','torchvision'])"),
                   capture_output=True, text=True, timeout=360)
    cli(["exec", "-s", sname, "--timeout", "60"],
        stdin="import os;os.makedirs('/content/ckpt',exist_ok=True)")
    ok = all([
        u(CODE / "train_max.py", "/content/train_max.py"),
        u(CODE / "single_campaign.py", "/content/single_campaign.py"),
        u(SPLIT_MAX, "/content/split_single.json"),
    ])
    if not ok:
        emit_event("diat_max_reentry_failed", f"uploads fallaron en {sname}")
        save_state(status="vm_dead_need_human", session=sname, session_n=n)
        return False

    active = load_state().get("active_run")
    if active:
        src = LOCAL / active / "last.pt"
        if src.exists():
            cli(["exec", "-s", sname, "--timeout", "60"],
                stdin=f"import os;os.makedirs('/content/ckpt/{active}',exist_ok=True)")
            split_upload(f"/content/ckpt/{active}", src, sname)
        st_txt = LOCAL / active / "train_log.csv"
        if st_txt.exists():
            lines = [l for l in st_txt.read_text().splitlines() if l.strip()]
            n_ep = 0
            for l in lines[1:]:
                try:
                    n_ep = max(n_ep, int(l.split(",")[1]))
                except Exception:
                    pass
            prefix = "\n".join(lines[: 1 + n_ep])
            p = Path("/tmp") / "log_prefix.csv"
            p.write_text(prefix)
            u(p, f"/content/ckpt/{active}/train_log.csv")

    if not launch_campaign_remote():
        emit_event("diat_max_reentry_failed", "campaign relaunch falló")
        save_state(status="vm_dead_need_human", session=sname, session_n=n)
        return False
    save_state(session=sname, session_n=n, status="resumed",
               reentry_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    emit_event("diat_max_resumed", f"re-entry ok en {sname}")
    print(f"re-entry OK en {sname}", flush=True)
    return True


def sync_run_dir(st: dict) -> None:
    active = st.get("active_run") or ""
    if not active:
        return
    rdir = LOCAL / active
    rdir.mkdir(parents=True, exist_ok=True)
    tmp = rdir / "train_log.csv.tmp"
    if dl(f"{REMOTE}/{active}/train_log.csv", tmp):
        try:
            cur = rdir / "train_log.csv"
            old_n = len(cur.read_text().splitlines()) if cur.exists() else 0
            lines = [l for l in tmp.read_text().splitlines() if l.strip()]
            if len(lines) > old_n:
                tmp.replace(cur)
                last_row = lines[-1].split(",")
                dl(f"{REMOTE}/{active}/last.pt", rdir / "last.pt")
                if dl(f"{REMOTE}/{active}/best.pt", rdir / "best.pt"):
                    pass
                save_state(active_run=active,
                           active_log_tail=",".join(lines[-1:]),
                           active_local_rows=len(lines))
                print(f"  synced {active} rows={len(lines)} "
                      f"stage={last_row[0] if last_row else '?'}", flush=True)
        except Exception as e:
            print(f"  run sync: {e}", flush=True)


def main() -> int:
    print(f"watch MAX session={session()} local={LOCAL}", flush=True)
    last_force = 0.0
    dead_streak = 0
    campaign_pid = None
    done_reported = False

    while True:
        sname = session()
        st_remote = Path("/tmp/watch_campaign_status.json")
        if dl(f"{REMOTE}/campaign_status.json", st_remote):
            try:
                cs = json.loads(st_remote.read_text())
                save_state(campaign_state=cs.get("state"),
                           active_run=cs.get("current"),
                           campaign_detail=cs.get("detail"))
            except Exception:
                pass
        if dl(f"{REMOTE}/grid_winner.json", LOCAL / "grid_winner.json"):
            try:
                gw = json.loads((LOCAL / "grid_winner.json").read_text())
                save_state(grid_winner=gw)
            except Exception:
                pass

        sync_run_dir(load_state())

        if dl(f"{REMOTE}/campaign_pid.txt", LOCAL / "campaign_pid.txt"):
            try:
                campaign_pid = int(
                    (LOCAL / "campaign_pid.txt").read_text().strip())
            except Exception:
                pass
        if dl(f"{REMOTE}/metrics_ensemble.json",
              LOCAL / "metrics_ensemble.json"):
            try:
                m = json.loads((LOCAL / "metrics_ensemble.json").read_text())
                save_state(ensemble_plain=m.get("ensemble_plain"),
                           ensemble_tta=m.get("ensemble_tta"),
                           gate_pass_max=m.get("gate_pass"),
                           status="max_ensemble_ready")
                print(f"ENSEMBLE plain={m.get('ensemble_plain')} "
                      f"tta={m.get('ensemble_tta')}", flush=True)
                done_reported = True
            except Exception:
                pass
        if not done_reported and dl(f"{REMOTE}/CAMPAIGN_DONE",
                                    LOCAL / "CAMPAIGN_DONE"):
            save_state(status="max_campaign_done")
            print("CAMPAIGN_DONE", flush=True)
            done_reported = True

        if time.time() - last_force > FORCE_SEC:
            active = load_state().get("active_run")
            if active:
                dl(f"{REMOTE}/{active}/last.pt", LOCAL / active / "last.pt")
            last_force = time.time()

        if campaign_pid and pid_alive(campaign_pid):
            dead_streak = 0
        elif campaign_pid:
            if load_state().get("status") in (
                    "max_campaign_done", "max_ensemble_ready"):
                print("campaign terminado — watch sale", flush=True)
                return 0
            dead_streak += 1
            print(f"  campaign PID miss {dead_streak}/3 "
                  f"(pid={campaign_pid})", flush=True)
            if dead_streak >= 3:
                if remote_ok("print(1)"):
                    print("VM viva, driver muerto → relanzar campaign", flush=True)
                    if launch_campaign_remote():
                        dead_streak = 0
                        save_state(status="campaign_relaunched")
                        continue
                save_state(status="vm_dead_max",
                           active_run=load_state().get("active_run"))
                (REPORTS / "DEAD_max_campaign.md").write_text(
                    f"campaign dead at {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                    f"session={sname}\n")
                emit_event("diat_max_dead",
                           f"session={sname} "
                           f"run={load_state().get('active_run')} "
                           "→ re-entry automático")
                if reentry_new_session():
                    dead_streak = 0
                    campaign_pid = None
                    continue
                emit_event("diat_max_watch_exit",
                           "re-entry falló — watch sale, espera orden")
                return 2

        time.sleep(POLL)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
