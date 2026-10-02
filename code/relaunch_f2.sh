#!/bin/bash
set -u
LOG=data 🧬/checkpoints/f2/relaunch.log
ROOT=data 🧬
CK=$ROOT/checkpoints/f2
STATE=$ROOT/state/state.json
SESS=diat-f2
SPLIT=$ROOT/state/split_f1.json
TRAIN=$ROOT/code/train_f2.py
LAUNCH=/tmp/launch_f2_resume.py
PIDPY=/tmp/f2_pid.py

ts() { date '+%Y-%m-%dT%H:%M:%S'; }
log() { echo "[$(ts)] $*" >>"$LOG"; }

cat > "$LAUNCH" <<'PY'
import subprocess, sys
from pathlib import Path
Path("/content/ckpt").mkdir(parents=True, exist_ok=True)
logf = open("/content/ckpt/relaunch.log", "a")
cmd = [
    sys.executable, "/content/train_f2.py",
    "--split", "/content/split_f1.json",
    "--resume", "/content/ckpt/last.pt",
    "--epochs", "60", "--batch", "32", "--lr", "0.0005",
    "--model", "efficientnet_b3", "--img-size", "300", "--gate", "85",
]
print("CMD", cmd, flush=True)
p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
print("PID:", p.pid, flush=True)
Path("/content/ckpt/pid.txt").write_text(str(p.pid))
PY

printf 'print(open("/content/ckpt/pid.txt").read().strip())\n' > "$PIDPY"

log "relaunch loop start pid=$$"
DELAY=20
while true; do
  if colab sessions 2>/dev/null | grep -qF "$SESS"; then
    log "session $SESS already up"
    break
  fi
  log "try colab new delay=$DELAY"
  OUT=$(colab new -s "$SESS" --gpu T4 2>&1) || true
  if echo "$OUT" | grep -qiE 'Traceback|Service Unavailable|TooMany|quota|412'; then
    log "new fail: $(echo "$OUT" | tr '\n' ' ' | tail -c 350)"
    sleep "$DELAY"
    DELAY=$(( DELAY * 2 )); [ "$DELAY" -gt 300 ] && DELAY=300
    continue
  fi
  if colab sessions 2>/dev/null | grep -qF "$SESS"; then
    log "session created"
    break
  fi
  log "new ambiguous: $(echo "$OUT" | tr '\n' ' ' | tail -c 350)"
  sleep "$DELAY"
  DELAY=$(( DELAY * 2 )); [ "$DELAY" -gt 300 ] && DELAY=300
done

log "upload artifacts"
for pair in "$SPLIT|/content/split_f1.json" "$TRAIN|/content/train_f2.py" "$LAUNCH|/content/launch_resume.py" "$CK/last.pt|/content/ckpt/last.pt" "$CK/train_log.csv|/content/ckpt/train_log.csv" "$PIDPY|/content/f2_pid.py"; do
  src="${pair%%|*}"; dst="${pair##*|}"
  if ! colab upload -s "$SESS" "$src" "$dst" >>"$LOG" 2>&1; then
    log "upload FAIL $src"
    exit 1
  fi
  log "uploaded $dst"
done

log "exec launch_resume"
colab exec -s "$SESS" -f /content/launch_resume.py --timeout 90 >>"$LOG" 2>&1 || { log "exec FAIL"; exit 1; }
sleep 4
PID=$(colab exec -s "$SESS" -f /content/f2_pid.py --timeout 40 2>/dev/null | tr -d '\r' | tail -1 | tr -cd '0-9')
log "remote pid=$PID"
[ -z "$PID" ] && { log "no pid"; exit 1; }

python3 - "$STATE" "$SESS" "$PID" <<'PY'
import json, time, sys
from pathlib import Path
p, sess, pid = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
st = json.loads(p.read_text())
st.update({
  "status": "training_f2",
  "session": sess,
  "pid": pid,
  "epoch_f2": 9,
  "best_val_acc": 75.1051,
  "notes": "F2 RESUMED desde ép9 (75.11%) tras caída VM.",
  "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
})
p.write_text(json.dumps(st, indent=2))
print("state resumed")
PY

# endpoint for keep-alive
EP=$(colab sessions 2>/dev/null | grep -F "$SESS" | awk -F'|' '{print $1}' | sed 's/.*\[/[/;s/\]//'] | head -1 | tr -d '[]')
# format: [diat-f2] gpu-t4-...
EP=$(colab sessions 2>/dev/null | grep -F "$SESS" | sed -n 's/.*\] \([^ |]*\).*/\1/p' | head -1)
log "endpoint=$EP"
if [ -n "$EP" ]; then
  nohup colab keep-alive "$EP" "$SESS" >/dev/null 2>&1 &
  echo $! > /tmp/keepalive_cli.pid
  log "cli keep-alive pid=$!"
fi

nohup python3 "$ROOT/code/watch_f2.py" >>"$CK/watch.log" 2>&1 &
echo $! > /tmp/watch_f2.pid
nohup bash /tmp/keepalive_f2.sh >/dev/null 2>&1 &
echo $! > /tmp/keepalive_f2.pid
log "daemons watch=$(cat /tmp/watch_f2.pid) local_ka=$(cat /tmp/keepalive_f2.pid)"
log "relaunch DONE remote_pid=$PID"
echo DONE > "$CK/relaunch.done"
exit 0
