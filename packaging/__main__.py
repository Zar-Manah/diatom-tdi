#!/usr/bin/env python3
"""zar manah diatom tdi - single-file engine.

Everything the index needs lives inside this zip: the classifier checkpoint, the
per-taxon table and the UI. The index math itself is not duplicated here - it is
`tdi.py`, bundled verbatim, so the app and the repository can never drift apart.

Resource layout at runtime
    The zip is not a filesystem, so the payload is extracted once to a temp dir
    and `DIATOM_ROOT` / `DIATOM_CKPT` point `tdi.py` at it.
"""
import sys, os, subprocess, zipfile, io, time, signal, socket, threading, webbrowser, tempfile, shutil, argparse
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json


def ensure_dependencies():
    missing = []
    for mod, pkg in [('torch', 'torch'), ('torchvision', 'torchvision'), ('PIL', 'pillow'), ('timm', 'timm')]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f'[*] Faltan dependencias: {", ".join(missing)}')
        print('[*] Instalando (solo la primera vez; requiere conexion)...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-input'] + missing)


ensure_dependencies()

from PIL import Image


def find_self_zip():
    candidates = [os.path.abspath(sys.argv[0]), os.path.abspath(__file__)]
    for c in candidates:
        if os.path.exists(c) and zipfile.is_zipfile(c):
            return c
    p = os.path.abspath(__file__)
    while p:
        if zipfile.is_zipfile(p):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return os.path.abspath(sys.argv[0])


ARCHIVE_PATH = find_self_zip()
RESOURCE_DIR = tempfile.mkdtemp(prefix='diatom_tdi_')


def get_resource_bytes(name: str) -> bytes:
    with zipfile.ZipFile(ARCHIVE_PATH, 'r') as zf:
        return zf.read(name)


def extract_resources() -> None:
    """Unpack the payload where `tdi.py` expects a normal repository."""
    data_dir = os.path.join(RESOURCE_DIR, 'data')
    ckpt_dir = os.path.join(RESOURCE_DIR, 'checkpoints', 'f2_edited')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    for name in ('tdi_values_101.csv', 'morphometric_prior_101.json'):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            with open(path, 'wb') as fh:
                fh.write(get_resource_bytes(name))

    ckpt_path = os.path.join(ckpt_dir, 'best_edited.pt')
    if not os.path.exists(ckpt_path):
        with open(ckpt_path, 'wb') as fh:
            fh.write(get_resource_bytes('best.pt'))

    os.environ['DIATOM_ROOT'] = RESOURCE_DIR
    os.environ['DIATOM_CKPT'] = ckpt_path


extract_resources()
sys.path.insert(0, ARCHIVE_PATH)
from tdi import (  # noqa: E402
    DiatomClassifier,
    TdiTable,
    count_cells,
    compute_tdi,
    sweep_plan,
    CKPT,
)

LAST_HEARTBEAT = time.time()
CLIENT_ATTACHED = False
SHUTDOWN_TRIGGERED = False
IS_ANALYZING = False
CLASSIFIER = None
TDI_TABLE = None

# A 12 MP photo at stride 0.5 needs ~2600 windows; 1200 keeps a full-frame sweep
# under a few seconds on Metal while staying dense enough for 150 px cells.
MAX_WINDOWS = 1200
WINDOW = 150


def ensure_model_loaded():
    global CLASSIFIER, TDI_TABLE
    if TDI_TABLE is None:
        TDI_TABLE = TdiTable.load()
    if CLASSIFIER is None:
        CLASSIFIER = DiatomClassifier(ckpt=CKPT)


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'


def get_free_port(start_port: int = 8765) -> int:
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port


def watchdog_thread():
    global LAST_HEARTBEAT, SHUTDOWN_TRIGGERED, IS_ANALYZING
    while not SHUTDOWN_TRIGGERED:
        time.sleep(2.0)
        if CLIENT_ATTACHED and not IS_ANALYZING:
            if time.time() - LAST_HEARTBEAT > 45.0:
                SHUTDOWN_TRIGGERED = True
                os._exit(0)


class AppRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), 'application/json; charset=utf-8', code)

    def do_GET(self):
        global LAST_HEARTBEAT, CLIENT_ATTACHED
        if self.path in ('/', '/index.html'):
            CLIENT_ATTACHED = True
            LAST_HEARTBEAT = time.time()
            self._send(get_resource_bytes('index.html'), 'text/html; charset=utf-8')
            return
        if self.path == '/icon.svg':
            self._send(get_resource_bytes('icon.svg'), 'image/svg+xml')
            return
        if self.path == '/api/sample':
            self._send(get_resource_bytes('sample.png'), 'image/png')
            return
        if self.path == '/api/status':
            ensure_model_loaded()
            self._json({'status': 'online', 'classes': len(TDI_TABLE.classes),
                        'checkpoint_epoch': CLASSIFIER.epoch,
                        'checkpoint_val_acc': CLASSIFIER.best_acc})
            return
        self.send_error(404)

    def do_POST(self):
        global LAST_HEARTBEAT
        if self.path == '/api/heartbeat':
            LAST_HEARTBEAT = time.time()
            self._json({'status': 'ok'})
            return
        if self.path == '/api/shutdown':
            if self.client_address[0] not in ('127.0.0.1', '::1'):
                self.send_error(403, 'Local access only')
                return
            self._json({'status': 'terminated'})
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)), daemon=True).start()
            return
        if self.path != '/api/analyze':
            self.send_error(404)
            return

        global IS_ANALYZING
        IS_ANALYZING = True
        LAST_HEARTBEAT = time.time()
        try:
            ensure_model_loaded()
            content_len = int(self.headers.get('Content-Length', 0))
            if content_len == 0:
                self.send_error(400, 'Empty payload')
                return
            content_type = self.headers.get('Content-Type', '')
            body = self.rfile.read(content_len)
            raw_bytes = None
            if 'multipart/form-data' in content_type:
                header = b'Content-Type: ' + content_type.encode('latin-1', 'replace') + b'\r\n'
                msg = BytesParser(policy=default).parsebytes(
                    header + b'MIME-Version: 1.0\r\n\r\n' + body)
                for part in msg.iter_parts():
                    if part.get_param('name', header='content-disposition') == 'image':
                        raw_bytes = part.get_payload(decode=True)
                        break
            else:
                raw_bytes = body

            if not raw_bytes:
                self.send_error(400, 'No image received')
                return

            t0 = time.perf_counter()
            img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            orig_w, orig_h = img.size

            dets = count_cells(
                CLASSIFIER, img,
                window=WINDOW,
                stride_frac=0.5,
                conf_thr=0.48,
                nms_iou=0.30,
                cross_class_iou=0.70,
                merge=True,
                refine=False,
                max_windows=MAX_WINDOWS,
            )
            res = compute_tdi(dets, TDI_TABLE)
            dt = (time.perf_counter() - t0) * 1000

            xs, ys, step = sweep_plan(orig_w, orig_h, WINDOW, 0.5, MAX_WINDOWS)

            taxa_list = []
            for name, count in res.counts.items():
                taxon = TDI_TABLE.values[TDI_TABLE.classes.index(name)]
                taxa_list.append({
                    'name': name,
                    'count': count,
                    'percent': round(count / max(1, res.total_cells) * 100, 1),
                    'tdi_value': taxon.value,
                    'indicator_value': taxon.indicator,
                })
            taxa_list.sort(key=lambda t: -t['count'])

            det_boxes = []
            for d in dets:
                taxon = TDI_TABLE.values[d.cls_idx]
                det_boxes.append({
                    'box': [round(c, 1) for c in d.box],
                    'taxon': taxon.name,
                    'conf': round(d.conf, 3),
                    'tdi_value': taxon.value,
                })

            self._json({
                'success': True,
                'tdi': None if res.tdi is None else round(res.tdi, 3),
                'tdi100': None if res.tdi100 is None else round(res.tdi100, 1),
                'quality': res.quality or 'Undefined',
                'quality_desc': res.quality_es,
                'total_cells': res.total_cells,
                'valued_cells': res.valued_cells,
                'value_mass': round(res.value_mass * 100, 1),
                'mean_confidence': round(res.mean_confidence * 100, 1),
                'planktonic_fraction': round(res.planktonic_fraction * 100, 1),
                'latency_ms': round(dt, 0),
                'image_size': [orig_w, orig_h],
                'sweep': {
                    'windows': len(xs) * len(ys),
                    'step_px': step,
                    'window_px': WINDOW,
                    # The grid always spans edge to edge; `dense` is false when the
                    # window budget forced a stride wider than the one requested.
                    'covers_full_frame': xs[-1] + WINDOW >= orig_w and ys[-1] + WINDOW >= orig_h,
                    'dense_grid': step <= max(1, int(WINDOW * 0.5)) + 1,
                },
                'taxa': taxa_list,
                'detections': det_boxes,
            })
        except Exception as e:
            self._json({'success': False, 'error': str(e)}, 500)
        finally:
            IS_ANALYZING = False
            LAST_HEARTBEAT = time.time()


def main():
    ap = argparse.ArgumentParser(description='zar manah diatom tdi')
    ap.add_argument('--lan', action='store_true',
                    help='expose the UI on the local network (default: this machine only)')
    ap.add_argument('--port', type=int, default=None)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()

    signal.signal(signal.SIGINT, lambda sig, frame: _cleanup(0))
    signal.signal(signal.SIGTERM, lambda sig, frame: _cleanup(0))
    port = args.port if args.port else get_free_port(8765)
    host = '0.0.0.0' if args.lan else '127.0.0.1'
    lan_ip = get_lan_ip() if args.lan else '127.0.0.1'

    print('=' * 55)
    print('zar manah diatom tdi')
    print(f'En este equipo:    http://127.0.0.1:{port}')
    if args.lan and lan_ip != '127.0.0.1':
        print(f'En tu movil / red: http://{lan_ip}:{port}')
    print('=' * 55)

    threading.Thread(target=ensure_model_loaded, daemon=True).start()
    threading.Thread(target=watchdog_thread, daemon=True).start()

    server = ThreadingHTTPServer((host, port), AppRequestHandler)
    if not args.no_browser:
        webbrowser.open(f'http://127.0.0.1:{port}')

    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass


def _cleanup(code: int):
    shutil.rmtree(RESOURCE_DIR, ignore_errors=True)
    os._exit(code)


if __name__ == '__main__':
    main()
