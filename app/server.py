#!/usr/bin/env python3
"""zar manah diatom tdi"""
from __future__ import annotations

import argparse
import io
import json
import os
import signal
import socket
import sys
import threading
import time
import webbrowser
from email.parser import BytesParser
from email.policy import default
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Robust Path Resolution
APP_DIR = Path(__file__).resolve().parent
if (APP_DIR / "code" / "tdi.py").exists():
    ROOT_DIR = APP_DIR
elif (APP_DIR.parent / "code" / "tdi.py").exists():
    ROOT_DIR = APP_DIR.parent
else:
    ROOT_DIR = APP_DIR.parent

CODE_DIR = ROOT_DIR / "code"
sys.path.insert(0, str(CODE_DIR))

from PIL import Image
import torch
from tdi import DiatomClassifier, TdiTable, count_cells, compute_tdi, sweep_plan, CKPT, TABLE

# Global State
LAST_HEARTBEAT = time.time()
CLIENT_ATTACHED = False
SHUTDOWN_TRIGGERED = False
IS_ANALYZING = False
CLASSIFIER: DiatomClassifier | None = None
TDI_TABLE: TdiTable | None = None

def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

def get_free_port(start_port: int = 8765) -> int:
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port


def _sweep_info(width: int, height: int, window: int = 150,
                stride_frac: float = 0.5, max_windows: int = 1200) -> dict:
    """Honest description of the grid that was actually swept.

    `covers_full_frame` is true by construction (origins span edge to edge);
    `dense_grid` is false when the window budget forced a wider stride than the
    one requested, which the UI surfaces instead of pretending it was dense.
    """
    xs, ys, step = sweep_plan(width, height, window, stride_frac, max_windows)
    return {
        "windows": len(xs) * len(ys),
        "step_px": step,
        "window_px": window,
        "covers_full_frame": xs[-1] + window >= width and ys[-1] + window >= height,
        "dense_grid": step <= max(1, int(window * stride_frac)) + 1,
    }


def ensure_model_loaded():
    global CLASSIFIER, TDI_TABLE
    if TDI_TABLE is None:
        TDI_TABLE = TdiTable.load(TABLE)
    if CLASSIFIER is None:
        CLASSIFIER = DiatomClassifier(ckpt=CKPT)


def _multipart_image(content_type: str, body: bytes) -> bytes | None:
    """Pull the `image` part out of a multipart/form-data body.

    `cgi` was removed from the standard library in Python 3.13, so the email
    parser does the job instead: it is part of the stdlib and understands the
    same MIME framing.
    """
    header = b"Content-Type: " + content_type.encode("latin-1", "replace") + b"\r\n"
    msg = BytesParser(policy=default).parsebytes(
        header + b"MIME-Version: 1.0\r\n\r\n" + body
    )
    for part in msg.iter_parts():
        if part.get_param("name", header="content-disposition") == "image":
            payload = part.get_payload(decode=True)
            if payload:
                return payload
    return None

def watchdog_thread():
    """Shuts down server automatically if client closes the tab/window."""
    global LAST_HEARTBEAT, SHUTDOWN_TRIGGERED, IS_ANALYZING
    while not SHUTDOWN_TRIGGERED:
        time.sleep(2.0)
        # Only shutdown if client was previously connected and has been completely silent for > 45s
        if CLIENT_ATTACHED and not IS_ANALYZING:
            elapsed = time.time() - LAST_HEARTBEAT
            if elapsed > 45.0:
                SHUTDOWN_TRIGGERED = True
                os._exit(0)

class AppRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def log_message(self, format, *args):
        # Keep console output clean
        return

    def do_GET(self):
        global LAST_HEARTBEAT, CLIENT_ATTACHED
        if self.path in ("/", "/index.html"):
            CLIENT_ATTACHED = True
            LAST_HEARTBEAT = time.time()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            index_path = APP_DIR / "index.html"
            self.wfile.write(index_path.read_bytes())
            return
        elif self.path == "/icon.svg":
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.end_headers()
            self.wfile.write((APP_DIR / "icon.svg").read_bytes())
            return
        elif self.path == "/icon.png":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write((APP_DIR / "icon.png").read_bytes())
            return
        elif self.path == "/api/sample":
            samples = list((ROOT_DIR / "data" / "samples").glob("*.png"))
            if not samples:
                samples = list((ROOT_DIR / "data" / "UDE Diatoms in the Wild 2024" / "images").glob("*.png"))
            if samples:
                sample_file = samples[0]
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(sample_file.read_bytes())
                return
            self.send_error(404, "No sample image found")
            return
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "online",
                "classes": len(TDI_TABLE.classes) if TDI_TABLE else 101
            }).encode())
            return

        self.send_error(404)

    def do_POST(self):
        global LAST_HEARTBEAT, CLIENT_ATTACHED, SHUTDOWN_TRIGGERED
        
        if self.path == "/api/heartbeat":
            CLIENT_ATTACHED = True
            LAST_HEARTBEAT = time.time()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"alive"}')
            return

        elif self.path == "/api/shutdown":
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                self.send_error(403, "Local access only")
                return
            SHUTDOWN_TRIGGERED = True
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"terminated"}')
            def delayed_exit():
                time.sleep(1.2)
                # If heartbeat was received in the last 1.0s (e.g. user refreshed the page), cancel shutdown!
                if time.time() - LAST_HEARTBEAT < 1.0:
                    globals()["SHUTDOWN_TRIGGERED"] = False
                    return
                os._exit(0)
            threading.Thread(target=delayed_exit, daemon=True).start()
            return

        elif self.path == "/api/analyze":
            global IS_ANALYZING
            IS_ANALYZING = True
            LAST_HEARTBEAT = time.time()
            try:
                ensure_model_loaded()
                content_len = int(self.headers.get("Content-Length", 0))
                if content_len == 0:
                    self.send_error(400, "Empty payload")
                    return

                content_type = self.headers.get("Content-Type", "")
                body = self.rfile.read(content_len)
                raw_bytes = None

                if "multipart/form-data" in content_type:
                    raw_bytes = _multipart_image(content_type, body)
                else:
                    raw_bytes = body

                if not raw_bytes:
                    self.send_error(400, "No image received")
                    return

                t0 = time.perf_counter()
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                orig_w, orig_h = img.size

                dets = count_cells(
                    classifier=CLASSIFIER,
                    image=img,
                    window=150,
                    stride_frac=0.5,
                    conf_thr=0.48,
                    nms_iou=0.30,
                    cross_class_iou=0.70,
                    merge=True,
                    refine=False,
                    max_windows=1200,
                    scale=1.0,
                )
                res = compute_tdi(dets, TDI_TABLE)
                dt = (time.perf_counter() - t0) * 1000

                # Taxa breakdown
                taxa_list = []
                for name, count in res.counts.items():
                    val = TDI_TABLE.values[TDI_TABLE.classes.index(name)].value
                    taxa_list.append({
                        "name": name,
                        "count": count,
                        "percent": round(count / max(1, res.total_cells) * 100, 1),
                        "tdi_value": val,
                    })

                det_boxes = []
                for d in dets:
                    taxon_name = TDI_TABLE.classes[d.cls_idx]
                    tdi_v = TDI_TABLE.values[d.cls_idx].value
                    det_boxes.append({
                        "box": [round(c, 1) for c in d.box],
                        "taxon": taxon_name,
                        "conf": round(d.conf, 3),
                        "tdi_value": tdi_v,
                    })

                payload = {
                    "success": True,
                    "tdi": None if res.tdi is None else round(res.tdi, 3),
                    "tdi100": None if res.tdi100 is None else round(res.tdi100, 1),
                    "quality": res.quality or "Undefined",
                    "quality_desc": res.quality_es,
                    "total_cells": res.total_cells,
                    "valued_cells": res.valued_cells,
                    "value_mass": round(res.value_mass * 100, 1),
                    "mean_confidence": round(res.mean_confidence * 100, 1),
                    "planktonic_fraction": round(res.planktonic_fraction * 100, 1),
                    "latency_ms": round(dt, 0),
                    "image_size": [orig_w, orig_h],
                    "sweep": _sweep_info(orig_w, orig_h),
                    "taxa": taxa_list,
                    "detections": det_boxes,
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())
            finally:
                IS_ANALYZING = False
                LAST_HEARTBEAT = time.time()
            return

        self.send_error(404)

def open_app_window(url: str):
    """Fallback browser opener when running directly via CLI without native macOS wrapper."""
    webbrowser.open(url)

def main():
    parser = argparse.ArgumentParser(description="zar manah diatom tdi")
    parser.add_argument("--lan", action="store_true", help="Bind to 0.0.0.0")
    parser.add_argument("--host", type=str, default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser window")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))
    signal.signal(signal.SIGTERM, lambda sig, frame: os._exit(0))

    port = args.port if args.port else get_free_port(8765)
    host = args.host if args.host else ("0.0.0.0" if args.lan else "127.0.0.1")
    lan_ip = get_lan_ip() if host == "0.0.0.0" else "127.0.0.1"

    print("=" * 50)
    print("zar manah diatom tdi")
    print(f"En este ordenador: http://127.0.0.1:{port}")
    if host == "0.0.0.0" and lan_ip != "127.0.0.1":
        print(f"En tu movil / red:  http://{lan_ip}:{port}")
    print("=" * 50)

    # Pre-load model in background thread
    threading.Thread(target=ensure_model_loaded, daemon=True).start()

    # Start heartbeat watchdog for clean auto-exit
    threading.Thread(target=watchdog_thread, daemon=True).start()

    server = ThreadingHTTPServer((host, port), AppRequestHandler)
    if not args.no_browser:
        open_app_window(f"http://127.0.0.1:{port}")

    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass

if __name__ == "__main__":
    main()
