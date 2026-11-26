#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serve preprocessed policy-observation cameras over MJPEG.

The policy generator already resize/crops frames for inference. This server
re-encodes those arrays as JPEG so a browser or data_manager can see exactly
what the current policy uses. It does not subscribe to extra ROS cameras.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8004
DEFAULT_HZ = 10.0
DEFAULT_QUALITY = 70


def encode_observation_jpeg(image: np.ndarray, quality: int = DEFAULT_QUALITY) -> bytes:
    """Encode a policy observation image (RGB or gray) as JPEG."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("empty observation image")
    frame = image
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError(f"unsupported observation image shape {frame.shape}")
    if frame.dtype != np.uint8:
        peak = float(np.max(frame)) if frame.size else 1.0
        scale = 255.0 / peak if peak > 1.5 else 255.0
        frame = np.clip(frame.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))],
    )
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return encoded.tobytes()


def _placeholder_jpeg(message: str) -> bytes:
    canvas = np.full((180, 320, 3), 32, dtype=np.uint8)
    cv2.putText(
        canvas,
        message[:40],
        (16, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    if not ok:
        raise RuntimeError("placeholder JPEG encode failed")
    return encoded.tobytes()


def _viewer_html(policy_name: str, cameras: List[Dict[str, Any]]) -> bytes:
    cards = []
    for camera in cameras:
        name = camera["name"]
        label = camera.get("label") or name
        preprocess = camera.get("preprocess") or ""
        cards.append(
            f'<section class="card"><h2>{label}</h2>'
            f'<p class="meta">{preprocess}</p>'
            f'<img src="/stream/{name}" alt="{label}"/></section>'
        )
    body = "".join(cards) or "<p>이 정책에 카메라가 없습니다.</p>"
    html = f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8"/>
<title>{policy_name} inference cameras</title>
<style>
body {{ margin:0; font-family:sans-serif; background:#111; color:#eee; }}
header {{ padding:12px 16px; background:#1c1c1c; }}
h1 {{ font-size:16px; margin:0; }}
.grid {{ display:flex; flex-wrap:wrap; gap:12px; padding:12px; }}
.card {{ background:#1a1a1a; padding:8px; border-radius:8px; }}
.card img {{ display:block; max-width:480px; background:#000; }}
.meta {{ color:#9ca3af; font-size:12px; margin:4px 0 8px; }}
</style></head>
<body>
<header><h1>{policy_name} — 추론 카메라 (crop 적용)</h1></header>
<div class="grid">{body}</div>
</body></html>
"""
    return html.encode("utf-8")


class InferenceImageStreamServer:
    """Background HTTP server for the current policy's observation cameras."""

    def __init__(
        self,
        cameras: List[Dict[str, Any]],
        get_frame: Callable[[str], Optional[np.ndarray]],
        policy_name: str = "policy",
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        hz: float = DEFAULT_HZ,
        quality: int = DEFAULT_QUALITY,
    ):
        if not cameras:
            raise ValueError("stream_inference_images requires at least one camera")
        self.cameras = list(cameras)
        self._cameras_by_name = {str(item["name"]): item for item in self.cameras}
        self.get_frame = get_frame
        self.policy_name = policy_name
        self.host = host
        self.port = int(port)
        self.hz = max(1.0, float(hz))
        self.quality = int(np.clip(quality, 1, 100))
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, Any]] = {
            name: {
                "frame_count": 0,
                "bytes": 0,
                "last_size": 0,
                "last_ts": 0.0,
            }
            for name in self._cameras_by_name
        }
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._connections: set = set()
        self._connections_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urlparse(self.path)
                path = unquote(parsed.path)
                if path in ("/", "/viewer"):
                    body = _viewer_html(owner.policy_name, owner.cameras)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/health":
                    self._send_json(
                        {
                            "status": "ok",
                            "policy": owner.policy_name,
                            "cameras": [item["name"] for item in owner.cameras],
                        }
                    )
                    return
                if path == "/api/active-topics":
                    self._send_json(
                        {
                            "ok": True,
                            "policy": owner.policy_name,
                            "camera_topics": owner.cameras,
                        }
                    )
                    return
                if path == "/api/stats":
                    self._send_json({"ok": True, "cameras": owner.stats()})
                    return
                if path.startswith("/stream/"):
                    camera_id = path[len("/stream/") :].strip("/")
                    owner._stream_camera(self, camera_id)
                    return
                self._send_json({"ok": False, "error": "not found"}, status=404)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        # MJPEG clients keep the handler thread alive; do not wait on them at close.
        self._httpd.daemon_threads = True
        self._httpd.block_on_close = False
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name="inference_image_stream",
        )
        self._thread.start()

    def _register_connection(self, sock) -> None:
        if sock is None:
            return
        with self._connections_lock:
            self._connections.add(sock)

    def _unregister_connection(self, sock) -> None:
        if sock is None:
            return
        with self._connections_lock:
            self._connections.discard(sock)

    def _close_connections(self) -> None:
        with self._connections_lock:
            sockets = list(self._connections)
            self._connections.clear()
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def stop(self) -> None:
        with self._stop_lock:
            httpd = self._httpd
            thread = self._thread
            self._httpd = None
            self._thread = None
            self._stop_event.set()
            self._close_connections()
            if httpd is not None:
                try:
                    httpd.shutdown()
                except Exception:
                    pass
                try:
                    httpd.server_close()
                except Exception:
                    pass
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)

    def stats(self) -> Dict[str, Dict[str, Any]]:
        now = time.time()
        with self._lock:
            payload = {}
            for name, item in self._stats.items():
                elapsed = now - item["last_ts"] if item["last_ts"] else None
                payload[name] = {
                    "frame_count": item["frame_count"],
                    "hz": self.hz,
                    "avg_size_kb": (
                        (item["bytes"] / item["frame_count"]) / 1024.0
                        if item["frame_count"]
                        else 0.0
                    ),
                    "last_size_kb": item["last_size"] / 1024.0,
                    "silent_ms": None if elapsed is None else elapsed * 1000.0,
                }
            return payload

    def _record_frame(self, camera_id: str, payload: bytes) -> None:
        with self._lock:
            item = self._stats[camera_id]
            item["frame_count"] += 1
            item["bytes"] += len(payload)
            item["last_size"] = len(payload)
            item["last_ts"] = time.time()

    def _stream_camera(self, handler: BaseHTTPRequestHandler, camera_id: str) -> None:
        if camera_id not in self._cameras_by_name:
            handler.send_response(404)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(b'{"ok":false,"error":"unknown camera"}')
            return
        handler.connection.settimeout(1.0)
        self._register_connection(handler.connection)
        handler.send_response(200)
        handler.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        handler.send_header("Pragma", "no-cache")
        handler.end_headers()
        interval = 1.0 / self.hz
        try:
            self._write_stream_frames(handler, camera_id, interval)
        finally:
            self._unregister_connection(handler.connection)

    def _write_stream_frames(
        self, handler: BaseHTTPRequestHandler, camera_id: str, interval: float
    ) -> None:
        while not self._stop_event.is_set():
            started = time.time()
            image = self.get_frame(camera_id)
            try:
                if image is None:
                    payload = _placeholder_jpeg("waiting for policy frame")
                else:
                    payload = encode_observation_jpeg(image, self.quality)
                    self._record_frame(camera_id, payload)
            except Exception:
                payload = _placeholder_jpeg("encode failed")
            try:
                handler.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(payload)).encode("ascii")
                    + b"\r\n\r\n"
                    + payload
                    + b"\r\n"
                )
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            sleep_for = interval - (time.time() - started)
            if sleep_for > 0 and self._stop_event.wait(sleep_for):
                return
