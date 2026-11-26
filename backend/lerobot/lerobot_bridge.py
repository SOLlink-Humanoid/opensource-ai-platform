"""Read latest LeRobot action chunk from a shared JSON file (or mock)."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LerobotSnapshot:
    timestamp: float
    action_chunk: tuple[tuple[float, ...], ...]
    source: str


class LerobotBridge:
    """Poll ``LEROBOT_STATE_FILE``; in mock mode synthesize a moving chunk."""

    def __init__(
        self,
        *,
        state_file: Path | None,
        mock: bool,
        synthetic_mock: bool = True,
    ) -> None:
        self._state_file = state_file
        self._mock = mock
        self._synthetic_mock = synthetic_mock
        self._lock = threading.Lock()
        self._latest: LerobotSnapshot | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="lerobot-bridge",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def latest(self) -> LerobotSnapshot | None:
        with self._lock:
            return self._latest

    def _read_file(self) -> LerobotSnapshot | None:
        if self._state_file is None or not self._state_file.is_file():
            return None
        try:
            raw = self._state_file.read_text(encoding="utf-8")
            payload = json.loads(raw)
            ts = float(payload.get("timestamp", time.time()))
            chunk = payload.get("action_chunk") or payload.get("actions") or []
            rows: list[tuple[float, ...]] = []
            for row in chunk:
                if isinstance(row, (list, tuple)):
                    rows.append(tuple(float(v) for v in row))
            if not rows:
                return None
            return LerobotSnapshot(
                timestamp=ts,
                action_chunk=tuple(rows),
                source=str(self._state_file),
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            _log.debug("LeRobot state read failed: %s", exc)
            return None

    def _mock_snapshot(self) -> LerobotSnapshot:
        self._phase += 0.15
        horizon = 8
        dim = 6
        rows: list[tuple[float, ...]] = []
        for step in range(horizon):
            row = []
            for joint in range(dim):
                value = 0.25 * math.sin(self._phase + step * 0.2 + joint * 0.4)
                row.append(round(value, 4))
            rows.append(tuple(row))
        return LerobotSnapshot(
            timestamp=time.time(),
            action_chunk=tuple(rows),
            source="mock",
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = self._read_file()
            if snap is None and self._mock and self._synthetic_mock:
                snap = self._mock_snapshot()
            if snap is not None:
                with self._lock:
                    self._latest = snap
            self._stop.wait(0.1)
