"""Persist Task Planner OpenVLA+YOLO → LeRobot policy mapping rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TaskPolicyRulesStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def ensure_file(self) -> None:
        if self.path.is_file():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write([])

    def read(self) -> list[dict[str, Any]]:
        self.ensure_file()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict) and str(item.get("id") or "").strip():
                out.append(item)
        return out

    def replace(self, rules: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write(rules)

    def _write(self, rules: list[dict[str, Any]]) -> None:
        self.path.write_text(
            json.dumps(rules, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )