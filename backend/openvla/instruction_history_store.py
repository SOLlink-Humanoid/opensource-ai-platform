"""Persist OpenVLA task instructions under models/OpenVLA/instruction_history.json."""

from __future__ import annotations

import json
import logging
import secrets
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_FILENAME = "instruction_history.json"
_MAX_ENTRIES = 100


def _legacy_instruction_history_path(openvla_root: Path | None) -> Path:
    if openvla_root is not None:
        root = Path(openvla_root).expanduser()
    else:
        root = Path(__file__).resolve().parents[2] / "models" / "OpenVLA"
    return root / _FILENAME


def instruction_history_path(openvla_root: Path | None) -> Path:
    from ..platform.model_config_dir import (
        migrate_file_if_missing,
        model_config_dir,
        openvla_instruction_history_file,
    )

    dest = openvla_instruction_history_file(create_dir=True)
    base = model_config_dir()
    migrate_file_if_missing(
        dest,
        base / "openvla_instruction_history.json",
        _legacy_instruction_history_path(openvla_root),
    )
    return dest


def _empty() -> dict[str, Any]:
    return {"last_instruction": "", "entries": []}


def _normalize(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return _empty()
    raw_entries = data.get("entries")
    entries: list[dict[str, Any]] = []
    if isinstance(raw_entries, list):
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            entry_id = str(item.get("id") or "").strip() or secrets.token_hex(6)
            try:
                used = float(item.get("last_used_at") or 0.0)
            except (TypeError, ValueError):
                used = 0.0
            entries.append(
                {"id": entry_id, "text": text, "last_used_at": used}
            )
    entries.sort(key=lambda row: float(row["last_used_at"]), reverse=True)
    last = str(data.get("last_instruction") or "").strip()
    if not last and entries:
        last = str(entries[0]["text"])
    return {"last_instruction": last, "entries": entries[:_MAX_ENTRIES]}


def load_history(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("Could not read %s: %s", path, exc)
        return _empty()
    return _normalize(raw)


def _write_history(path: Path, data: dict[str, Any]) -> None:
    payload = _normalize(data)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def record_instruction(path: Path, text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        return load_history(path)
    data = load_history(path)
    now = time.time()
    entries: list[dict[str, Any]] = []
    matched: dict[str, Any] | None = None
    for item in data["entries"]:
        if str(item["text"]).strip().lower() == cleaned.lower():
            matched = {
                "id": item["id"],
                "text": cleaned,
                "last_used_at": now,
            }
            continue
        entries.append(item)
    if matched is None:
        matched = {
            "id": secrets.token_hex(6),
            "text": cleaned,
            "last_used_at": now,
        }
    entries.insert(0, matched)
    data["entries"] = entries[:_MAX_ENTRIES]
    data["last_instruction"] = cleaned
    _write_history(path, data)
    return data


def delete_instruction(path: Path, entry_id: str) -> dict[str, Any]:
    data = load_history(path)
    needle = entry_id.strip()
    data["entries"] = [e for e in data["entries"] if str(e.get("id")) != needle]
    last = str(data.get("last_instruction") or "").strip()
    if last and not any(str(e["text"]) == last for e in data["entries"]):
        data["last_instruction"] = (
            str(data["entries"][0]["text"]) if data["entries"] else ""
        )
    _write_history(path, data)
    return data
