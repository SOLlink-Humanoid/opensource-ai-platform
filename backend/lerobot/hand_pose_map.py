"""Per-policy map from LeRobot hand keys (0.1, 0.2, …) to joint vectors."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_PLATFORM_ROOT = Path(__file__).resolve().parents[2]
_STORE_NAME = "lerobot_hand_mappings.json"
_HAND_JOINTS = 10


def platform_root() -> Path:
    raw = (
        os.environ.get("OAP_PLATFORM_ROOT")
        or os.environ.get("OAP_VLA_ROOT")
        or ""
    ).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _PLATFORM_ROOT


def hand_poses_dir() -> Path:
    """Shared hand pose JSON library (bridge / ``poses/hand_poses``, not platform ``config/``)."""
    raw = os.environ.get("OAP_HAND_POSES_DIR", "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (platform_root() / path).resolve()
        return path
    return (platform_root().parent / "poses" / "hand_poses").resolve()


def mappings_path() -> Path:
    from ..platform.model_config_dir import lerobot_hand_mappings_file

    return lerobot_hand_mappings_file(create_dir=True)


def _policy_token(policy: str) -> str:
    token = str(policy or "").strip()
    if not token or token in {".", ".."} or "/" in token or "\\" in token:
        raise ValueError("잘못된 정책 이름입니다")
    return token


def _file_token(name: str) -> str:
    token = str(name or "").strip()
    if not token:
        return ""
    if token != Path(token).name or not token.endswith(".json") or token.startswith("."):
        raise ValueError("핸드 자세 파일 이름이 올바르지 않습니다")
    if not (hand_poses_dir() / token).is_file():
        raise ValueError(f"핸드 자세 파일이 없습니다: {token}")
    return token


def _side_rows(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        joints = item.get("joints")
        if not isinstance(joints, list) or len(joints) != _HAND_JOINTS:
            continue
        try:
            key = round(float(item["key"]), 4)
            values = [float(value) for value in joints]
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(
            {
                "key": key,
                "description": str(item.get("description") or "").strip(),
                "joints": values,
            }
        )
    rows.sort(key=lambda row: float(row["key"]))
    return rows


def load_pose_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"핸드 자세 파일을 읽지 못했습니다: {path.name} ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"핸드 자세 파일 형식이 올바르지 않습니다: {path.name}")
    left = _side_rows(data.get("left"))
    right = _side_rows(data.get("right"))
    if not left and not right:
        raise ValueError(f"핸드 자세 파일에 키 값이 없습니다: {path.name}")
    return {
        "file": path.name,
        "left": left,
        "right": right,
    }


def list_pose_files() -> list[dict[str, Any]]:
    folder = hand_poses_dir()
    if not folder.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")):
        if not path.is_file():
            continue
        try:
            table = load_pose_file(path)
        except ValueError:
            continue
        items.append(
            {
                "id": path.name,
                "name": path.stem,
                "left": [
                    {"key": row["key"], "description": row["description"]}
                    for row in table["left"]
                ],
                "right": [
                    {"key": row["key"], "description": row["description"]}
                    for row in table["right"]
                ],
            }
        )
    return items


def _read_store() -> dict[str, str]:
    path = mappings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get("selections") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {}
    selections: dict[str, str] = {}
    known = {item["id"] for item in list_pose_files()}
    for key, value in raw.items():
        try:
            policy = _policy_token(str(key))
        except ValueError:
            continue
        file_name = str(value or "").strip()
        if file_name and file_name in known:
            selections[policy] = file_name
    return selections


def selections() -> dict[str, str]:
    return _read_store()


def set_selection(policy: str, file_name: str) -> dict[str, str]:
    token = _policy_token(policy)
    chosen = _file_token(file_name)
    current = _read_store()
    if chosen:
        current[token] = chosen
    else:
        current.pop(token, None)
    path = mappings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"selections": current}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return current


def normalize_pose_filename(name: str) -> str:
    """Empty clears the selection. A name must be a JSON file in hand_poses."""
    token = str(name or "").strip()
    if not token:
        return ""
    return _file_token(token)


def table_for_file(file_name: str) -> dict[str, Any] | None:
    token = str(file_name or "").strip()
    if not token:
        return None
    path = hand_poses_dir() / token
    if not path.is_file():
        return None
    return load_pose_file(path)


def table_for_policy(policy: str) -> dict[str, Any] | None:
    token = _policy_token(policy)
    file_name = _read_store().get(token)
    if not file_name:
        return None
    return load_pose_file(hand_poses_dir() / file_name)
