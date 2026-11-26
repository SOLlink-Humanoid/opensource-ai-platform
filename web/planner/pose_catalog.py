"""Named joint poses and per-group error against the live robot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.action_layout import load_action_layout
from shared.joint_names import HEAD_JOINTS, PART_ERROR_KEYS, joint_group, joint_part_group

_PART_KEYS = ("waist", "body", "head", "left_arm", "right_arm", "left_hand", "right_hand")


def pose_json_files(path: Path) -> list[Path]:
    """JSON pose sources for a file or a directory of pose files.

    A missing single-file path falls back to every JSON file in that directory,
    which is how a former ``poses.json`` becomes ``full_body_poses/*.json``.
    """
    target = Path(path)
    if target.is_dir():
        return _substantive_json_files(target)
    if target.is_file():
        return [target]
    if target.suffix.lower() == ".json" and target.parent.is_dir():
        siblings = _substantive_json_files(target.parent)
        if siblings:
            return siblings
    return []


def _substantive_json_files(directory: Path) -> list[Path]:
    files = sorted(
        (
            child
            for child in directory.iterdir()
            if child.is_file() and child.suffix.lower() == ".json"
        ),
        key=lambda child: child.name,
    )
    substantive = [child for child in files if _json_has_entries(child)]
    return substantive or files


def _json_has_entries(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if isinstance(data, dict):
        return bool(data)
    if isinstance(data, list):
        return bool(data)
    return data is not None


class PoseCatalog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def source_files(self) -> list[Path]:
        return pose_json_files(self.path)

    def location(self) -> Path:
        if self.path.is_dir():
            return self.path
        files = self.source_files()
        if files and not self.path.is_file():
            return files[0].parent
        return self.path

    def _load_path(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _poses_from(self, data: dict[str, Any]) -> dict[str, Any]:
        nested = data.get("poses")
        if isinstance(nested, dict) and nested:
            return nested
        return {
            str(name): entry
            for name, entry in data.items()
            if isinstance(entry, dict)
            and (
                isinstance(entry.get("joints"), dict)
                or any(key in entry for key in _PART_KEYS)
            )
        }

    def _file_named(self, filename: str | None) -> Path | None:
        name = str(filename or "").strip()
        if not name:
            return None
        for path in self.source_files():
            if path.name == name:
                return path
        return None

    def list_file_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for path in self.source_files():
            poses = self._poses_from(self._load_path(path))
            rows: list[dict[str, str | None]] = []
            for name in sorted(str(key) for key in poses):
                entry = poses[name]
                pose_id = None
                if isinstance(entry, dict) and entry.get("id"):
                    pose_id = str(entry["id"])
                rows.append({"name": name, "id": pose_id, "file": path.name})
            groups.append({"file": path.name, "poses": rows})
        return groups

    def list_pose_names(self) -> list[str]:
        names: set[str] = set()
        for group in self.list_file_groups():
            for row in group["poses"]:
                names.add(str(row["name"]))
        return sorted(names)

    def list_pose_infos(self) -> list[dict[str, str | None]]:
        rows: list[dict[str, str | None]] = []
        for group in self.list_file_groups():
            rows.extend(group["poses"])
        return rows

    def list_pose_ids(self) -> set[str]:
        ids: set[str] = set()
        for row in self.list_pose_infos():
            pose_id = row.get("id")
            if pose_id:
                ids.add(str(pose_id))
        return ids

    def _part_joint_names(self) -> dict[str, list[str]]:
        """Pose-file part keys aligned with shared/model_action_layout.json."""
        names: dict[str, list[str]] = {"head": list(HEAD_JOINTS)}
        for channel in load_action_layout().channels:
            joints = [str(name) for name in channel["joints"]]
            channel_id = channel["id"]
            if channel_id == "WAIST":
                names["waist"] = joints
                names["body"] = joints
            elif channel_id == "LEFT_ARM":
                names["left_arm"] = joints
            elif channel_id == "RIGHT_ARM":
                names["right_arm"] = joints
            elif channel_id == "LEFT_GRIPPER":
                names["left_hand"] = joints
            elif channel_id == "RIGHT_GRIPPER":
                names["right_hand"] = joints
        return names

    def _joints_from_entry(self, entry: dict[str, Any]) -> dict[str, float] | None:
        if not isinstance(entry, dict):
            return None
        raw = entry.get("joints")
        if isinstance(raw, dict):
            out: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    out[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            return out or None
        part_names = self._part_joint_names()
        expanded: dict[str, float] = {}
        for key, joint_names in part_names.items():
            values = entry.get(key)
            if not isinstance(values, list):
                continue
            for name, value in zip(joint_names, values):
                try:
                    expanded[name] = float(value)
                except (TypeError, ValueError):
                    continue
        return expanded or None

    def _documents(self, file: str | None = None) -> list[dict[str, Any]]:
        if file:
            path = self._file_named(file)
            if path is None:
                return []
            return [self._poses_from(self._load_path(path))]
        return [
            self._poses_from(self._load_path(path)) for path in self.source_files()
        ]

    def joints_for(self, name: str, *, file: str | None = None) -> dict[str, float] | None:
        if not name:
            return None
        for poses in self._documents(file):
            entry = poses.get(name)
            if not isinstance(entry, dict):
                continue
            joints = self._joints_from_entry(entry)
            if joints:
                return joints
        return None

    def joints_for_id(self, pose_id: str, *, file: str | None = None) -> dict[str, float] | None:
        if not pose_id:
            return None
        wanted = str(pose_id)
        for poses in self._documents(file):
            for entry in poses.values():
                if isinstance(entry, dict) and str(entry.get("id") or "") == wanted:
                    joints = self._joints_from_entry(entry)
                    if joints:
                        return joints
        return None

    def joints_for_phase(self, phase: dict[str, Any]) -> dict[str, float] | None:
        pose_file = str(phase.get("pose_file") or "").strip() or None
        phase_id = str(phase.get("id") or "")
        if phase_id:
            by_id = self.joints_for_id(phase_id, file=pose_file)
            if by_id:
                return by_id
        pose_name = str(phase.get("pose_name") or "")
        if pose_name:
            return self.joints_for(pose_name, file=pose_file)
        return None

    def grouped_joint_errors(
        self,
        pose_name: str,
        current_joints: dict[str, float],
    ) -> dict[str, float] | None:
        target = self.joints_for(pose_name)
        if not target or not current_joints:
            return None
        totals = {"body": 0.0, "arms": 0.0, "head": 0.0}
        seen = {"body": False, "arms": False, "head": False}
        for name, goal in target.items():
            if name not in current_joints:
                continue
            group = joint_group(name)
            if group is None:
                continue
            totals[group] += abs(float(goal) - float(current_joints[name]))
            seen[group] = True
        if not any(seen.values()):
            return None
        return {group: totals[group] for group, ok in seen.items() if ok}

    def grouped_joint_errors_for_phase(
        self,
        phase: dict[str, Any],
        current_joints: dict[str, float],
    ) -> dict[str, float] | None:
        target = self.joints_for_phase(phase)
        if not target or not current_joints:
            return None
        totals = {"body": 0.0, "arms": 0.0, "head": 0.0}
        seen = {"body": False, "arms": False, "head": False}
        for name, goal in target.items():
            if name not in current_joints:
                continue
            group = joint_group(name)
            if group is None:
                continue
            totals[group] += abs(float(goal) - float(current_joints[name]))
            seen[group] = True
        if not any(seen.values()):
            return None
        return {group: totals[group] for group, ok in seen.items() if ok}

    def part_joint_errors_for_phase(
        self,
        phase: dict[str, Any],
        predicted_joints: dict[str, float],
        *,
        inferred_joint_names: set[str] | None = None,
    ) -> dict[str, float | None] | None:
        """Σ|catalog pose − predicted| per body part; None when model did not output that part."""
        target = self.joints_for_phase(phase)
        if not target or not predicted_joints:
            return None
        totals = {key: 0.0 for key in PART_ERROR_KEYS}
        counted = {key: False for key in PART_ERROR_KEYS}
        for name, goal in target.items():
            if inferred_joint_names is not None and name not in inferred_joint_names:
                continue
            if name not in predicted_joints:
                continue
            part = joint_part_group(name)
            if part is None:
                continue
            totals[part] += abs(float(goal) - float(predicted_joints[name]))
            counted[part] = True
        if not any(counted.values()):
            return None
        return {part: (totals[part] if counted[part] else None) for part in PART_ERROR_KEYS}
