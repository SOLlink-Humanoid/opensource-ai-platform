"""Phase profiles persisted as a JSON document (profile name -> phase list)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ACTIVE_PROFILE_KEY = "active_profile"
DEFAULT_PROFILE_NAME = "기본"
_MAX_PROFILE_TAGS = 5
_RESERVED_PROFILE_KEYS = frozenset({ACTIVE_PROFILE_KEY})

ProfileBucket = dict[str, Any]


class PhaseStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def ensure_file(self) -> None:
        if self.path.is_file():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_document(DEFAULT_PROFILE_NAME, {DEFAULT_PROFILE_NAME: self._empty_profile()})

    @staticmethod
    def _empty_profile() -> ProfileBucket:
        return {"phases": [], "tags": [], "hand_pose_file": ""}

    @staticmethod
    def _clean_hand_pose_file(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if (
            text != Path(text).name
            or not text.endswith(".json")
            or text.startswith(".")
            or "/" in text
            or "\\" in text
        ):
            return ""
        return text

    def _read_raw(self) -> dict[str, Any]:
        self.ensure_file()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _phase_number(item: Any, index: int) -> float:
        fallback = float(index + 1)
        if not isinstance(item, dict):
            return round(fallback, 1)
        raw = item.get("number")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return round(fallback, 1)
        try:
            return round(float(raw), 1)
        except (TypeError, ValueError):
            return round(fallback, 1)

    @staticmethod
    def _clean_phase_rows(phases: list[Any]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for index, item in enumerate(phases):
            if not isinstance(item, dict):
                continue
            row: dict[str, Any] = {
                "id": str(item.get("id") or f"phase-{index}"),
                "number": PhaseStore._phase_number(item, index),
                "description": str(item.get("description") or ""),
                "pose_name": str(item.get("pose_name") or ""),
            }
            pose_file = str(item.get("pose_file") or "").strip()
            if pose_file:
                row["pose_file"] = pose_file
            cleaned.append(row)
        return cleaned

    @staticmethod
    def _clean_tags(tags: Any) -> list[str]:
        if not isinstance(tags, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in tags:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
            if len(cleaned) >= _MAX_PROFILE_TAGS:
                break
        return cleaned

    @classmethod
    def _normalize_profile_bucket(cls, value: Any) -> ProfileBucket:
        if isinstance(value, list):
            return {"phases": cls._clean_phase_rows(value), "tags": [], "hand_pose_file": ""}
        if not isinstance(value, dict):
            return cls._empty_profile()
        phases = value.get("phases")
        if not isinstance(phases, list):
            return cls._empty_profile()
        return {
            "phases": cls._clean_phase_rows(phases),
            "tags": cls._clean_tags(value.get("tags")),
            "hand_pose_file": cls._clean_hand_pose_file(value.get("hand_pose_file")),
        }

    def _normalize_document(
        self, data: dict[str, Any]
    ) -> tuple[str, dict[str, ProfileBucket]]:
        profiles: dict[str, ProfileBucket] = {}

        legacy_phases = data.get("phases")
        if isinstance(legacy_phases, list):
            profiles[DEFAULT_PROFILE_NAME] = self._normalize_profile_bucket(
                {"phases": legacy_phases}
            )

        active = data.get(ACTIVE_PROFILE_KEY)
        active_name = (
            str(active).strip()
            if isinstance(active, str) and str(active).strip()
            else None
        )

        for key, value in data.items():
            if key in _RESERVED_PROFILE_KEYS or key == "phases":
                continue
            name = str(key).strip()
            if not name or name in _RESERVED_PROFILE_KEYS:
                continue
            profiles[name] = self._normalize_profile_bucket(value)

        if not profiles:
            profiles[DEFAULT_PROFILE_NAME] = self._empty_profile()

        if active_name and active_name in profiles:
            current_active = active_name
        else:
            current_active = next(iter(profiles))

        return current_active, profiles

    def _write_document(
        self,
        active_profile: str,
        profiles: dict[str, ProfileBucket],
    ) -> None:
        if active_profile not in profiles:
            raise ValueError(f"Unknown active profile: {active_profile}")

        payload: dict[str, Any] = {ACTIVE_PROFILE_KEY: active_profile}
        for name, bucket in profiles.items():
            if name in _RESERVED_PROFILE_KEYS:
                continue
            normalized = self._normalize_profile_bucket(bucket)
            phases_out: list[dict[str, Any]] = []
            for row in normalized["phases"]:
                phase_row: dict[str, Any] = {
                    "id": row["id"],
                    "number": row["number"],
                    "description": row["description"],
                    "pose_name": row["pose_name"],
                }
                pose_file = str(row.get("pose_file") or "").strip()
                if pose_file:
                    phase_row["pose_file"] = pose_file
                phases_out.append(phase_row)
            entry: dict[str, Any] = {"phases": phases_out}
            if normalized["tags"]:
                entry["tags"] = list(normalized["tags"])
            hand_pose_file = str(normalized.get("hand_pose_file") or "")
            if hand_pose_file:
                entry["hand_pose_file"] = hand_pose_file
            payload[name] = entry

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def active_profile(self) -> str:
        active, _ = self._normalize_document(self._read_raw())
        return active

    def profile_names(self) -> list[str]:
        _, profiles = self._normalize_document(self._read_raw())
        return list(profiles.keys())

    def all_profiles(self) -> dict[str, ProfileBucket]:
        _, profiles = self._normalize_document(self._read_raw())
        return {
            name: {
                "phases": list(bucket["phases"]),
                "tags": list(bucket["tags"]),
                "hand_pose_file": str(bucket.get("hand_pose_file") or ""),
            }
            for name, bucket in profiles.items()
        }

    def hand_pose_file_for(self, profile_name: str) -> str:
        name = str(profile_name or "").strip()
        active, profiles = self._normalize_document(self._read_raw())
        bucket = profiles.get(name) if name else None
        if bucket is None:
            bucket = profiles.get(active) or self._empty_profile()
        return str(bucket.get("hand_pose_file") or "")

    def all(self) -> list[dict[str, Any]]:
        active, profiles = self._normalize_document(self._read_raw())
        bucket = profiles.get(active) or self._empty_profile()
        return list(bucket["phases"])

    def phases_for(self, profile_name: str) -> list[dict[str, Any]]:
        name = str(profile_name or "").strip()
        if not name:
            return self.all()
        _, profiles = self._normalize_document(self._read_raw())
        bucket = profiles.get(name)
        if bucket is None:
            return []
        return list(bucket["phases"])

    def replace(self, phases: list[dict[str, Any]]) -> None:
        active, profiles = self._normalize_document(self._read_raw())
        bucket = profiles.get(active) or self._empty_profile()
        bucket["phases"] = self._clean_phase_rows(phases)
        profiles[active] = bucket
        self._write_document(active, profiles)

    def replace_document(
        self,
        *,
        active_profile: str,
        profiles: dict[str, ProfileBucket],
    ) -> None:
        name = str(active_profile or "").strip() or DEFAULT_PROFILE_NAME
        cleaned_profiles: dict[str, ProfileBucket] = {}
        for key, bucket in profiles.items():
            profile_name = str(key).strip()
            if not profile_name or profile_name in _RESERVED_PROFILE_KEYS:
                continue
            if profile_name in cleaned_profiles:
                raise ValueError(f"Duplicate profile name: {profile_name}")
            cleaned_profiles[profile_name] = self._normalize_profile_bucket(bucket)

        if not cleaned_profiles:
            cleaned_profiles[name] = self._empty_profile()
        if name not in cleaned_profiles:
            raise ValueError(f"Active profile must exist in profiles: {name}")

        self._write_document(name, cleaned_profiles)
