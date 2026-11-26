"""Append-only LeRobot policy action / waist-yaw trace for OAP (lerobot_action.log)."""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

_OAP_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEROBOT_ACTION_LOG = _OAP_ROOT / "lerobot_action.log"
WAIST_YAW_INDEX_IN_BODY = 4
FLAT_EPS_RAD = 0.0005


def _direction(delta: float) -> str:
    if delta > FLAT_EPS_RAD:
        return "increase"
    if delta < -FLAT_EPS_RAD:
        return "decrease"
    return "flat"


def _fmt_q5(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.5f}"


def _ts() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class LerobotActionLog:
    """Thread-safe append-only logger (sessions survive policy stop)."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        path: Path | str | None = None,
        publish_sample_every: int = 10,
    ) -> None:
        self.enabled = bool(enabled)
        self.path = Path(path) if path else DEFAULT_LEROBOT_ACTION_LOG
        self.publish_sample_every = max(1, int(publish_sample_every))
        self._lock = threading.Lock()
        self._prev_cmd_idx05: Optional[float] = None
        self._prev_publish_dir: Optional[str] = None
        self._prev_chunk_dir: Optional[str] = None
        self._chunk_dir_flips = 0
        self._publish_dir_flips = 0
        self._session_label = ""

    def begin_session(self, label: str = "") -> None:
        if not self.enabled:
            return
        with self._lock:
            self._session_label = label.strip()
            self._prev_cmd_idx05 = None
            self._prev_publish_dir = None
            self._prev_chunk_dir = None
            self._chunk_dir_flips = 0
            self._publish_dir_flips = 0
        self._append(
            f"{_ts()} SESSION_START policy={self._session_label or 'unknown'}"
        )

    def end_session(self, reason: str = "stopped") -> None:
        if not self.enabled:
            return
        with self._lock:
            flips = (
                f"chunk_dir_flips={self._chunk_dir_flips} "
                f"publish_dir_flips={self._publish_dir_flips}"
            )
        self._append(f"{_ts()} SESSION_END reason={reason} {flips}")

    def _append(self, line: str) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line.rstrip("\n") + "\n")
                    handle.flush()
        except OSError:
            pass

    def note_chunk_splice(
        self,
        *,
        inference_count: int,
        reference: str,
        shift_norm_rad: float,
        on_append: float,
        first_q5_before_rad: float | None = None,
        first_q5_after_rad: float | None = None,
        ref_q5_rad: float | None = None,
        grip_left_before: float | None = None,
        grip_left_after: float | None = None,
        grip_left_ref: float | None = None,
        grip_right_before: float | None = None,
        grip_right_after: float | None = None,
        grip_right_ref: float | None = None,
        **_: Any,
    ) -> None:
        if not self.enabled:
            return
        q5 = ""
        if first_q5_before_rad is not None and first_q5_after_rad is not None:
            q5 = (
                f" q5={_fmt_q5(first_q5_before_rad)}->{_fmt_q5(first_q5_after_rad)}"
                f" ref_q5={_fmt_q5(ref_q5_rad)}"
            )
        extras: list[str] = []
        for side, before, after, ref in (
            ("left", grip_left_before, grip_left_after, grip_left_ref),
            ("right", grip_right_before, grip_right_after, grip_right_ref),
        ):
            if before is None and after is None:
                continue
            piece = f"{side}_g={_fmt_q5(before)}->{_fmt_q5(after)}"
            if ref is not None:
                piece += f"(ref={_fmt_q5(ref)})"
            extras.append(piece)
        grip = f" {' '.join(extras)}" if extras else ""
        self._append(
            f"{_ts()} INFO CHUNK_SPLICE n={inference_count} ref={reference} "
            f"on_append={int(on_append)} shift_norm={shift_norm_rad:.5f}{q5}{grip}"
        )

    def note_queue_splice(
        self,
        *,
        inference_count: int,
        queue_cleared: bool,
        deferred: int,
        queue_before: int,
        reverse_trimmed: float,
        reverse_remain: float,
        first_proj_rad: float | None = None,
        **_: Any,
    ) -> None:
        if not self.enabled:
            return
        proj = (
            f" first_proj={float(first_proj_rad):.5f}"
            if first_proj_rad is not None
            else ""
        )
        self._append(
            f"{_ts()} INFO QUEUE_SPLICE n={inference_count} "
            f"cleared={int(queue_cleared)} deferred={deferred} "
            f"queue_before={queue_before} "
            f"reverse_trimmed={int(reverse_trimmed)} "
            f"remain={int(reverse_remain)}{proj}"
        )

    def note_waist_prefix_fix(
        self,
        *,
        inference_count: int,
        ref_rad: float,
        shift_rad: float,
        first_before_rad: float,
        first_after_rad: float,
    ) -> None:
        if not self.enabled:
            return
        self._append(
            f"{_ts()} INFO WAIST_PREFIX_FIX n={inference_count} "
            f"ref_q5={_fmt_q5(ref_rad)} "
            f"first_q5={_fmt_q5(first_before_rad)}->{_fmt_q5(first_after_rad)} "
            f"shift={shift_rad:.5f}"
        )

    def note_inference(
        self,
        *,
        inference_count: int,
        obs_idx05: Optional[float],
        live_idx05: Optional[float],
        n_consumed: int,
        remaining_actions_before: int,
        queue_cleared: bool,
        stale_skipped: int,
        actions_list: List[Any],
        inference_mode: str,
        inference_ms: float,
        action_vector_index: int = WAIST_YAW_INDEX_IN_BODY,
    ) -> bool:
        """Log one inference cycle. Returns True if chunk waist direction flipped."""
        if not self.enabled or not actions_list:
            return False
        idx = int(action_vector_index)
        try:
            first = float(actions_list[0][idx])
            last = float(actions_list[-1][idx])
        except (TypeError, ValueError, IndexError):
            return False
        span = last - first
        chunk_dir = _direction(span)
        chunk_flipped = False
        with self._lock:
            if (
                self._prev_chunk_dir is not None
                and chunk_dir != "flat"
                and self._prev_chunk_dir != "flat"
                and chunk_dir != self._prev_chunk_dir
            ):
                chunk_flipped = True
                self._chunk_dir_flips += 1
            if chunk_dir != "flat":
                self._prev_chunk_dir = chunk_dir

        level = "WARN" if chunk_flipped else "INFO"
        parts = [
            _ts(),
            level,
            "INFERENCE",
            f"n={inference_count}",
            f"mode={inference_mode}",
            f"obs_q5={_fmt_q5(obs_idx05)}",
            f"live_q5={_fmt_q5(live_idx05)}",
            f"chunk_q5={_fmt_q5(first)}->{_fmt_q5(last)}",
            f"chunk_dir={chunk_dir}",
            f"chunk_span={span:.5f}",
            f"queue_cleared={int(queue_cleared)}",
            f"queue_before={remaining_actions_before}",
            f"stale_skipped={stale_skipped}",
            f"consumed={n_consumed}",
            f"chunk_len={len(actions_list)}",
            f"infer_ms={inference_ms:.1f}",
        ]
        if chunk_flipped:
            parts.append("chunk_dir_flipped=1")
        self._append(" ".join(parts))
        return chunk_flipped

    def note_publish(
        self,
        *,
        tick: int,
        cmd_idx05: float,
        live_idx05: Optional[float],
        queue_remaining: int,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            delta = (
                cmd_idx05 - self._prev_cmd_idx05
                if self._prev_cmd_idx05 is not None
                else 0.0
            )
            direction = (
                _direction(delta) if self._prev_cmd_idx05 is not None else "flat"
            )
            flipped = False
            if (
                self._prev_publish_dir is not None
                and direction != "flat"
                and self._prev_publish_dir != "flat"
                and direction != self._prev_publish_dir
            ):
                flipped = True
                self._publish_dir_flips += 1
            if direction != "flat":
                self._prev_publish_dir = direction
            self._prev_cmd_idx05 = cmd_idx05

        sample = tick % self.publish_sample_every == 0
        if not flipped and not sample:
            return

        level = "WARN" if flipped else "INFO"
        tag = "PUBLISH_FLIP" if flipped else "PUBLISH"
        line = (
            f"{_ts()} {level} {tag} tick={tick} "
            f"cmd_q5={_fmt_q5(cmd_idx05)} live_q5={_fmt_q5(live_idx05)} "
            f"delta_cmd={delta:.5f} dir={direction} queue={queue_remaining}"
        )
        if flipped:
            line += " publish_dir_flipped=1"
        self._append(line)