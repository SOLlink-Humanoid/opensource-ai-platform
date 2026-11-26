"""Background OpenVLA weight load at platform startup."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from web.openvla.openvla_engine import OpenVLAEngine

_log = logging.getLogger(__name__)


def start_openvla_load(
    *,
    engine: OpenVLAEngine,
    settings,
    state: dict[str, Any],
    resolve_version: Callable[..., str],
    deps_error: Callable[[], str | None],
    inference_lock: threading.Lock,
    warmup: bool,
    unnorm_key: str,
) -> None:
    """Load OpenVLA on a daemon thread so HTTP and other tabs stay usable."""

    if settings.mock:
        state["openvla_load_status"] = "ready"
        state["openvla_load_error"] = None
        return

    if not getattr(settings, "openvla_enabled", True):
        state["openvla_load_status"] = "disabled"
        state["openvla_load_error"] = None
        _log.info("OpenVLA load skipped (OPENVLA_ENABLED=0)")
        return

    err = deps_error()
    if err:
        state["openvla_load_status"] = "error"
        state["openvla_load_error"] = err.splitlines()[0]
        return

    state["openvla_load_status"] = "loading"
    state["openvla_load_error"] = None

    def _run() -> None:
        try:
            version = resolve_version(settings)
            with inference_lock:
                engine.load(version)
                if warmup:
                    engine.warmup(version, unnorm_key=unnorm_key)
            state["openvla_load_status"] = "ready"
            state["openvla_load_error"] = None
            _log.info("OpenVLA %s ready on %s", version, engine.device)
        except Exception as exc:  # noqa: BLE001
            _log.exception("OpenVLA startup load failed")
            state["openvla_load_status"] = "error"
            state["openvla_load_error"] = str(exc)
            engine.last_error = str(exc)

    threading.Thread(target=_run, name="openvla-load", daemon=True).start()
