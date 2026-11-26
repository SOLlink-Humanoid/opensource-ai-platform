"""Serialize Metal (MPS) work — OpenVLA + YOLO must not encode concurrently on Mac."""

from __future__ import annotations

import threading

MPS_INFERENCE_LOCK = threading.Lock()
