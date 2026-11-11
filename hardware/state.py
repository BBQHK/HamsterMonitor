"""Shared mutable runtime state for cameras, sensors, and activity."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional


class RuntimeState:
    """Holds live camera frames, publishers, and latest sensor/activity values."""

    def __init__(self) -> None:
        self.cameras: Dict[int, Any] = {}
        self.latest_frames: Dict[int, Optional[bytes]] = {}
        self.frame_locks: Dict[int, threading.Lock] = {}
        self.capture_threads: Dict[int, threading.Thread] = {}
        self.capture_stop = threading.Event()

        self.h264_publishers: Dict[int, Any] = {}

        self.last_activity_result = {
            "activity": "Unknown",
            "activity_probability": 0.0,
        }
        self.last_sensor_readings = {
            "temperature": 0.0,
            "humidity": 0.0,
            "air_quality": "Unknown",
            "air_quality_ppm": 0.0,
            "last_read_time": 0.0,
        }
        self.api_error_count = 0
        self.api_error_threshold = 3
