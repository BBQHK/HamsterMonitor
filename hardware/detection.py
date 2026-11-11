"""Poll activity results from the activity monitor server."""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from config import Config
    from state import RuntimeState


class DetectionPoller:
    def __init__(self, config: "Config", state: "RuntimeState") -> None:
        self.config = config
        self.state = state
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="detection-poller",
            daemon=True,
        )
        self._thread.start()

    def _poll_loop(self) -> None:
        while True:
            try:
                response = requests.get(self.config.detection_result_url, timeout=2)
                if response.status_code == 200:
                    result = response.json()
                    self.state.last_activity_result.update(
                        {
                            "activity": result.get("activity", "Unknown"),
                            "activity_probability": result.get(
                                "activity_probability", 0.0
                            ),
                        }
                    )
                    self.state.api_error_count = 0
                else:
                    self.state.api_error_count += 1
                    print(f"Failed to get detection result: {response.status_code}")
            except (requests.RequestException, json.JSONDecodeError) as exc:
                self.state.api_error_count += 1
                print(f"Detection result polling error: {exc}")
            except Exception as exc:
                print(f"Error in detection result polling: {exc}")

            time.sleep(0.5)
