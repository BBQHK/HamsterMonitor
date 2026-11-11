"""Binary and path helpers shared by RTSP publishing."""

from __future__ import annotations

import os
import shutil

from config import hardware_dir


def resolve_bin(configured: str, name: str) -> str | None:
    """Find a binary from config, next to main.py, or PATH."""
    candidates = []
    if configured:
        candidates.append(configured)
    candidates.append(os.path.join(hardware_dir(), name))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return shutil.which(name)


def rtsp_path(camera_index: int) -> str:
    return f"camera{camera_index}"
