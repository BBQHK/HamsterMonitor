"""Load and validate hardware settings from config.conf."""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from typing import List


def hardware_dir() -> str:
    """Directory that contains config.conf, mediamtx, and main.py."""
    return os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Config:
    camera_indices: List[int]
    detection_index: int
    frame_width: int
    frame_height: int
    fps: int
    frame_skip: int
    api_url: str
    detection_result_url: str
    rtsp_enabled: bool
    rtsp_port: int
    rtsp_bitrate_kbps: int
    ffmpeg_bin: str
    mediamtx_bin: str

    @property
    def frame_interval(self) -> float:
        return 1.0 / self.fps


def load_config(config_path: str | None = None) -> Config:
    """Load camera and API settings from config.conf.

    Override the path with the HAMSTER_CONFIG environment variable, or pass
    config_path explicitly.
    """
    if config_path is None:
        default_path = os.path.join(hardware_dir(), "config.conf")
        config_path = os.environ.get("HAMSTER_CONFIG", default_path)

    parser = configparser.ConfigParser()
    read = parser.read(config_path)
    if not read:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    indices = [
        int(part.strip())
        for part in parser.get("camera", "indices").split(",")
        if part.strip()
    ]
    if not indices:
        raise ValueError(f"camera.indices is empty in {config_path}")

    detection_index = parser.getint("camera", "detection_index")
    if detection_index not in indices:
        raise ValueError(
            f"camera.detection_index {detection_index} is not in camera.indices {indices}"
        )

    api_url = parser.get("api", "url").rstrip("/")
    rtsp_enabled = True
    rtsp_port = 8554
    rtsp_bitrate_kbps = 800
    ffmpeg_bin = ""
    mediamtx_bin = ""
    if parser.has_section("rtsp"):
        if parser.has_option("rtsp", "enabled"):
            rtsp_enabled = parser.getboolean("rtsp", "enabled")
        if parser.has_option("rtsp", "port"):
            rtsp_port = parser.getint("rtsp", "port")
        if parser.has_option("rtsp", "bitrate_kbps"):
            rtsp_bitrate_kbps = parser.getint("rtsp", "bitrate_kbps")
        if parser.has_option("rtsp", "ffmpeg_bin"):
            ffmpeg_bin = parser.get("rtsp", "ffmpeg_bin").strip()
        if parser.has_option("rtsp", "mediamtx_bin"):
            mediamtx_bin = parser.get("rtsp", "mediamtx_bin").strip()
    if rtsp_port < 1 or rtsp_port > 65535:
        raise ValueError(f"rtsp.port is out of range in {config_path}")
    if rtsp_bitrate_kbps < 1:
        raise ValueError(f"rtsp.bitrate_kbps must be positive in {config_path}")

    return Config(
        camera_indices=indices,
        detection_index=detection_index,
        frame_width=parser.getint("camera", "frame_width"),
        frame_height=parser.getint("camera", "frame_height"),
        fps=parser.getint("camera", "fps"),
        frame_skip=parser.getint("camera", "frame_skip"),
        api_url=api_url,
        detection_result_url=f"{api_url}/detection_result",
        rtsp_enabled=rtsp_enabled,
        rtsp_port=rtsp_port,
        rtsp_bitrate_kbps=rtsp_bitrate_kbps,
        ffmpeg_bin=ffmpeg_bin,
        mediamtx_bin=mediamtx_bin,
    )
