"""MediaMTX process management and per-camera H.264 publishers."""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from overlay import annotate_frame
from paths import resolve_bin, rtsp_path

if TYPE_CHECKING:
    from config import Config
    from sensors import Sensors
    from state import RuntimeState


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_port(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.1)
    return False


def _log_process_output(proc: subprocess.Popen, prefix: str) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(f"{prefix}{line.rstrip()}")


class H264Publisher:
    """Publish one camera to MediaMTX as baseline H.264 over RTSP.

    Frames are written on a private pipe so ffmpeg's stdin is not the video.
    The capture thread only stores the newest picture. This thread emits one
    frame per tick at the configured FPS and draws the clock at send time, so
    the on-screen second matches the stream timeline.
    """

    def __init__(
        self,
        camera_index: int,
        config: "Config",
        state: "RuntimeState",
        sensors: "Sensors",
        ffmpeg_bin_cache: Dict[str, Optional[str]],
    ) -> None:
        self.camera_index = camera_index
        self.config = config
        self.state = state
        self.sensors = sensors
        self._ffmpeg_bin_cache = ffmpeg_bin_cache
        self.latest = None
        self.cond = threading.Condition()
        self.stop_event = threading.Event()
        self.proc = None
        self.write_fd = None
        self.proc_lock = threading.Lock()
        self._warned = False
        self.thread = threading.Thread(
            target=self._run,
            name=f"h264-{camera_index}",
            daemon=True,
        )

    def submit(self, frame) -> None:
        if self.stop_event.is_set():
            return
        with self.cond:
            self.latest = frame
            self.cond.notify()

    def stop(self) -> None:
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()
        self._stop_ffmpeg()
        self.thread.join(timeout=3)

    def _ffmpeg_bin(self) -> str:
        if "bin" not in self._ffmpeg_bin_cache:
            self._ffmpeg_bin_cache["bin"] = (
                resolve_bin(self.config.ffmpeg_bin, "ffmpeg") or ""
            )
        return self._ffmpeg_bin_cache["bin"] or ""

    def _open_ffmpeg(self):
        binary = self._ffmpeg_bin()
        if not binary:
            if not self._warned:
                print("ffmpeg not found. Install it with: sudo apt install ffmpeg")
                self._warned = True
            return None

        read_fd, write_fd = os.pipe()
        gop = str(max(1, self.config.fps))
        url = f"rtsp://127.0.0.1:{self.config.rtsp_port}/{rtsp_path(self.camera_index)}"
        bitrate = self.config.rtsp_bitrate_kbps
        command = [
            binary, "-hide_banner", "-loglevel", "warning",
            "-thread_queue_size", "64",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-video_size", f"{self.config.frame_width}x{self.config.frame_height}",
            "-framerate", str(self.config.fps),
            "-i", f"pipe:{read_fd}",
            # Surveillance Station resets video-only streams. VLC does not.
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=mono:sample_rate=8000",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-level:v", "3.1",
            "-pix_fmt", "yuv420p",
            "-g", gop,
            "-keyint_min", gop,
            "-bf", "0",
            "-b:v", f"{bitrate}k",
            "-maxrate", f"{bitrate}k",
            "-bufsize", f"{bitrate}k",
            "-x264-params", "repeat-headers=1:scenecut=0:force-cfr=1",
            "-c:a", "pcm_mulaw",
            "-ar", "8000",
            "-ac", "1",
            "-muxdelay", "0",
            "-muxpreload", "0",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            "-pkt_size", "1200",
            url,
        ]
        opened = False
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                pass_fds=(read_fd,),
            )
            opened = True
        except OSError as exc:
            print(f"Failed to start ffmpeg for camera {self.camera_index}: {exc}")
            return None
        finally:
            os.close(read_fd)
            if not opened:
                os.close(write_fd)
        if not opened:
            return None

        with self.proc_lock:
            self.proc = proc
            self.write_fd = write_fd
        print(
            f"Publishing H.264 rtsp://0.0.0.0:{self.config.rtsp_port}/"
            f"{rtsp_path(self.camera_index)}"
        )
        return proc

    def _stop_ffmpeg(self) -> None:
        with self.proc_lock:
            proc = self.proc
            write_fd = self.write_fd
            self.proc = None
            self.write_fd = None
        if write_fd is not None:
            try:
                os.close(write_fd)
            except OSError:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _write_all(self, data: bytes) -> None:
        with self.proc_lock:
            write_fd = self.write_fd
        if write_fd is None:
            raise BrokenPipeError
        view = memoryview(data)
        while view:
            written = os.write(write_fd, view)
            view = view[written:]

    def _write_frames(self, proc) -> None:
        """Write one frame every tick and never burst to catch up."""
        frame_interval = self.config.frame_interval
        while not self.stop_event.is_set():
            tick_start = time.monotonic()
            with self.cond:
                frame = self.latest
            if frame is None:
                if proc.poll() is not None:
                    return
                if self.stop_event.wait(timeout=frame_interval):
                    return
                continue

            frame = frame.copy()
            if self.camera_index != self.config.detection_index:
                annotate_frame(frame, self.config, self.state, self.sensors)
            try:
                self._write_all(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError, ValueError):
                return
            if proc.poll() is not None:
                return

            remain = frame_interval - (time.monotonic() - tick_start)
            if remain > 0 and self.stop_event.wait(timeout=remain):
                return

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                proc = self._open_ffmpeg()
            except Exception as exc:
                print(
                    f"H.264 publisher for camera {self.camera_index} failed to start: {exc}"
                )
                self.stop_event.wait(2)
                continue
            if proc is None:
                self.stop_event.wait(2)
                continue
            self._write_frames(proc)
            code = proc.poll()
            self._stop_ffmpeg()
            if self.stop_event.is_set():
                return
            print(
                f"H.264 publisher for camera {self.camera_index} exited ({code}); restarting"
            )
            self.stop_event.wait(1)


class RtspService:
    """Starts MediaMTX (if needed) and one H.264 publisher per camera."""

    def __init__(self, config: "Config", state: "RuntimeState", sensors: "Sensors") -> None:
        self.config = config
        self.state = state
        self.sensors = sensors
        self.mediamtx_proc: subprocess.Popen | None = None
        self.mediamtx_config_path: str | None = None
        self._ffmpeg_bin_cache: Dict[str, Optional[str]] = {}

    def _write_mediamtx_config(self) -> str:
        lines = [
            "logLevel: info",
            "rtsp: true",
            "rtspTransports: [tcp]",
            f"rtspAddress: :{self.config.rtsp_port}",
            "rtmp: false",
            "hls: false",
            "webrtc: false",
            "srt: false",
            "authMethod: internal",
            "authInternalUsers:",
            "  - user: any",
            '    pass: ""',
            "    ips: []",
            "    permissions:",
            "      - action: publish",
            '        path: ""',
            "      - action: read",
            '        path: ""',
            "pathDefaults:",
            "  source: publisher",
            "paths:",
        ]
        for camera_index in self.config.camera_indices:
            lines.append(f"  {rtsp_path(camera_index)}:")
            lines.append("    source: publisher")
        fd, path = tempfile.mkstemp(prefix="hamster-mediamtx-", suffix=".yml")
        with os.fdopen(fd, "w", encoding="utf-8") as config_file:
            config_file.write("\n".join(lines) + "\n")
        return path

    def start_mediamtx(self) -> bool:
        if _port_open(self.config.rtsp_port):
            print(
                f"RTSP port {self.config.rtsp_port} is already open; "
                "publishing to the existing server"
            )
            return True

        binary = resolve_bin(self.config.mediamtx_bin, "mediamtx")
        if not binary:
            print("mediamtx not found. Download the linux_arm64 build from")
            print("https://github.com/bluenviron/mediamtx/releases")
            print("and place the mediamtx binary next to start.py, or on PATH.")
            return False

        self.mediamtx_config_path = self._write_mediamtx_config()
        try:
            self.mediamtx_proc = subprocess.Popen(
                [binary, self.mediamtx_config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            print(f"Failed to start mediamtx: {exc}")
            self.stop_mediamtx()
            return False

        threading.Thread(
            target=_log_process_output,
            args=(self.mediamtx_proc, "mediamtx: "),
            daemon=True,
        ).start()

        if _wait_port(self.config.rtsp_port, 5):
            print(f"RTSP server listening on :{self.config.rtsp_port}")
            return True

        code = self.mediamtx_proc.poll()
        print(f"mediamtx did not open port {self.config.rtsp_port} (exit {code})")
        self.stop_mediamtx()
        return False

    def stop_mediamtx(self) -> None:
        proc = self.mediamtx_proc
        self.mediamtx_proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self.mediamtx_config_path and os.path.exists(self.mediamtx_config_path):
            try:
                os.remove(self.mediamtx_config_path)
            except OSError:
                pass
        self.mediamtx_config_path = None

    def start(self) -> None:
        if not self.config.rtsp_enabled:
            print("H.264 RTSP is disabled in config.conf")
            return
        if not self.start_mediamtx():
            print("H.264 RTSP did not start. /detection MJPEG is still available.")
            return
        for camera_index in self.config.camera_indices:
            publisher = H264Publisher(
                camera_index,
                self.config,
                self.state,
                self.sensors,
                self._ffmpeg_bin_cache,
            )
            self.state.h264_publishers[camera_index] = publisher
            publisher.thread.start()

    def stop(self) -> None:
        for publisher in self.state.h264_publishers.values():
            publisher.stop()
        self.state.h264_publishers.clear()
        self.stop_mediamtx()
