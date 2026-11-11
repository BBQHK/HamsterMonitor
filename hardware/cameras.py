"""OpenCV camera capture and MJPEG frame encoding."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Generator, Optional

import cv2

if TYPE_CHECKING:
    from config import Config
    from state import RuntimeState


def encode_jpeg(frame) -> Optional[bytes]:
    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return None
    return buffer.tobytes()


class CameraService:
    """Owns VideoCapture objects and per-camera grab threads."""

    def __init__(self, config: "Config", state: "RuntimeState") -> None:
        self.config = config
        self.state = state

    def setup_camera(self, camera_index: int):
        camera = cv2.VideoCapture(camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        camera.set(cv2.CAP_PROP_FPS, self.config.fps)
        # Keep only the newest frame so brief stalls do not build multi-day lag
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera

    def fit_frame(self, frame):
        if frame is None or frame.size == 0:
            return None
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            return None
        if (
            frame.shape[1] != self.config.frame_width
            or frame.shape[0] != self.config.frame_height
        ):
            frame = cv2.resize(
                frame, (self.config.frame_width, self.config.frame_height)
            )
        else:
            frame = frame.copy()
        return frame

    def initialize(self) -> None:
        for camera_index in self.config.camera_indices:
            camera = self.setup_camera(camera_index)
            if camera.isOpened():
                self.state.cameras[camera_index] = camera
                self.state.frame_locks[camera_index] = threading.Lock()
                self.state.latest_frames[camera_index] = None
                thread = threading.Thread(
                    target=self.capture_loop,
                    args=(camera_index,),
                    daemon=True,
                    name=f"camera-capture-{camera_index}",
                )
                self.state.capture_threads[camera_index] = thread
                thread.start()
                print(f"Successfully initialized camera {camera_index}")
            else:
                print(f"Failed to initialize camera {camera_index}")

    def capture_loop(self, camera_index: int) -> None:
        """Grab frames for one camera.

        The detection camera is published as MJPEG on /detection with no overlay.
        Every camera, including that one, is also published as H.264 for Surveillance Station.
        Viewing cameras have the status overlay burned into the H.264 stream.
        """
        camera = self.state.cameras.get(camera_index)
        if camera is None:
            return

        lock = self.state.frame_locks[camera_index]
        is_detection = camera_index == self.config.detection_index

        while not self.state.capture_stop.is_set():
            success, frame = camera.read()
            if not success:
                time.sleep(0.05)
                continue

            frame = self.fit_frame(frame)
            if frame is None:
                time.sleep(0.05)
                continue

            if is_detection:
                jpeg = encode_jpeg(frame)
                if jpeg is not None:
                    with lock:
                        self.state.latest_frames[camera_index] = jpeg

            publisher = self.state.h264_publishers.get(camera_index)
            if publisher is not None:
                publisher.submit(frame)

    def generate_frames(self, camera_index: int) -> Generator[bytes, None, None]:
        """Stream the latest JPEG without touching VideoCapture."""
        if camera_index not in self.state.cameras:
            return

        lock = self.state.frame_locks[camera_index]
        while not self.state.capture_stop.is_set():
            with lock:
                frame = self.state.latest_frames.get(camera_index)

            if frame is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(self.config.frame_interval)

    def release(self) -> None:
        self.state.capture_stop.set()
        for camera in self.state.cameras.values():
            camera.release()
        self.state.cameras.clear()
