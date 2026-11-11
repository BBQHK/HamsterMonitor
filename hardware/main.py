"""Wire sensors, cameras, RTSP, and Flask into one hardware application."""

from __future__ import annotations

from cameras import CameraService
from config import Config, load_config
from detection import DetectionPoller
from paths import rtsp_path
from rtsp import RtspService
from sensors import Sensors
from state import RuntimeState
from web import create_flask_app


class HardwareApp:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.state = RuntimeState()
        self.sensors = Sensors(self.state)
        self.cameras = CameraService(self.config, self.state)
        self.rtsp = RtspService(self.config, self.state, self.sensors)
        self.detection = DetectionPoller(self.config, self.state)
        self.flask_app = create_flask_app(self)

    def run(self, host: str = "0.0.0.0", port: int = 8081) -> None:
        config = self.config
        try:
            print(
                f"Detection camera: {config.detection_index} "
                "(no overlay, MJPEG at /detection)"
            )
            self.rtsp.start()
            for camera_index in config.camera_indices:
                print(
                    f"Surveillance Station RTSP: "
                    f"rtsp://<pi>:{config.rtsp_port}/{rtsp_path(camera_index)}"
                )
            self.cameras.initialize()
            self.sensors.start()
            self.detection.start()
            self.flask_app.run(host=host, port=port, threaded=True)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.state.capture_stop.set()
        self.rtsp.stop()
        self.cameras.release()


def main() -> None:
    HardwareApp().run()


if __name__ == "__main__":
    main()
