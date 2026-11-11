"""Flask HTTP routes for RTSP links, MJPEG detection, and cage status."""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

from flask import Flask, Response, request

from overlay import get_current_timestamp
from paths import rtsp_path

if TYPE_CHECKING:
    from main import HardwareApp


def create_flask_app(hw: "HardwareApp") -> Flask:
    app = Flask(__name__)
    config = hw.config
    state = hw.state

    @app.route("/camera<int:camera_index>")
    def camera_feed(camera_index: int):
        """Return the H.264 RTSP address. Browsers cannot play this URL in an img tag."""
        if camera_index not in config.camera_indices:
            return "Invalid camera index", 400
        host = request.host.split(":")[0]
        url = f"rtsp://{host}:{config.rtsp_port}/{rtsp_path(camera_index)}"
        return f"H.264 RTSP: {url}\n", 200, {"Content-Type": "text/plain; charset=utf-8"}

    @app.route("/detection")
    def detection_feed():
        """Stream the detection camera with no text overlay."""
        return Response(
            hw.cameras.generate_frames(config.detection_index),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/")
    def index():
        """List H.264 RTSP addresses and show the detection MJPEG preview."""
        host = request.host.split(":")[0]
        if not config.rtsp_enabled:
            notice = "H.264 RTSP is disabled in config.conf."
        elif not state.h264_publishers:
            notice = (
                "H.264 RTSP did not start. Install ffmpeg and mediamtx, then restart."
            )
        else:
            notice = (
                "Surveillance Station: User Define, Streaming-RTSP, H.264, transport TCP."
            )

        camera_feed_html = ""
        for camera_index in config.camera_indices:
            url = f"rtsp://{host}:{config.rtsp_port}/{rtsp_path(camera_index)}"
            if camera_index == config.detection_index:
                detail = "No overlay. Activity detection still uses /detection (MJPEG)."
            else:
                detail = "Status overlay is burned into this recording."
            camera_feed_html += f"""
                <div class="camera-feed">
                    <h3>camera {camera_index}</h3>
                    <p>{html.escape(detail)}</p>
                    <p><code>{html.escape(url)}</code></p>
                </div>
            """

        return f"""
    <html>
        <head>
            <title>Hamster cameras</title>
            <style>
                body {{ margin: 0; padding: 20px; background: #333; color: white; font-family: sans-serif; }}
                .camera-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    max-width: 1600px;
                    margin: 0 auto;
                }}
                .camera-feed {{
                    background: #444;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .camera-feed h3 {{
                    margin: 0 0 10px 0;
                    color: #4CAF50;
                }}
                code {{ color: #fff; word-break: break-all; }}
                img {{ width: 100%; max-width: 640px; height: auto; background: #222; }}
            </style>
        </head>
        <body>
            <h1>Hamster cameras</h1>
            <p>{html.escape(notice)}</p>
            <div class="camera-grid">
                {camera_feed_html}
            </div>
            <h2>Activity detection</h2>
            <img src="/detection" alt="detection" />
        </body>
    </html>
    """

    @app.route("/status")
    def get_status():
        """Return current cage status including sensors and activity."""
        temperature, humidity, air_quality, air_quality_ppm = hw.sensors.read()
        status = {
            "timestamp": get_current_timestamp(),
            "cage_temperature": temperature,
            "cage_humidity": humidity,
            "air_quality": air_quality,
            "cage_ammonia_level": air_quality_ppm,
            "hamster_activity": state.last_activity_result["activity"],
        }
        return json.dumps(status, indent=2)

    return app
