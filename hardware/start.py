from flask import Flask, Response, request
import cv2
import requests
import json
import numpy as np
from datetime import datetime
import board
import adafruit_dht
import time
import threading
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import configparser
import html
import os
import shutil
import socket
import subprocess
import tempfile

def load_config():
    """Load camera and API settings from config.conf next to this script.

    Override the path with the HAMSTER_CONFIG environment variable.
    """
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.conf")
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
    return {
        "camera_indices": indices,
        "detection_index": detection_index,
        "frame_width": parser.getint("camera", "frame_width"),
        "frame_height": parser.getint("camera", "frame_height"),
        "fps": parser.getint("camera", "fps"),
        "frame_skip": parser.getint("camera", "frame_skip"),
        "api_url": api_url,
        "detection_result_url": f"{api_url}/detection_result",
        "rtsp_enabled": rtsp_enabled,
        "rtsp_port": rtsp_port,
        "rtsp_bitrate_kbps": rtsp_bitrate_kbps,
        "ffmpeg_bin": ffmpeg_bin,
        "mediamtx_bin": mediamtx_bin,
    }

_config = load_config()
CAMERA_INDICES = _config["camera_indices"]
DETECTION_CAMERA_INDEX = _config["detection_index"]
FRAME_WIDTH = _config["frame_width"]
FRAME_HEIGHT = _config["frame_height"]
FPS = _config["fps"]
API_URL = _config["api_url"]
DETECTION_RESULT_URL = _config["detection_result_url"]
FRAME_SKIP = _config["frame_skip"]
RTSP_ENABLED = _config["rtsp_enabled"]
RTSP_PORT = _config["rtsp_port"]
RTSP_BITRATE_KBPS = _config["rtsp_bitrate_kbps"]
FFMPEG_BIN_CONFIG = _config["ffmpeg_bin"]
MEDIAMTX_BIN_CONFIG = _config["mediamtx_bin"]

# DHT11 settings
DHT_PIN = board.D4  # GPIO pin number where DHT11 is connected
SENSOR_READ_INTERVAL = 2  # Read sensor every 2 seconds

# MQ-135 settings
RO_CLEAN_AIR = 9.20  # Calibrated in clean air (from test_mq135.py)
RL = 1.0  # Load resistance in kOhm
VOLTAGE_SUPPLY = 5.0  # Supply voltage in volts
A_NH3 = 102.2  # NH3 curve constant
B_NH3 = -2.243  # NH3 curve constant

# Initialize DHT sensor
dht_device = adafruit_dht.DHT22(DHT_PIN)

# Initialize I2C bus
try:
    print("Initializing I2C bus...")
    i2c = busio.I2C(board.SCL, board.SDA)
    print("I2C bus initialized successfully")
    
    print("Initializing ADS1115...")
    ads = ADS.ADS1115(i2c)
    print("ADS1115 initialized successfully")
    
    # Create single-ended input on channel 0
    mq135_channel = AnalogIn(ads, ADS.P0)
    print("MQ-135 channel configured successfully")
except Exception as e:
    print(f"Error initializing I2C or ADS1115: {e}")
    print("Please check your connections:")
    print("1. ADS1115 VDD -> Raspberry Pi 3.3V")
    print("2. ADS1115 GND -> Raspberry Pi GND")
    print("3. ADS1115 SDA -> Raspberry Pi GPIO2 (Pin 3)")
    print("4. ADS1115 SCL -> Raspberry Pi GPIO3 (Pin 5)")
    print("\nRun 'i2cdetect -y 1' to check if the device is detected")
    raise

# Text overlay constants
FONT_SCALE = 0.5
FONT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
BACKGROUND_ALPHA = 0.5
TEXT_COLOR = (255, 255, 255)  # White
BACKGROUND_COLOR = (0, 0, 0)  # Black
TEXT_PADDING = 5

# Initialize Flask app
app = Flask(__name__)

# Dictionary to store camera objects
cameras = {}

# Latest JPEG for the detection camera. Other cameras are published as H.264.
latest_frames = {}
frame_locks = {}
capture_threads = {}
capture_stop = threading.Event()
FRAME_INTERVAL = 1.0 / FPS

h264_publishers = {}
mediamtx_proc = None
mediamtx_config_path = None
_ffmpeg_bin = None


def resolve_bin(configured, name):
    """Find a binary from config, next to this script, or PATH."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if configured:
        candidates.append(configured)
    candidates.append(os.path.join(script_dir, name))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return shutil.which(name)


def rtsp_path(camera_index):
    return f"camera{camera_index}"


def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_port(port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.1)
    return False


def _write_mediamtx_config():
    """Write a temporary MediaMTX config that only serves these camera paths."""
    lines = [
        "logLevel: info",
        "rtsp: true",
        "rtspTransports: [tcp]",
        f"rtspAddress: :{RTSP_PORT}",
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
    for camera_index in CAMERA_INDICES:
        lines.append(f"  {rtsp_path(camera_index)}:")
        lines.append("    source: publisher")
    fd, path = tempfile.mkstemp(prefix="hamster-mediamtx-", suffix=".yml")
    with os.fdopen(fd, "w", encoding="utf-8") as config_file:
        config_file.write("\n".join(lines) + "\n")
    return path


def start_mediamtx():
    """Start MediaMTX, or use an RTSP server that is already listening."""
    global mediamtx_proc, mediamtx_config_path
    if _port_open(RTSP_PORT):
        print(f"RTSP port {RTSP_PORT} is already open; publishing to the existing server")
        return True

    binary = resolve_bin(MEDIAMTX_BIN_CONFIG, "mediamtx")
    if not binary:
        print("mediamtx not found. Download the linux_arm64 build from")
        print("https://github.com/bluenviron/mediamtx/releases")
        print("and place the mediamtx binary next to start.py, or on PATH.")
        return False

    mediamtx_config_path = _write_mediamtx_config()
    try:
        mediamtx_proc = subprocess.Popen(
            [binary, mediamtx_config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        print(f"Failed to start mediamtx: {exc}")
        stop_mediamtx()
        return False
    threading.Thread(
        target=_log_process_output,
        args=(mediamtx_proc, "mediamtx: "),
        daemon=True,
    ).start()
    if _wait_port(RTSP_PORT, 5):
        print(f"RTSP server listening on :{RTSP_PORT}")
        return True

    code = mediamtx_proc.poll()
    print(f"mediamtx did not open port {RTSP_PORT} (exit {code})")
    stop_mediamtx()
    return False


def _log_process_output(proc, prefix):
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(f"{prefix}{line.rstrip()}")


def stop_mediamtx():
    global mediamtx_proc, mediamtx_config_path
    proc = mediamtx_proc
    mediamtx_proc = None
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    if mediamtx_config_path and os.path.exists(mediamtx_config_path):
        try:
            os.remove(mediamtx_config_path)
        except OSError:
            pass
    mediamtx_config_path = None


class H264Publisher:
    """Publish one camera to MediaMTX as baseline H.264 over RTSP.

    Frames are written on a private pipe so ffmpeg's stdin is not the video.
    The capture thread only stores the newest picture. This thread emits one
    frame per tick at the configured FPS and draws the clock at send time, so
    the on-screen second matches the stream timeline.
    """

    def __init__(self, camera_index):
        self.camera_index = camera_index
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

    def submit(self, frame):
        if self.stop_event.is_set():
            return
        with self.cond:
            self.latest = frame
            self.cond.notify()

    def stop(self):
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()
        self._stop_ffmpeg()
        self.thread.join(timeout=3)

    def _ffmpeg_bin(self):
        global _ffmpeg_bin
        if _ffmpeg_bin is None:
            _ffmpeg_bin = resolve_bin(FFMPEG_BIN_CONFIG, "ffmpeg") or ""
        return _ffmpeg_bin

    def _open_ffmpeg(self):
        binary = self._ffmpeg_bin()
        if not binary:
            if not self._warned:
                print("ffmpeg not found. Install it with: sudo apt install ffmpeg")
                self._warned = True
            return None

        read_fd, write_fd = os.pipe()
        gop = str(max(1, FPS))
        url = f"rtsp://127.0.0.1:{RTSP_PORT}/{rtsp_path(self.camera_index)}"
        command = [
            binary, "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-video_size", f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "-framerate", str(FPS),
            "-i", f"pipe:{read_fd}",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-pix_fmt", "yuv420p",
            "-g", gop,
            "-keyint_min", gop,
            "-bf", "0",
            "-b:v", f"{RTSP_BITRATE_KBPS}k",
            "-maxrate", f"{RTSP_BITRATE_KBPS}k",
            "-bufsize", f"{RTSP_BITRATE_KBPS}k",
            "-x264-params", "repeat-headers=1:scenecut=0:force-cfr=1:aud=1:slice-max-size=1200",
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
        print(f"Publishing H.264 rtsp://0.0.0.0:{RTSP_PORT}/{rtsp_path(self.camera_index)}")
        return proc

    def _stop_ffmpeg(self):
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

    def _write_all(self, data):
        with self.proc_lock:
            write_fd = self.write_fd
        if write_fd is None:
            raise BrokenPipeError
        view = memoryview(data)
        while view:
            written = os.write(write_fd, view)
            view = view[written:]

    def _write_frames(self, proc):
        """Write one frame every tick and never burst to catch up.

        ffmpeg stamps each frame at 1/FPS. A late frame used to be followed by
        several frames sent back to back, so Surveillance Station played that
        burst fast and then waited.
        """
        while not self.stop_event.is_set():
            tick_start = time.monotonic()
            with self.cond:
                frame = self.latest
            if frame is None:
                if proc.poll() is not None:
                    return
                if self.stop_event.wait(timeout=FRAME_INTERVAL):
                    return
                continue

            # Copy before drawing. The capture thread keeps replacing latest.
            frame = frame.copy()
            if self.camera_index != DETECTION_CAMERA_INDEX:
                annotate_frame(frame)
            try:
                self._write_all(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError, ValueError):
                return
            if proc.poll() is not None:
                return

            remain = FRAME_INTERVAL - (time.monotonic() - tick_start)
            if remain > 0 and self.stop_event.wait(timeout=remain):
                return

    def _run(self):
        while not self.stop_event.is_set():
            try:
                proc = self._open_ffmpeg()
            except Exception as exc:
                print(f"H.264 publisher for camera {self.camera_index} failed to start: {exc}")
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
            print(f"H.264 publisher for camera {self.camera_index} exited ({code}); restarting")
            self.stop_event.wait(1)


def start_rtsp():
    """Publish every camera as H.264. /detection is unaffected."""
    if not RTSP_ENABLED:
        print("H.264 RTSP is disabled in config.conf")
        return
    if not start_mediamtx():
        print("H.264 RTSP did not start. /detection MJPEG is still available.")
        return
    for camera_index in CAMERA_INDICES:
        publisher = H264Publisher(camera_index)
        h264_publishers[camera_index] = publisher
        publisher.thread.start()


def stop_rtsp():
    for publisher in h264_publishers.values():
        publisher.stop()
    h264_publishers.clear()
    stop_mediamtx()

# Store last activity result (shared across all cameras)
last_activity_result = {
    'activity': 'Unknown',
    'activity_probability': 0.0
}

# Store last sensor readings
last_sensor_readings = {
    'temperature': 0.0,
    'humidity': 0.0,
    'air_quality': 'Unknown',
    'air_quality_ppm': 0.0,
    'last_read_time': 0
}

# Add new global variables for detection result polling
api_error_count = 0
api_error_threshold = 3
sensor_thread = None  # Thread for sensor readings

def get_current_timestamp():
    """Get current timestamp in formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_mq135_resistance(voltage):
    """Calculate sensor resistance from voltage reading."""
    if voltage == 0:
        return float('inf')
    return ((VOLTAGE_SUPPLY * RL) / voltage) - RL

def get_mq135_ppm(resistance):
    """Convert resistance to PPM (parts per million) using NH3 curve."""
    if resistance == float('inf'):
        return 0
    rs_r0 = resistance / RO_CLEAN_AIR
    return A_NH3 * rs_r0 ** B_NH3

def get_air_quality():
    """Read and calculate air quality from MQ-135 sensor."""
    try:
        voltage = mq135_channel.voltage
        resistance = get_mq135_resistance(voltage)
        ppm = get_mq135_ppm(resistance)
        
        # NH3-based air quality thresholds
        if ppm < 50:
            quality = "Excellent"
        elif ppm < 100:
            quality = "Good"
        elif ppm < 200:
            quality = "Moderate"
        elif ppm < 300:
            quality = "Poor"
        else:
            quality = "Very Poor"
            
        return quality, ppm
    except Exception as e:
        print(f"Error reading air quality: {e}")
        return "Unknown", 0.0

def read_sensors_background():
    """Background thread to continuously read sensors."""
    global last_sensor_readings
    while True:
        try:
            current_time = time.time()
            if current_time - last_sensor_readings['last_read_time'] >= SENSOR_READ_INTERVAL:
                max_retries = 3
                retry_delay = 0.5  # seconds
                
                for attempt in range(max_retries):
                    try:
                        # Add a small delay before reading
                        time.sleep(0.1)
                        temperature = dht_device.temperature
                        humidity = dht_device.humidity
                        
                        if humidity is not None and temperature is not None:
                            # Read air quality
                            air_quality, air_quality_ppm = get_air_quality()
                            
                            last_sensor_readings.update({
                                'temperature': temperature,
                                'humidity': humidity,
                                'air_quality': air_quality,
                                'air_quality_ppm': air_quality_ppm,
                                'last_read_time': current_time
                            })
                            break  # Success, exit retry loop
                        else:
                            print(f"Attempt {attempt + 1}: Invalid readings, retrying...")
                            time.sleep(retry_delay)
                            
                    except Exception as e:
                        print(f"Attempt {attempt + 1}: Error reading sensors: {e}")
                        if attempt < max_retries - 1:  # Don't sleep on the last attempt
                            time.sleep(retry_delay)
            
            time.sleep(0.1)  # Small delay between checks
            
        except Exception as e:
            print(f"Error in sensor reading thread: {e}")
            time.sleep(1)  # Longer delay on error

def read_sensors():
    """Get the latest sensor readings from cache."""
    return (last_sensor_readings['temperature'], 
            last_sensor_readings['humidity'], 
            last_sensor_readings['air_quality'], 
            last_sensor_readings['air_quality_ppm'])

def add_text_overlay(frame, texts):
    """Add text overlay to a frame.
    
    Args:
        frame: The frame to add text overlay to
        texts: Array of text strings to display
    """
    if not texts:
        return
        
    # Calculate total height needed and max width
    total_height = 0
    max_width = 0
    text_sizes = []
    
    for text in texts:
        (width, height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        text_sizes.append((width, height))
        max_width = max(max_width, width)
        total_height += height
    
    # Add padding between texts and around the box
    padding = TEXT_PADDING
    total_height += padding * (len(texts) + 1)  # Padding between texts and around the box
    max_width += padding * 2  # Padding on both sides
    
    # Add semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (padding, padding),
        (padding + max_width, padding + total_height),
        BACKGROUND_COLOR,
        -1
    )
    cv2.addWeighted(overlay, BACKGROUND_ALPHA, frame, 1 - BACKGROUND_ALPHA, 0, frame)
    
    # Add text
    y = padding + text_sizes[0][1] + padding
    for i, text in enumerate(texts):
        cv2.putText(
            frame, 
            text, 
            (padding + 5, y), 
            FONT, 
            FONT_SCALE, 
            TEXT_COLOR, 
            FONT_THICKNESS
        )
        if i < len(texts) - 1:
            y += text_sizes[i + 1][1] + padding

def setup_camera(camera_index):
    """Setup a camera with specified index."""
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)
    # Keep only the newest frame so brief stalls do not build multi-day lag
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera

def initialize_cameras():
    """Initialize all cameras and start one capture thread per camera."""
    for camera_index in CAMERA_INDICES:
        camera = setup_camera(camera_index)
        if camera.isOpened():
            cameras[camera_index] = camera
            frame_locks[camera_index] = threading.Lock()
            latest_frames[camera_index] = None
            thread = threading.Thread(
                target=capture_loop,
                args=(camera_index,),
                daemon=True,
                name=f"camera-capture-{camera_index}",
            )
            capture_threads[camera_index] = thread
            thread.start()
            print(f"Successfully initialized camera {camera_index}")
        else:
            print(f"Failed to initialize camera {camera_index}")

def get_camera(camera_index):
    """Get a camera object for the given index."""
    return cameras.get(camera_index)

def fit_frame(frame):
    """Return a BGR frame at the configured size, or None if it cannot be used."""
    if frame is None or frame.size == 0:
        return None
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        return None
    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    else:
        frame = frame.copy()
    return frame


def annotate_frame(frame):
    """Burn cage status into a viewing camera. The detection camera stays clean."""
    try:
        current_time = get_current_timestamp()
        temperature, humidity, air_quality, air_quality_ppm = read_sensors()
        texts = [
            f"Time: {current_time}",
            f"Temp: {temperature:.1f}C  Hum: {humidity:.1f}%",
            f"Air Quality: {air_quality} ({air_quality_ppm:.1f} PPM)"
        ]
        if api_error_count >= api_error_threshold:
            texts.append("Activity: API Unavailable")
        elif last_activity_result['activity'] == "Unknown":
            texts.append("Activity: Unknown")
        else:
            texts.append(
                f"Activity: {last_activity_result['activity']} "
                f"({last_activity_result['activity_probability']*100:.1f}%)"
            )
        add_text_overlay(frame, texts)
    except Exception as e:
        print(f"Error processing frame: {e}")
        cv2.putText(
            frame,
            f"Error: {str(e)}",
            (50, FRAME_HEIGHT // 2),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )


def encode_jpeg(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return None
    return buffer.tobytes()


def capture_loop(camera_index):
    """Grab frames for one camera.

    The detection camera is published as MJPEG on /detection with no overlay.
    Every camera, including that one, is also published as H.264 for Surveillance Station.
    Viewing cameras have the status overlay burned into the H.264 stream.
    """
    camera = get_camera(camera_index)
    if camera is None:
        return

    lock = frame_locks[camera_index]
    is_detection = camera_index == DETECTION_CAMERA_INDEX

    while not capture_stop.is_set():
        success, frame = camera.read()
        if not success:
            time.sleep(0.05)
            continue

        frame = fit_frame(frame)
        if frame is None:
            time.sleep(0.05)
            continue

        if is_detection:
            jpeg = encode_jpeg(frame)
            if jpeg is not None:
                with lock:
                    latest_frames[camera_index] = jpeg

        publisher = h264_publishers.get(camera_index)
        if publisher is not None:
            publisher.submit(frame)

        # camera.read() already waits for the next frame. Sleeping again here
        # made the published rate slower than the 15 fps ffmpeg timeline, so
        # the burned-in clock jumped.

def poll_detection_results():
    """Poll detection results from main.py server."""
    global last_activity_result, api_error_count
    
    while True:
        try:
            # Get detection result from main.py
            response = requests.get(DETECTION_RESULT_URL, timeout=2)
            if response.status_code == 200:
                result = response.json()
                last_activity_result.update({
                    'activity': result.get('activity', 'Unknown'),
                    'activity_probability': result.get('activity_probability', 0.0)
                })
                api_error_count = 0  # Reset error count on success
            else:
                api_error_count += 1
                print(f"Failed to get detection result: {response.status_code}")
                
        except (requests.RequestException, json.JSONDecodeError) as e:
            api_error_count += 1
            print(f"Detection result polling error: {e}")
        except Exception as e:
            print(f"Error in detection result polling: {e}")
            
        time.sleep(0.5)  # Poll every 500ms

def generate_frames(camera_index):
    """Stream the latest JPEG for a camera without touching VideoCapture.

    Clients only consume the newest published frame, so a slow browser cannot
    back up the camera buffer or steal frames from other clients.
    """
    if camera_index not in cameras:
        return

    lock = frame_locks[camera_index]

    while not capture_stop.is_set():
        with lock:
            frame = latest_frames.get(camera_index)

        if frame is None:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(FRAME_INTERVAL)

@app.route('/camera<int:camera_index>')
def camera_feed(camera_index):
    """Return the H.264 RTSP address. Browsers cannot play this URL in an img tag."""
    if camera_index not in CAMERA_INDICES:
        return "Invalid camera index", 400
    host = request.host.split(":")[0]
    url = f"rtsp://{host}:{RTSP_PORT}/{rtsp_path(camera_index)}"
    return f"H.264 RTSP: {url}\n", 200, {"Content-Type": "text/plain; charset=utf-8"}

@app.route('/detection')
def detection_feed():
    """Stream the detection camera with no text overlay."""
    return Response(
        generate_frames(DETECTION_CAMERA_INDEX),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )

@app.route('/')
def index():
    """List H.264 RTSP addresses and show the detection MJPEG preview."""
    host = request.host.split(":")[0]
    if not RTSP_ENABLED:
        notice = "H.264 RTSP is disabled in config.conf."
    elif not h264_publishers:
        notice = "H.264 RTSP did not start. Install ffmpeg and mediamtx, then restart."
    else:
        notice = "Surveillance Station: User Define, Streaming-RTSP, H.264, transport TCP."

    camera_feed_html = ""
    for camera_index in CAMERA_INDICES:
        url = f"rtsp://{host}:{RTSP_PORT}/{rtsp_path(camera_index)}"
        if camera_index == DETECTION_CAMERA_INDEX:
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

@app.route('/status')
def get_status():
    """Return current cage status including timestamp, temperature, humidity, and activity."""
    # Get current readings
    temperature, humidity, air_quality, air_quality_ppm = read_sensors()
    
    status = {
        'timestamp': get_current_timestamp(),
        'cage_temperature': temperature,
        'cage_humidity': humidity,
        'air_quality': air_quality,
        'cage_ammonia_level': air_quality_ppm,
        'hamster_activity': last_activity_result['activity'],
        # 'hamster_activity_probability': last_activity_result['activity_probability']
    }
    
    return json.dumps(status, indent=2)


if __name__ == '__main__':
    try:
        print(f"Detection camera: {DETECTION_CAMERA_INDEX} (no overlay, MJPEG at /detection)")
        start_rtsp()
        for camera_index in CAMERA_INDICES:
            print(f"Surveillance Station RTSP: rtsp://<pi>:{RTSP_PORT}/{rtsp_path(camera_index)}")
        # Initialize all cameras before starting the server
        initialize_cameras()
        
        # Start sensor reading thread
        sensor_thread = threading.Thread(target=read_sensors_background, daemon=True)
        sensor_thread.start()
        
        # Start the detection result polling thread
        detection_polling_thread = threading.Thread(target=poll_detection_results, daemon=True)
        detection_polling_thread.start()
        
        app.run(host='0.0.0.0', port=8081, threaded=True)
    finally:
        capture_stop.set()
        stop_rtsp()
        # Release all camera resources when the application stops
        for camera in cameras.values():
            camera.release()
