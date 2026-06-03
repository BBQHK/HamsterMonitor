from flask import Flask, Response
import cv2
import requests
import json
import numpy as np
from datetime import datetime
import board
import adafruit_dht
import time
import threading
import subprocess
import shutil
import queue
import os
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Constants
CAMERA_INDICES = [0, 2, 4]  # List of camera indices to use
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
H264_CODEC = os.getenv("H264_CODEC", "libx264")
H264_PRESET = os.getenv("H264_PRESET", "ultrafast")
H264_CRF = os.getenv("H264_CRF", "28")
API_URL = "http://192.168.50.99:8082"  # URL of activity monitor server API
DETECTION_RESULT_URL = f"{API_URL}/detection_result"  # URL for getting detection results
FRAME_SKIP = 3  # Process every 3rd frame

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
    return camera

def initialize_cameras():
    """Initialize all cameras at startup."""
    for camera_index in CAMERA_INDICES:
        camera = setup_camera(camera_index)
        if camera.isOpened():
            cameras[camera_index] = camera
            print(f"Successfully initialized camera {camera_index}")
        else:
            print(f"Failed to initialize camera {camera_index}")

def get_camera(camera_index):
    """Get a camera object for the given index."""
    return cameras.get(camera_index)

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

def prepare_stream_frame(camera_index, frame):
    """Apply sensor/activity overlays before encoding."""
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

        if camera_index != 4:
            add_text_overlay(frame, texts)
    except Exception as e:
        print(f"Error processing frame: {e}")
        cv2.putText(
            frame, f"Error: {str(e)}", (50, FRAME_HEIGHT // 2),
            FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS
        )
    return frame

def build_ffmpeg_command():
    """Build ffmpeg command for H.264 MPEG-TS streaming from raw BGR frames."""
    return [
        'ffmpeg',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}',
        '-r', str(FPS),
        '-i', 'pipe:0',
        '-an',
        '-c:v', H264_CODEC,
        '-preset', H264_PRESET,
        '-tune', 'zerolatency',
        '-crf', H264_CRF,
        '-g', str(FPS),
        '-pix_fmt', 'yuv420p',
        '-f', 'mpegts',
        'pipe:1',
    ]

def generate_h264_stream(camera_index):
    """Encode camera frames to H.264 and stream as MPEG-TS."""
    camera = get_camera(camera_index)
    if camera is None:
        return

    if shutil.which('ffmpeg') is None:
        print("ffmpeg not found; install ffmpeg to enable H.264 streaming")
        return

    proc = subprocess.Popen(
        build_ffmpeg_command(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stop_event = threading.Event()
    output_queue = queue.Queue(maxsize=64)
    stream_end = object()

    def read_encoded_output():
        try:
            while not stop_event.is_set():
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                output_queue.put(chunk)
        finally:
            output_queue.put(stream_end)

    def capture_frames():
        frame_interval = 1.0 / FPS
        next_frame_time = time.monotonic()
        try:
            while not stop_event.is_set():
                success, frame = camera.read()
                if not success:
                    break
                frame = prepare_stream_frame(camera_index, frame)
                proc.stdin.write(frame.tobytes())
                next_frame_time += frame_interval
                delay = next_frame_time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        except BrokenPipeError:
            pass
        except Exception as e:
            print(f"H.264 capture error for camera {camera_index}: {e}")
        finally:
            stop_event.set()
            try:
                proc.stdin.close()
            except Exception:
                pass

    threading.Thread(target=read_encoded_output, daemon=True).start()
    threading.Thread(target=capture_frames, daemon=True).start()

    try:
        while True:
            chunk = output_queue.get()
            if chunk is stream_end:
                break
            yield chunk
    finally:
        stop_event.set()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

@app.route('/camera<int:camera_index>')
def camera_feed(camera_index):
    """Stream H.264 video (MPEG-TS) from specified camera index."""
    if camera_index not in CAMERA_INDICES:
        return "Invalid camera index", 400
    if shutil.which('ffmpeg') is None:
        return "ffmpeg is required for H.264 streaming", 503
    return Response(
        generate_h264_stream(camera_index),
        mimetype='video/mp2t',
        headers={'Cache-Control': 'no-cache, no-store, must-revalidate'},
    )

@app.route('/')
def index():
    """Serve a simple HTML page with camera feeds."""
    camera_feed_html = ""
    for camera_index in CAMERA_INDICES:
        camera_feed_html += f"""
                <div class="camera-feed">
                    <h3>/camera{camera_index}</h3>
                    <video src="/camera{camera_index}" autoplay muted playsinline controls></video>
                </div>
        """

    return f"""
    <html>
        <head>
            <title>Multi-Camera Feed</title>
            <style>
                body {{ margin: 0; padding: 20px; background: #333; color: white; }}
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
                video {{ width: 100%; height: auto; background: #000; }}
            </style>
        </head>
        <body>
            <h1>Multi-Camera Feed</h1>
            <div class="camera-grid">
                {camera_feed_html}
            </div>
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
        # Release all camera resources when the application stops
        for camera in cameras.values():
            camera.release()
