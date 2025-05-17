"""Configuration settings for the Hamster Monitor system."""

# Camera settings
CAMERA_INDICES = [0, 2, 4]  # List of camera indices to use
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
FRAME_SKIP = 3  # Process every 3rd frame

# API settings
MAIN_API_URL = "http://192.168.50.168:8081/process_frame"
API_ERROR_THRESHOLD = 3

# DHT22 settings
DHT_PIN = "D4"  # GPIO pin number where DHT22 is connected
SENSOR_READ_INTERVAL = 2  # Read sensor every 2 seconds

# MQ-135 settings
RO_CLEAN_AIR = 7.37  # Calibrated in clean air
RL = 1.0  # Load resistance in kOhm
VOLTAGE_SUPPLY = 5.0  # Supply voltage in volts
A_NH3 = 102.2  # NH3 curve constant
B_NH3 = -2.243  # NH3 curve constant

# Text overlay settings
FONT_SCALE = 0.5
FONT_THICKNESS = 1
FONT = "FONT_HERSHEY_SIMPLEX"
BACKGROUND_ALPHA = 0.5
TEXT_COLOR = (255, 255, 255)  # White
BACKGROUND_COLOR = (0, 0, 0)  # Black
TEXT_PADDING = 5

# Web server settings
HOST = '0.0.0.0'
PORT = 8081 