import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import math

# Constants
RL = 1000  # Load resistor in ohms (check module)
VCC = 5.0   # MQ-135 supply voltage
R0 = 1559   # Calibrated in clean air (example)
A_NH3 = 25.0  # NH3 curve constant
B_NH3 = -1.5  # NH3 curve constant
TEMP = 24.1   # Temperature in °C (replace with DHT22 reading)
HUMIDITY = 69.0  # Relative humidity in % (replace with DHT22 reading)

# Initialize ADS1115
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
channel = AnalogIn(ads, ADS.P0)

try:
    while True:
        vout = channel.voltage
        rs = RL * ((VCC / vout) - 1)
        # Apply environmental corrections
        correction = (1 + 0.007 * (HUMIDITY - 40)) * (1 + 0.004 * (TEMP - 20))
        rs_corrected = rs / correction
        rs_r0 = rs_corrected / R0
        ppm = A_NH3 * (rs_r0 ** B_NH3)
        print(f"Voltage: {vout:.3f}V, RS: {rs:.0f}Ω, RS/R0: {rs_r0:.3f}, NH3: {ppm:.2f} ppm")
        time.sleep(1)
except KeyboardInterrupt:
    print("Program terminated")