import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import math

# Constants
RL = 1  # Load resistor in kohms (check module)
VCC = 5.0   # MQ-135 supply voltage
R0 = 7.37   # Calibrated in clean air (example)
A_NH3 = 102.2  # NH3 curve constant (example, calibrate for your sensor)
B_NH3 = -2.243   # NH3 curve constant (example, calibrate for your sensor)

# Initialize ADS1115
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
channel = AnalogIn(ads, ADS.P0)

try:
    while True:
        vout = channel.voltage
        rs = ((5.0 * RL) / vout) - RL
        rs_r0 = rs / R0
        ppm = A_NH3 * rs_r0 ** B_NH3
        print(f"Voltage: {vout:.3f}V, RS: {rs:.0f}Ω, RS/R0: {rs_r0:.3f}, NH3: {ppm:.2f} ppm")
        time.sleep(1)
except KeyboardInterrupt:
    print("Program terminated")