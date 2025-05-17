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
A_CO2 = 110.0  # CO2 curve constant (example, calibrate for your sensor)
B_CO2 = -2.3   # CO2 curve constant (example, calibrate for your sensor)

# Initialize ADS1115
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
channel = AnalogIn(ads, ADS.P0)

try:
    while True:
        vout = channel.voltage
        rs = RL * ((VCC / vout) - 1)
        rs_r0 = rs / R0
        ppm = A_CO2 * (rs_r0 ** B_CO2)
        print(f"Voltage: {vout:.3f}V, RS: {rs:.0f}Ω, RS/R0: {rs_r0:.3f}, CO2: {ppm:.2f} ppm")
        time.sleep(1)
except KeyboardInterrupt:
    print("Program terminated")