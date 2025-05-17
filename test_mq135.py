import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Initialize I2C bus and ADS1115
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)

# Configure ADS1115 to read from AIN0 (single-ended)
channel = AnalogIn(ads, ADS.P0)

try:
    while True:
        voltage = channel.voltage
        raw_value = channel.value
        print(f"Raw Value: {raw_value}, Voltage: {voltage:.3f}V")
        time.sleep(1)
except KeyboardInterrupt:
    print("Program terminated")