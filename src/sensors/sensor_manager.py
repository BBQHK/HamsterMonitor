"""Sensor management module for handling DHT22 and MQ-135 sensors."""

import time
import threading
from datetime import datetime
import board
import adafruit_dht
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from src.config import settings

class SensorManager:
    def __init__(self):
        self.last_readings = {
            'temperature': 0.0,
            'humidity': 0.0,
            'air_quality': 'Unknown',
            'air_quality_ppm': 0.0,
            'last_read_time': 0
        }
        self._initialize_sensors()
        self._start_background_thread()

    def _initialize_sensors(self):
        """Initialize DHT22 and MQ-135 sensors."""
        try:
            # Initialize DHT22
            self.dht_device = adafruit_dht.DHT22(getattr(board, settings.DHT_PIN))
            
            # Initialize I2C and ADS1115
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.ads = ADS.ADS1115(self.i2c)
            self.mq135_channel = AnalogIn(self.ads, ADS.P0)
            
        except Exception as e:
            print(f"Error initializing sensors: {e}")
            print("Please check your connections:")
            print("1. ADS1115 VDD -> Raspberry Pi 3.3V")
            print("2. ADS1115 GND -> Raspberry Pi GND")
            print("3. ADS1115 SDA -> Raspberry Pi GPIO2 (Pin 3)")
            print("4. ADS1115 SCL -> Raspberry Pi GPIO3 (Pin 5)")
            raise

    def _get_mq135_resistance(self, voltage):
        """Calculate sensor resistance from voltage reading."""
        if voltage == 0:
            return float('inf')
        return ((settings.VOLTAGE_SUPPLY * settings.RL) / voltage) - settings.RL

    def _get_mq135_ppm(self, resistance):
        """Convert resistance to PPM using NH3 curve."""
        if resistance == float('inf'):
            return 0
        rs_r0 = resistance / settings.RO_CLEAN_AIR
        return settings.A_NH3 * rs_r0 ** settings.B_NH3

    def _get_air_quality(self):
        """Read and calculate air quality from MQ-135 sensor."""
        try:
            voltage = self.mq135_channel.voltage
            resistance = self._get_mq135_resistance(voltage)
            ppm = self._get_mq135_ppm(resistance)
            
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

    def _read_sensors(self):
        """Read all sensors and update last_readings."""
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                time.sleep(0.1)
                temperature = self.dht_device.temperature
                humidity = self.dht_device.humidity
                
                if humidity is not None and temperature is not None:
                    air_quality, air_quality_ppm = self._get_air_quality()
                    
                    self.last_readings.update({
                        'temperature': temperature,
                        'humidity': humidity,
                        'air_quality': air_quality,
                        'air_quality_ppm': air_quality_ppm,
                        'last_read_time': time.time()
                    })
                    return True
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error reading sensors: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        return False

    def _background_reading_loop(self):
        """Background thread to continuously read sensors."""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_readings['last_read_time'] >= settings.SENSOR_READ_INTERVAL:
                    self._read_sensors()
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in sensor reading thread: {e}")
                time.sleep(1)

    def _start_background_thread(self):
        """Start the background sensor reading thread."""
        self.thread = threading.Thread(target=self._background_reading_loop, daemon=True)
        self.thread.start()

    def get_readings(self):
        """Get the latest sensor readings."""
        return (
            self.last_readings['temperature'],
            self.last_readings['humidity'],
            self.last_readings['air_quality'],
            self.last_readings['air_quality_ppm']
        )

    def get_status(self):
        """Get current status including timestamp and all sensor readings."""
        temp, hum, air_qual, air_ppm = self.get_readings()
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'cage_temperature': temp,
            'cage_humidity': hum,
            'air_quality': air_qual,
            'air_quality_ppm': air_ppm
        } 