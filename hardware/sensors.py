"""DHT temperature/humidity and MQ-135 air quality sensors."""

from __future__ import annotations

import threading
import time
from typing import Tuple

import adafruit_ads1x15.ads1115 as ADS
import adafruit_dht
import board
import busio
from adafruit_ads1x15.analog_in import AnalogIn

from state import RuntimeState

# DHT22 settings
DHT_PIN = board.D4
SENSOR_READ_INTERVAL = 2

# MQ-135 settings
RO_CLEAN_AIR = 9.20
RL = 1.0
VOLTAGE_SUPPLY = 5.0
A_NH3 = 102.2
B_NH3 = -2.243


class Sensors:
    """Background sensor reader backed by RuntimeState cache."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state
        self._thread: threading.Thread | None = None
        self.dht_device = adafruit_dht.DHT22(DHT_PIN)
        self.mq135_channel = self._init_mq135()

    def _init_mq135(self) -> AnalogIn:
        try:
            print("Initializing I2C bus...")
            i2c = busio.I2C(board.SCL, board.SDA)
            print("I2C bus initialized successfully")

            print("Initializing ADS1115...")
            ads = ADS.ADS1115(i2c)
            print("ADS1115 initialized successfully")

            channel = AnalogIn(ads, ADS.P0)
            print("MQ-135 channel configured successfully")
            return channel
        except Exception as exc:
            print(f"Error initializing I2C or ADS1115: {exc}")
            print("Please check your connections:")
            print("1. ADS1115 VDD -> Raspberry Pi 3.3V")
            print("2. ADS1115 GND -> Raspberry Pi GND")
            print("3. ADS1115 SDA -> Raspberry Pi GPIO2 (Pin 3)")
            print("4. ADS1115 SCL -> Raspberry Pi GPIO3 (Pin 5)")
            print("\nRun 'i2cdetect -y 1' to check if the device is detected")
            raise

    @staticmethod
    def get_mq135_resistance(voltage: float) -> float:
        if voltage == 0:
            return float("inf")
        return ((VOLTAGE_SUPPLY * RL) / voltage) - RL

    @staticmethod
    def get_mq135_ppm(resistance: float) -> float:
        if resistance == float("inf"):
            return 0
        rs_r0 = resistance / RO_CLEAN_AIR
        return A_NH3 * rs_r0 ** B_NH3

    def get_air_quality(self) -> Tuple[str, float]:
        try:
            voltage = self.mq135_channel.voltage
            resistance = self.get_mq135_resistance(voltage)
            ppm = self.get_mq135_ppm(resistance)

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
        except Exception as exc:
            print(f"Error reading air quality: {exc}")
            return "Unknown", 0.0

    def read(self) -> Tuple[float, float, str, float]:
        readings = self.state.last_sensor_readings
        return (
            readings["temperature"],
            readings["humidity"],
            readings["air_quality"],
            readings["air_quality_ppm"],
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._read_loop,
            name="sensors",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        while True:
            try:
                current_time = time.time()
                last_read = self.state.last_sensor_readings["last_read_time"]
                if current_time - last_read >= SENSOR_READ_INTERVAL:
                    max_retries = 3
                    retry_delay = 0.5

                    for attempt in range(max_retries):
                        try:
                            time.sleep(0.1)
                            temperature = self.dht_device.temperature
                            humidity = self.dht_device.humidity

                            if humidity is not None and temperature is not None:
                                air_quality, air_quality_ppm = self.get_air_quality()
                                self.state.last_sensor_readings.update(
                                    {
                                        "temperature": temperature,
                                        "humidity": humidity,
                                        "air_quality": air_quality,
                                        "air_quality_ppm": air_quality_ppm,
                                        "last_read_time": current_time,
                                    }
                                )
                                break
                            print(f"Attempt {attempt + 1}: Invalid readings, retrying...")
                            time.sleep(retry_delay)
                        except Exception as exc:
                            print(f"Attempt {attempt + 1}: Error reading sensors: {exc}")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)

                time.sleep(0.1)
            except Exception as exc:
                print(f"Error in sensor reading thread: {exc}")
                time.sleep(1)
