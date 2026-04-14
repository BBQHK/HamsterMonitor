/*
 * Hamster wheel counter — receiver (Arduino Pro Mini 3.3 V / 8 MHz + AK-R240108 or similar)
 *
 * IDE: Board = Arduino Pro or Pro Mini, Processor = ATmega328P (3.3V, 8 MHz).
 *
 * Receiver: AK-R240108 (or any raw ASK/OOK demodulator with a digital DATA output).
 * Same software stack as XY-MK-5V: RH_ASK decodes the bit stream from DATA.
 *
 * Wiring (confirm pin labels on your PCB — order varies by batch):
 *   GND  -> Pro Mini GND
 *   VCC  -> Pro Mini VCC (3.3 V) when the module is rated for 3.0–3.6 V / 3.3 V
 *   DATA -> D11 (must match rfDriver rxPin below). Same 3.3 V logic as the MCU — no level shifter.
 * Optional ANT / external wire: only if your board has a pad and the datasheet says so;
 * many R240108 variants use an on-board antenna.
 *
 * Match frequency to the transmitter (433.92 MHz vs 315 MHz).
 *
 * Library: RadioHead (same as transmitter).
 * Serial Monitor: 115200 baud (use a 3.3 V–safe USB serial adapter on Pro Mini RX0/TX0).
 */

#include <RH_ASK.h>
#include <SPI.h>
#include <string.h>

// Speed must match TX (2000). ASK receiver DATA -> D11. D10 is dummy PTT like RadioHead defaults.
RH_ASK rfDriver(2000, 11, 12, 10);

struct __attribute__((packed)) WheelPacket {
  uint8_t magic;
  uint32_t revolutions;
  uint8_t xorChecksum;
};

static uint8_t checksumOf(const WheelPacket& p) {
  uint8_t x = p.magic;
  x ^= (uint8_t)(p.revolutions & 0xFF);
  x ^= (uint8_t)((p.revolutions >> 8) & 0xFF);
  x ^= (uint8_t)((p.revolutions >> 16) & 0xFF);
  x ^= (uint8_t)((p.revolutions >> 24) & 0xFF);
  return x;
}

void setup() {
  Serial.begin(115200);
  if (!rfDriver.init()) {
    Serial.println(F("RH_ASK init failed"));
    while (true) {
      delay(1000);
    }
  }
  Serial.println(F("Hamster wheel RX ready — waiting for packets…"));
}

void loop() {
  uint8_t buf[sizeof(WheelPacket)];
  uint8_t len = sizeof(buf);

  if (!rfDriver.recv(buf, &len)) {
    return;
  }

  if (len != sizeof(WheelPacket)) {
    return;
  }

  WheelPacket pkt;
  memcpy(&pkt, buf, sizeof(pkt));

  if (pkt.magic != 0xA7) {
    return;
  }
  if (pkt.xorChecksum != checksumOf(pkt)) {
    Serial.println(F("Bad checksum (noise or other TX)"));
    return;
  }

  Serial.print(F("Revolutions: "));
  Serial.println(pkt.revolutions);
}
