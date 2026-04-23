/*
 * Hamster wheel counter — transmitter, no MCU sleep, RF switched by FR120N
 *
 * IDE: Board = Arduino Pro or Pro Mini, Processor = ATmega328P (3.3V, 8 MHz).
 *
 * Hardware: Hall (e.g. A3144 / DRV5032) + FS1000A + FR120N opto-MOSFET.
 * Wiring:
 *   - Hall D2
 *   - FS1000A DATA D7, dummy PTT D10 (NC)
 *   - FS1000A VCC via FR120N switched rail; FR120N IN+ D4 (HIGH = ON), IN-/GND common
 *
 * Library: RadioHead.
 * RF: power FS1000A only around send; packet on each revolution and at each 24 h boundary.
 */

#include <RH_ASK.h>
#include <SPI.h>

RH_ASK rfDriver(2000, 11, 7, 10);  // RX=11, TX=7, PTT=10 (NC)

const uint8_t PIN_HALL = 2;
const uint8_t PIN_LED = LED_BUILTIN;
const uint8_t PIN_RF_POWER = 4;    // FR120N IN+ (HIGH = FS1000A supply ON)

const unsigned long DEBOUNCE_RELEASE_US = 45000UL;

const uint32_t SECONDS_PER_DAY = 86400UL;
const uint32_t MS_PER_DAY = SECONDS_PER_DAY * 1000UL;

struct __attribute__((packed)) WheelPacket {
  uint8_t magic;
  uint32_t revolutions;
  uint8_t xorChecksum;
};

static uint32_t revolutions = 0;

/** millis() anchor so (millis() - dayEpochStartMs) is time within the current 24 h window */
static uint32_t dayEpochStartMs = 0;

static bool waitingMagnetRelease = false;
static bool sawHighSinceTrigger = false;
static unsigned long highSinceUs = 0;

static uint8_t checksumOf(const WheelPacket& p) {
  uint8_t x = p.magic;
  x ^= (uint8_t)(p.revolutions & 0xFF);
  x ^= (uint8_t)((p.revolutions >> 8) & 0xFF);
  x ^= (uint8_t)((p.revolutions >> 16) & 0xFF);
  x ^= (uint8_t)((p.revolutions >> 24) & 0xFF);
  return x;
}

static bool magnetPresent() {
  return digitalRead(PIN_HALL) == LOW;
}

static void rfPowerOn() {
  digitalWrite(PIN_RF_POWER, HIGH);
  delay(15);   // opto + RF module settle
}

static void rfPowerOff() {
  delay(5);
  digitalWrite(PIN_RF_POWER, LOW);
}

static void sendCountUpdate() {
  rfPowerOn();

  WheelPacket pkt;
  pkt.magic = 0xA7;
  pkt.revolutions = revolutions;
  pkt.xorChecksum = checksumOf(pkt);

  rfDriver.send((uint8_t*)&pkt, sizeof(pkt));
  rfDriver.waitPacketSent();

  rfPowerOff();
}

static void applyDayRolloverFromMillis() {
  while ((uint32_t)(millis() - dayEpochStartMs) >= MS_PER_DAY) {
    dayEpochStartMs += MS_PER_DAY;
    revolutions = 0;
    sendCountUpdate();
  }
}

static void processHall() {
  const unsigned long nowUs = micros();

  if (waitingMagnetRelease) {
    if (!magnetPresent()) {
      if (!sawHighSinceTrigger) {
        sawHighSinceTrigger = true;
        highSinceUs = nowUs;
      } else if ((unsigned long)(nowUs - highSinceUs) >= DEBOUNCE_RELEASE_US) {
        waitingMagnetRelease = false;
        sawHighSinceTrigger = false;
      }
    } else {
      sawHighSinceTrigger = false;
    }
  } else {
    if (magnetPresent()) {
      revolutions++;
      waitingMagnetRelease = true;
      sawHighSinceTrigger = false;
      sendCountUpdate();
    }
  }
}

static bool releaseDebouncePending() {
  if (!waitingMagnetRelease || magnetPresent() || !sawHighSinceTrigger) {
    return false;
  }
  return (unsigned long)(micros() - highSinceUs) < DEBOUNCE_RELEASE_US;
}

void setup() {
  pinMode(PIN_HALL, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_RF_POWER, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  digitalWrite(PIN_RF_POWER, LOW);

  dayEpochStartMs = millis();

  if (!rfDriver.init()) {
    digitalWrite(PIN_LED, HIGH);
    while (true) {
      delay(500);
    }
  }

  if (magnetPresent()) {
    waitingMagnetRelease = true;
    sawHighSinceTrigger = false;
  }
}

void loop() {
  applyDayRolloverFromMillis();

  processHall();
  while (releaseDebouncePending()) {
    processHall();
  }
}
