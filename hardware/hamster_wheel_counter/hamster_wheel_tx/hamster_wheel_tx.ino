/*
 * Hamster wheel counter — transmitter (Arduino Pro Mini 3.3V/8MHz + A3144 + FS1000A + FR120N)
 *
 * IDE: Board = Arduino Pro or Pro Mini, Processor = ATmega328P (3.3V, 8 MHz).
 *
 * Power: ATmega stays in POWER_DOWN between events. Wake sources:
 *   - INT0 (D2) on Hall CHANGE
 *   - Watchdog interrupt every ~8 s (timebase for 24 h reset; coarse but no RTC)
 *
 * FS1000A VCC controlled by FR120N opto-MOSFET module (D4 HIGH = ON, LOW = OFF)
 * Wiring:
 *   - Hall D2
 *   - FS1000A DATA D7, dummy PTT D10 (NC)
 *   - FR120N IN+ D4, IN-/GND common
 * Library: RadioHead.
 *
 * RF: send on each new revolution and when the 24 h counter rolls.
 */

 #include <RH_ASK.h>
 #include <SPI.h>
 #include <avr/sleep.h>
 #include <avr/wdt.h>
 #include <avr/power.h>
 #include <avr/interrupt.h>
 
 RH_ASK rfDriver(2000, 11, 7, 10);  // RX=11, TX=7, PTT=10 (NC)
 
 const uint8_t PIN_HALL = 2;
 const uint8_t PIN_LED = LED_BUILTIN;
 const uint8_t PIN_RF_POWER = 4;    // FR120N IN+
 
 const unsigned long DEBOUNCE_RELEASE_US = 45000UL;
 
 // Watchdog tick (8 s). Day length in ticks: 86400/8 = 10800 (~24 h).
 const uint8_t WDP_BITS = (1 << WDP3) | (1 << WDP0);
 const uint32_t SECONDS_PER_WDT = 8;
 const uint32_t SECONDS_PER_DAY = 86400UL;
 
 struct __attribute__((packed)) WheelPacket {
   uint8_t magic;
   uint32_t revolutions;
   uint8_t xorChecksum;
 };
 
 static uint32_t revolutions = 0;
 static uint32_t elapsedSec = 0;
 
 static bool waitingMagnetRelease = false;
 static bool sawHighSinceTrigger = false;
 static unsigned long highSinceUs = 0;
 
 volatile static bool wdtTick = false;
 volatile static bool hallEdge = false;
 
 ISR(WDT_vect) {
   wdtTick = true;
 }
 
 static void hallWakeIsr() {
   hallEdge = true;
 }
 
 static void wdtEnableInterrupt8s() {
   cli();
   MCUSR &= ~(1 << WDRF);
   WDTCSR |= (1 << WDCE) | (1 << WDE);
   WDTCSR = (1 << WDIE) | WDP_BITS;
   sei();
 }
 
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
   digitalWrite(PIN_RF_POWER, HIGH);   // FR120N HIGH = ON
   delay(15);                          // 光耦 + 穩壓
 }
 
 static void rfPowerOff() {
   delay(5);                           // 確保發完
   digitalWrite(PIN_RF_POWER, LOW);    // FR120N LOW = OFF
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
 
 static void applyDayRolloverFromElapsed() {
   while (elapsedSec >= SECONDS_PER_DAY) {
     elapsedSec -= SECONDS_PER_DAY;
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
 
 static void sleepPowerDown() {
   set_sleep_mode(SLEEP_MODE_PWR_DOWN);
   cli();
   sleep_enable();
 #ifdef sleep_bod_disable
   sleep_bod_disable();
 #endif
   sei();
   sleep_cpu();
   sleep_disable();
 }
 
 void setup() {
   pinMode(PIN_HALL, INPUT_PULLUP);
   pinMode(PIN_LED, OUTPUT);
   pinMode(PIN_RF_POWER, OUTPUT);
   digitalWrite(PIN_LED, LOW);
   digitalWrite(PIN_RF_POWER, LOW);    // FR120N OFF
 
   if (!rfDriver.init()) {
     digitalWrite(PIN_LED, HIGH);
     while (true) {
       delay(500);
     }
   }
 
   // Power saving
   ADCSRA &= ~_BV(ADEN);
   power_adc_disable();
   power_twi_disable();
 
   attachInterrupt(digitalPinToInterrupt(PIN_HALL), hallWakeIsr, CHANGE);
   wdtEnableInterrupt8s();
 
   // Initial Hall sync
   if (magnetPresent()) {
     waitingMagnetRelease = true;
     sawHighSinceTrigger = false;
   }
 }
 
 void loop() {
   if (wdtTick) {
     wdtTick = false;
     elapsedSec += SECONDS_PER_WDT;
     applyDayRolloverFromElapsed();
     wdtEnableInterrupt8s();
   }
 
   if (hallEdge || magnetPresent() || waitingMagnetRelease) {
     hallEdge = false;
     processHall();
 
     while (releaseDebouncePending()) {
       processHall();
     }
   }
 
  // While waiting for a full "magnet away" cycle, if the Hall output is still LOW
  // we must not enter PWR_DOWN: there is no edge until the field clears, so the MCU
  // would ignore passes until WDT (~8 s) or a lucky edge. Poll briefly instead.
  if (waitingMagnetRelease && magnetPresent()) {
    delayMicroseconds(200);
  } else if (!releaseDebouncePending()) {
    sleepPowerDown();
  }
}