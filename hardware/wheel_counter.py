import cv2
import threading
import time

# Camera and ROI — adjust x/y/w/h once you see the overlay on /camera0
WHEEL_COUNTER_CAMERA = 0
WHEEL_ROI = (80, 230, 30, 30)  # x, y, width, height (640x480 frame)

# Hysteresis: enter dark below DARK, leave dark above LIGHT (reduces threshold flicker)
BRIGHTNESS_DARK = 80
BRIGHTNESS_LIGHT = 95

# Ignore repeat counts within this window (prevents bounce without needing 2 black frames)
MIN_COUNT_INTERVAL_S = 0.10


class WheelCounter:
    """Counts wheel revolutions by detecting black/white transitions in a fixed ROI."""

    def __init__(
        self,
        roi=WHEEL_ROI,
        dark_threshold=BRIGHTNESS_DARK,
        light_threshold=BRIGHTNESS_LIGHT,
        min_count_interval_s=MIN_COUNT_INTERVAL_S,
    ):
        self.roi = roi
        self.dark_threshold = dark_threshold
        self.light_threshold = light_threshold
        self.min_count_interval_s = min_count_interval_s

        self._lock = threading.Lock()
        self.revolutions = 0
        self.brightness = 0.0
        self.is_dark = False
        self._was_dark = False
        self._last_count_time = 0.0
        self._initialized = False

    def _classify(self, brightness):
        """Schmitt trigger: sticky dark/light so fast passes still register one edge."""
        if self.is_dark:
            if brightness >= self.light_threshold:
                self.is_dark = False
        elif brightness < self.dark_threshold:
            self.is_dark = True
        return self.is_dark

    def _update_state_machine(self, is_dark):
        if not self._initialized:
            self._was_dark = is_dark
            self._initialized = True
            return

        # Count on rising edge: light -> dark (sticker entered ROI)
        if is_dark and not self._was_dark:
            now = time.monotonic()
            if now - self._last_count_time >= self.min_count_interval_s:
                self.revolutions += 1
                self._last_count_time = now

        self._was_dark = is_dark

    def process_frame(self, frame):
        """Sample ROI brightness, update counter, return a copy-safe snapshot."""
        x, y, w, h = self.roi
        x2 = min(x + w, frame.shape[1])
        y2 = min(y + h, frame.shape[0])
        x = max(0, x)
        y = max(0, y)

        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            return self.get_status()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        with self._lock:
            self.brightness = brightness
            is_dark = self._classify(brightness)
            self._update_state_machine(is_dark)
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self):
        state = "BLACK" if self.is_dark else "WHITE"
        return {
            "revolutions": self.revolutions,
            "brightness": round(self.brightness, 1),
            "dark_threshold": self.dark_threshold,
            "light_threshold": self.light_threshold,
            "state": state,
            "roi": self.roi,
        }

    def get_status(self):
        with self._lock:
            return self._snapshot_unlocked()

    def draw_overlay(self, frame, status=None):
        """Draw ROI box on the frame."""
        if status is None:
            status = self.get_status()

        x, y, w, h = status["roi"]
        is_dark = status["state"] == "BLACK"
        box_color = (0, 0, 255) if is_dark else (0, 255, 0)  # red = black, green = white

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        return frame
