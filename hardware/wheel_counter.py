import cv2
import threading

# Camera and ROI — adjust x/y/w/h once you see the overlay on /camera0
WHEEL_COUNTER_CAMERA = 0
WHEEL_ROI = (260, 180, 100, 80)  # x, y, width, height (640x480 frame)

# Grayscale mean below this = sticker (black), above = wheel background (white)
BRIGHTNESS_THRESHOLD = 80

# Require this many consecutive frames in the new state before accepting a transition
DEBOUNCE_FRAMES = 2


class WheelCounter:
    """Counts wheel revolutions by detecting black/white transitions in a fixed ROI."""

    def __init__(
        self,
        roi=WHEEL_ROI,
        threshold=BRIGHTNESS_THRESHOLD,
        debounce_frames=DEBOUNCE_FRAMES,
    ):
        self.roi = roi
        self.threshold = threshold
        self.debounce_frames = debounce_frames

        self._lock = threading.Lock()
        self.revolutions = 0
        self.brightness = 0.0
        self.is_dark = False
        self._stable_state = None  # True = dark, False = light, None = not yet known
        self._pending_state = None
        self._pending_count = 0

    def _classify(self, brightness):
        return brightness < self.threshold

    def _update_state_machine(self, is_dark):
        if self._stable_state is None:
            self._stable_state = is_dark
            self._pending_state = None
            self._pending_count = 0
            return

        if is_dark == self._stable_state:
            self._pending_state = None
            self._pending_count = 0
            return

        if is_dark != self._pending_state:
            self._pending_state = is_dark
            self._pending_count = 1
        else:
            self._pending_count += 1

        if self._pending_count >= self.debounce_frames:
            # Light -> dark: sticker entered the ROI = one revolution
            if self._stable_state is False and self._pending_state is True:
                self.revolutions += 1
            self._stable_state = self._pending_state
            self._pending_state = None
            self._pending_count = 0

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
        is_dark = self._classify(brightness)

        with self._lock:
            self.brightness = brightness
            self.is_dark = is_dark
            self._update_state_machine(is_dark)
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self):
        state = "BLACK" if self.is_dark else "WHITE"
        if self._stable_state is None:
            state = "CALIBRATING"
        return {
            "revolutions": self.revolutions,
            "brightness": round(self.brightness, 1),
            "threshold": self.threshold,
            "state": state,
            "roi": self.roi,
        }

    def get_status(self):
        with self._lock:
            return self._snapshot_unlocked()

    def draw_overlay(self, frame, status=None):
        """Draw ROI box and live readings on the frame."""
        if status is None:
            status = self.get_status()

        x, y, w, h = status["roi"]
        is_dark = status["state"] == "BLACK"
        box_color = (0, 0, 255) if is_dark else (0, 255, 0)  # red = black, green = white
        label_bg = (0, 0, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        lines = [
            "Wheel ROI",
            f"Brightness: {status['brightness']:.1f} (thr {status['threshold']})",
            f"State: {status['state']}",
            f"Revolutions: {status['revolutions']}",
        ]

        line_height = 18
        box_w = 280
        box_h = line_height * len(lines) + 10
        box_x = x
        box_y = max(0, y - box_h - 4)

        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), label_bg, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (box_x + 6, box_y + 16 + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame
