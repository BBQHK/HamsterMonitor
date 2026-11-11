"""Burn cage status text into viewing-camera frames."""

from __future__ import annotations

from datetime import datetime
from typing import List, TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from config import Config
    from sensors import Sensors
    from state import RuntimeState

FONT_SCALE = 0.5
FONT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
BACKGROUND_ALPHA = 0.5
TEXT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)
TEXT_PADDING = 5


def get_current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_text_overlay(frame, texts: List[str]) -> None:
    if not texts:
        return

    total_height = 0
    max_width = 0
    text_sizes = []

    for text in texts:
        (width, height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        text_sizes.append((width, height))
        max_width = max(max_width, width)
        total_height += height

    padding = TEXT_PADDING
    total_height += padding * (len(texts) + 1)
    max_width += padding * 2

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (padding, padding),
        (padding + max_width, padding + total_height),
        BACKGROUND_COLOR,
        -1,
    )
    cv2.addWeighted(overlay, BACKGROUND_ALPHA, frame, 1 - BACKGROUND_ALPHA, 0, frame)

    y = padding + text_sizes[0][1] + padding
    for i, text in enumerate(texts):
        cv2.putText(
            frame,
            text,
            (padding + 5, y),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )
        if i < len(texts) - 1:
            y += text_sizes[i + 1][1] + padding


def annotate_frame(frame, config: "Config", state: "RuntimeState", sensors: "Sensors") -> None:
    """Burn cage status into a viewing camera. The detection camera stays clean."""
    try:
        temperature, humidity, air_quality, air_quality_ppm = sensors.read()
        texts = [
            f"Time: {get_current_timestamp()}",
            f"Temp: {temperature:.1f}C  Hum: {humidity:.1f}%",
            f"Air Quality: {air_quality} ({air_quality_ppm:.1f} PPM)",
        ]
        if state.api_error_count >= state.api_error_threshold:
            texts.append("Activity: API Unavailable")
        elif state.last_activity_result["activity"] == "Unknown":
            texts.append("Activity: Unknown")
        else:
            texts.append(
                f"Activity: {state.last_activity_result['activity']} "
                f"({state.last_activity_result['activity_probability'] * 100:.1f}%)"
            )
        add_text_overlay(frame, texts)
    except Exception as exc:
        print(f"Error processing frame: {exc}")
        cv2.putText(
            frame,
            f"Error: {str(exc)}",
            (50, config.frame_height // 2),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )
