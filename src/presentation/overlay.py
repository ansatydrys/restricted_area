from __future__ import annotations

from typing import Iterable

import cv2

from src.core.detections import Detection
from src.core.zones import Zone


def draw_zone(frame, zone: Zone, color: tuple[int, int, int] = (0, 255, 255)) -> None:
    cv2.polylines(frame, [zone.contour], isClosed=True, color=color, thickness=2)
    cv2.putText(
        frame,
        zone.name,
        (zone.points[0][0], zone.points[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_detections(
    frame,
    detections: Iterable[Detection],
    normal_color: tuple[int, int, int] = (0, 255, 0),
    intruder_color: tuple[int, int, int] = (0, 0, 255),
    intruder_ids: set[int | None] | None = None,
) -> None:
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.bbox)
        color = (
            intruder_color
            if intruder_ids is not None and detection.track_id in intruder_ids
            else normal_color
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"id:{detection.track_id}" if detection.track_id is not None else "person"
        label = f"{label} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_alarm(frame, active: bool) -> None:
    if not active:
        return
    text = "ALARM!"
    cv2.putText(
        frame,
        text,
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )

