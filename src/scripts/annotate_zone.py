from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.core.zones import Zone, ZoneRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate restricted zone on a video frame.")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("test.mp4"),
        help="Video file path used for annotation.",
    )
    parser.add_argument(
        "--zones",
        type=Path,
        default=Path("data/restricted_zones.json"),
        help="Path to save restricted zones JSON.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="restricted_area",
        help="Name of the zone.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Failed to open video: {args.video}")
        return 1

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Unable to read frame from video.")
        return 1

    window_name = "Annotate Restricted Zone"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.setMouseCallback(window_name, on_mouse)

    repo = ZoneRepository(args.zones)
    while True:
        canvas = frame.copy()
        if points:
            contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                canvas,
                [contour],
                isClosed=len(points) >= 3,
                color=(0, 255, 255),
                thickness=2,
            )
            for idx, (x, y) in enumerate(points, start=1):
                cv2.circle(canvas, (x, y), 4, (0, 255, 255), -1)
                cv2.putText(
                    canvas,
                    str(idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        overlay_lines = [
            "Left click: add point",
            "'u': undo last point",
            "'c': clear points",
            "'s' or Enter: save zone",
            "'q': quit without saving",
        ]
        for i, line in enumerate(overlay_lines):
            cv2.putText(
                canvas,
                line,
                (20, 30 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(50) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("u") and points:
            points.pop()
        if key == ord("c"):
            points.clear()
        if key in (ord("s"), 13):
            if len(points) >= 3:
                zone = Zone(name=args.name, points=tuple(points))
                repo.save([zone])
                print(f"Zone saved to {args.zones}")
                break
            else:
                print("Need at least 3 points to define a polygon.")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

