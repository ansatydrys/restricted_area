from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from src.config.settings import settings as default_settings
from src.core.alarm import AlarmController
from src.core.zones import ZoneRepository
from src.presentation.overlay import draw_alarm, draw_detections, draw_zone
from src.services.yolo_detector import YoloPersonDetector


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restricted area intrusion detection.")
    parser.add_argument(
        "--video",
        type=Path,
        default=default_settings.video_path,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--zones",
        type=Path,
        default=default_settings.restricted_zones_path,
        help="Path to restricted zones JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_settings.model_name,
        help="YOLO model name or path.",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable object tracking (IDs not guaranteed).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=default_settings.confidence_threshold,
        help="Confidence threshold for detections.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    zone_repo = ZoneRepository(args.zones)
    zones = zone_repo.load()
    if not zones:
        print(
            f"No restricted zones found. Please run `python -m src.scripts.annotate_zone --video {args.video}` first.",
            file=sys.stderr,
        )
        return 1
    zone = zones[0]

    detector = YoloPersonDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        use_tracking=not args.no_tracking,
        tracker_config=default_settings.tracker_config,
    )
    alarm = AlarmController(cooldown_seconds=default_settings.alarm_cooldown_seconds)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video file not found: {video_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        return 1

    window_name = "Restricted Area Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(frame)

            intruder_ids: set[int | None] = set()
            for detection in detections:
                cx, cy = detection.center
                if zone.contains(cx, cy):
                    intruder_ids.add(detection.track_id)

            draw_zone(frame, zone)
            draw_detections(
                frame,
                detections,
                intruder_ids=intruder_ids if intruder_ids else None,
            )
            draw_alarm(frame, alarm.update(bool(intruder_ids)))

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

