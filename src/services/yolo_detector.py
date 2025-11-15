from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ultralytics import YOLO

from src.core.detections import Detection


class YoloPersonDetector:
    """Wrapper around a YOLO model to detect people."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.4,
        use_tracking: bool = True,
        tracker_config: str | None = "bytetrack.yaml",
    ) -> None:
        self._model = YOLO(str(model_path))
        self._conf = confidence_threshold
        self._use_tracking = use_tracking
        self._tracker_config = tracker_config

    def detect(self, frame) -> list[Detection]:
        """Run detection (or tracking) for a single frame."""
        if self._use_tracking:
            results = self._model.track(
                frame,
                conf=self._conf,
                persist=True,
                tracker=self._tracker_config,
                verbose=False,
                classes=[0],  # person class
            )
        else:
            results = self._model.predict(
                frame,
                conf=self._conf,
                verbose=False,
                classes=[0],
            )
        return list(self._parse_results(results))

    def _parse_results(self, results: Iterable) -> Iterable[Detection]:
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0]) if box.cls is not None else None
                if cls != 0:
                    continue
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                x1, y1, x2, y2 = (
                    float(box.xyxy[0][0]),
                    float(box.xyxy[0][1]),
                    float(box.xyxy[0][2]),
                    float(box.xyxy[0][3]),
                )
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])
                yield Detection(
                    track_id=track_id,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                )
