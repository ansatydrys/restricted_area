from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import json
import numpy as np

Point = tuple[int, int]


@dataclass(frozen=True)
class Zone:
    """Represents a polygon restricted zone."""

    name: str
    points: tuple[Point, ...]

    @property
    def contour(self) -> np.ndarray:
        return np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))

    def contains(self, x: float, y: float) -> bool:
        import cv2

        point = (float(x), float(y))
        return cv2.pointPolygonTest(self.contour, point, False) >= 0


class ZoneRepository:
    """Persists restricted zones."""

    def __init__(self, storage_path: Path) -> None:
        self._path = storage_path

    def load(self) -> list[Zone]:
        if not self._path.exists():
            return []
        raw = self._path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid zone file at {self._path}") from exc
        zones: list[Zone] = []
        for zone_data in data:
            name = zone_data.get("name", "restricted_area")
            points_data: Iterable[dict[str, int]] = zone_data.get("points", [])
            points: list[Point] = []
            for item in points_data:
                points.append((int(item["x"]), int(item["y"])))
            if points:
                zones.append(Zone(name=name, points=tuple(points)))
        return zones

    def save(self, zones: Sequence[Zone]) -> None:
        payload = [
            {
                "name": zone.name,
                "points": [{"x": x, "y": y} for x, y in zone.points],
            }
            for zone in zones
        ]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

