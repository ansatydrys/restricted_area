from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """Represents a single person detection."""

    track_id: int | None
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2
