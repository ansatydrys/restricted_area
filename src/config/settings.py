from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application settings."""

    video_path: Path = Path("test.mp4")
    restricted_zones_path: Path = Path("data/restricted_zones.json")
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.4
    alarm_cooldown_seconds: float = 3.0
    use_tracking: bool = True
    tracker_config: str | None = "bytetrack.yaml"


settings = Settings()



