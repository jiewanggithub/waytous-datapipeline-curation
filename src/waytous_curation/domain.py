from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """Raised when an input sample cannot be safely processed."""


def _triple(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValidationError(f"{field_name} must contain three numbers")
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must contain three numbers") from exc


@dataclass(frozen=True)
class Box3D:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Box3D:
        if not isinstance(value, dict):
            raise ValidationError("bbox must be an object")
        box = cls(
            center=_triple(value.get("center"), "bbox.center"),
            size=_triple(value.get("size"), "bbox.size"),
            yaw=float(value.get("yaw", 0.0)),
        )
        if any(side <= 0 for side in box.size):
            raise ValidationError("bbox.size values must be positive")
        return box

    @property
    def volume(self) -> float:
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def bottom_z(self) -> float:
        return self.center[2] - self.size[2] / 2.0


@dataclass
class Sample:
    sample_id: str
    class_name: str
    bbox: Box3D
    points: list[tuple[float, float, float]]
    ground_z: float = 0.0
    neighbor_boxes: list[Box3D] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], base_dir: Path) -> Sample:
        if not isinstance(value, dict):
            raise ValidationError("sample must be a JSON object")
        sample_id = str(value.get("sample_id", "")).strip()
        class_name = str(value.get("class_name", "")).strip().lower()
        if not sample_id:
            raise ValidationError("sample_id is required")
        if not class_name:
            raise ValidationError("class_name is required")
        points = _load_points(value, base_dir)
        neighbors = [Box3D.from_dict(item) for item in value.get("neighbor_boxes", [])]
        return cls(
            sample_id=sample_id,
            class_name=class_name,
            bbox=Box3D.from_dict(value.get("bbox")),
            points=points,
            ground_z=float(value.get("ground_z", 0.0)),
            neighbor_boxes=neighbors,
            raw=value,
        )


def _load_points(value: dict[str, Any], base_dir: Path) -> list[tuple[float, float, float]]:
    if "points" in value:
        if not isinstance(value["points"], list):
            raise ValidationError("points must be a list")
        return [_triple(point, "point") for point in value["points"]]
    if "points_path" not in value:
        raise ValidationError("one of points or points_path is required")

    import csv
    import json

    path = Path(str(value["points_path"]))
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise ValidationError(f"points file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [_triple(point, "point") for point in data]
    if suffix == ".jsonl":
        points: list[tuple[float, float, float]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            point = json.loads(line)
            if isinstance(point, dict):
                point = [point.get("x"), point.get("y"), point.get("z")]
            points.append(_triple(point, "point"))
        return points
    if suffix == ".csv":
        points = []
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if not row or [cell.strip().lower() for cell in row[:3]] == ["x", "y", "z"]:
                    continue
                points.append(_triple(row, "point"))
        return points
    raise ValidationError("points_path must end in .json, .jsonl, or .csv")


@dataclass(frozen=True)
class Features:
    length: float
    width: float
    height: float
    volume: float
    point_count: int
    point_density: float
    ground_offset: float
    max_iou_3d: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "volume": self.volume,
            "point_count": self.point_count,
            "point_density": self.point_density,
            "ground_offset": self.ground_offset,
            "max_iou_3d": self.max_iou_3d,
        }
