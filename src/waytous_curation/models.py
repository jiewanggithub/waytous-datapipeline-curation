from __future__ import annotations

from typing import Any, Protocol

from .domain import Features, Sample


class QualityModel(Protocol):
    """Interface for replacing the placeholder with a trained classifier."""

    version: str

    def predict_quality(self, features: Features, sample: Sample) -> float:
        """Return a calibrated probability that the sample is acceptable."""


class HeuristicPlaceholderModel:
    """Deterministic stand-in used to exercise the end-to-end pipeline."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.version = str(config["model"].get("version", "heuristic-placeholder"))

    def predict_quality(self, features: Features, sample: Sample) -> float:
        class_config = self.config["classes"].get(
            sample.class_name, self.config["classes"]["default"]
        )
        weights = self.config["model"]["weights"]
        ranges = class_config["dimension_ranges"]
        checks = {
            "dimensions": sum(
                lo <= value <= hi
                for value, (lo, hi) in zip(
                    (features.length, features.width, features.height),
                    (ranges["length"], ranges["width"], ranges["height"]),
                )
            )
            / 3.0,
            "point_count": min(1.0, features.point_count / max(1, class_config["min_point_count"])),
            "point_density": _range_score(features.point_density, class_config["density_range"]),
            "ground_alignment": max(
                0.0,
                1.0 - abs(features.ground_offset) / class_config["max_abs_ground_offset"],
            ),
            "overlap": max(0.0, 1.0 - features.max_iou_3d / class_config["max_iou"]),
        }
        total_weight = sum(float(weights[name]) for name in checks)
        score = sum(float(weights[name]) * checks[name] for name in checks) / total_weight
        return round(max(0.0, min(1.0, score)), 6)


def _range_score(value: float, accepted_range: list[float]) -> float:
    low, high = float(accepted_range[0]), float(accepted_range[1])
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, value / low) if low else 0.0
    return max(0.0, 1.0 - (value - high) / max(high, 1e-9))
