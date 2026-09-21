from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .domain import Sample, ValidationError
from .geometry import extract_features
from .models import HeuristicPlaceholderModel, QualityModel

DECISIONS = ("keep", "reject", "human_review")


class CurationPipeline:
    def __init__(self, config: dict[str, Any], model: QualityModel | None = None):
        self.config = config
        self.model = model or HeuristicPlaceholderModel(config)
        routing = config["routing"]
        self.reject_max = float(routing["reject_max_score"])
        self.keep_min = float(routing["keep_min_score"])
        if not 0 <= self.reject_max < self.keep_min <= 1:
            raise ValueError("routing thresholds must satisfy 0 <= reject < keep <= 1")

    @classmethod
    def from_config_path(
        cls, path: str | Path, model: QualityModel | None = None
    ) -> CurationPipeline:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), model=model)

    def route(self, score: float) -> str:
        if score <= self.reject_max:
            return "reject"
        if score >= self.keep_min:
            return "keep"
        return "human_review"

    def process_record(self, record: dict[str, Any], base_dir: Path) -> dict[str, Any]:
        sample = Sample.from_dict(record, base_dir)
        features = extract_features(sample)
        score = float(self.model.predict_quality(features, sample))
        if not 0 <= score <= 1:
            raise ValueError(f"model returned score outside [0, 1] for {sample.sample_id}")
        decision = self.route(score)
        result = dict(record)
        result["curation"] = {
            "features": features.to_dict(),
            "quality_score": score,
            "decision": decision,
            "reasons": self._reasons(features, sample),
            "model_version": self.model.version,
        }
        return result

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        dataset_version: str = "unspecified",
        fail_fast: bool = False,
    ) -> dict[str, Any]:
        input_path = Path(input_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        buckets: dict[str, list[dict[str, Any]]] = {decision: [] for decision in DECISIONS}
        errors: list[dict[str, Any]] = []

        for line_number, record, parse_error in _read_jsonl(input_path):
            if parse_error is not None:
                if fail_fast:
                    raise ValidationError(parse_error)
                errors.append({"line": line_number, "sample_id": None, "error": parse_error})
                continue
            assert record is not None
            try:
                result = self.process_record(record, input_path.parent)
                buckets[result["curation"]["decision"]].append(result)
            except (ValidationError, ValueError, TypeError, KeyError) as exc:
                if fail_fast:
                    raise
                errors.append(
                    {"line": line_number, "sample_id": record.get("sample_id"), "error": str(exc)}
                )

        for decision, records in buckets.items():
            _atomic_write_jsonl(output_dir / f"{decision}.jsonl", records)
        _atomic_write_jsonl(output_dir / "errors.jsonl", errors)

        counts = Counter({decision: len(records) for decision, records in buckets.items()})
        processed = sum(counts.values())
        summary = {
            "dataset_version": dataset_version,
            "model_version": self.model.version,
            "processed": processed,
            "errors": len(errors),
            "counts": dict(counts),
            "rates": {
                decision: (counts[decision] / processed if processed else 0.0)
                for decision in DECISIONS
            },
        }
        _atomic_write_json(output_dir / "summary.json", summary)
        lineage = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_path": str(input_path),
            "input_sha256": _sha256_file(input_path),
            "config_sha256": hashlib.sha256(
                json.dumps(self.config, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "dataset_version": dataset_version,
            "model_version": self.model.version,
            "python_version": platform.python_version(),
            "routing": {"reject_max_score": self.reject_max, "keep_min_score": self.keep_min},
        }
        _atomic_write_json(output_dir / "lineage.json", lineage)
        return summary

    def _reasons(self, features, sample: Sample) -> list[str]:
        cfg = self.config["classes"].get(sample.class_name, self.config["classes"]["default"])
        reasons = []
        ranges = cfg["dimension_ranges"]
        for name, value in (
            ("length", features.length),
            ("width", features.width),
            ("height", features.height),
        ):
            low, high = ranges[name]
            if not low <= value <= high:
                reasons.append(f"{name}_outside_expected_range")
        if features.point_count < cfg["min_point_count"]:
            reasons.append("insufficient_points")
        if not cfg["density_range"][0] <= features.point_density <= cfg["density_range"][1]:
            reasons.append("abnormal_point_density")
        if abs(features.ground_offset) > cfg["max_abs_ground_offset"]:
            reasons.append("ground_misalignment")
        if features.max_iou_3d > cfg["max_iou"]:
            reasons.append("excessive_3d_overlap")
        return reasons or ["no_placeholder_rule_violations"]


def _read_jsonl(
    path: Path,
) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, None, f"invalid JSON: {exc.msg}"
                continue
            if not isinstance(value, dict):
                yield line_number, None, "line must contain a JSON object"
                continue
            yield line_number, value, None


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    _atomic_write(
        path, "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
