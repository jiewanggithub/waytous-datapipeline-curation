from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import CurationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate augmented point-cloud samples.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL manifest")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="Pipeline configuration JSON",
    )
    parser.add_argument("--dataset-version", default="unspecified")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = CurationPipeline.from_config_path(args.config)
    summary = pipeline.run(
        input_path=args.input,
        output_dir=args.output,
        dataset_version=args.dataset_version,
        fail_fast=args.fail_fast,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
