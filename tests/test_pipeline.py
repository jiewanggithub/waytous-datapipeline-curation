import json
import tempfile
import unittest
from pathlib import Path

from waytous_curation.pipeline import CurationPipeline

ROOT = Path(__file__).resolve().parents[1]


class FixedModel:
    version = "fixed-test-model"

    def __init__(self, score):
        self.score = score

    def predict_quality(self, features, sample):
        return self.score


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/default.json").read_text())

    def test_three_way_routing_boundaries(self):
        pipeline = CurationPipeline(self.config)
        self.assertEqual(pipeline.route(0.25), "reject")
        self.assertEqual(pipeline.route(0.26), "human_review")
        self.assertEqual(pipeline.route(0.79), "human_review")
        self.assertEqual(pipeline.route(0.80), "keep")

    def test_custom_model_is_injectable(self):
        pipeline = CurationPipeline(self.config, model=FixedModel(0.9))
        record = json.loads((ROOT / "examples/samples.jsonl").read_text().splitlines()[0])
        result = pipeline.process_record(record, ROOT / "examples")
        self.assertEqual(result["curation"]["decision"], "keep")
        self.assertEqual(result["curation"]["model_version"], "fixed-test-model")

    def test_end_to_end_writes_manifests_and_lineage(self):
        pipeline = CurationPipeline(self.config)
        with tempfile.TemporaryDirectory() as directory:
            summary = pipeline.run(ROOT / "examples/samples.jsonl", directory, "test-v1")
            output = Path(directory)
            self.assertEqual(summary["processed"], 3)
            self.assertEqual(summary["errors"], 0)
            for name in ["keep", "reject", "human_review", "errors"]:
                self.assertTrue((output / f"{name}.jsonl").is_file())
            lineage = json.loads((output / "lineage.json").read_text())
            self.assertEqual(lineage["dataset_version"], "test-v1")
            self.assertEqual(len(lineage["input_sha256"]), 64)

    def test_bad_lines_are_quarantined_without_stopping_batch(self):
        pipeline = CurationPipeline(self.config)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "bad.jsonl"
            input_path.write_text('{not-json}\n{"sample_id": "missing-fields"}\n')
            summary = pipeline.run(input_path, Path(directory) / "result")
            self.assertEqual(summary["processed"], 0)
            self.assertEqual(summary["errors"], 2)


if __name__ == "__main__":
    unittest.main()
