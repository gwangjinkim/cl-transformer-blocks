import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "model_benchmark", ROOT / "scripts" / "model-benchmark.py"
)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class ModelBenchmarkTests(unittest.TestCase):
    def test_settings_reject_context_overflow(self):
        with self.assertRaisesRegex(ValueError, "context"):
            BENCHMARK.validate_settings(
                {"max_position_embeddings": 16}, batch=1, prefill_tokens=12,
                decode_tokens=5, training_tokens=8, repeats=3, warmups=1,
            )

    def test_summary_reports_latency_and_throughput(self):
        summary = BENCHMARK.summarize("decode", [0.4, 0.2, 0.3], batch=2, tokens=1)
        self.assertEqual(summary["samples_seconds"], [0.4, 0.2, 0.3])
        self.assertAlmostEqual(summary["median_seconds"], 0.3)
        self.assertAlmostEqual(summary["tokens_per_second"], 2 / 0.3)

    def test_report_requires_matching_workload_fingerprints(self):
        native = {"workload": "prefill", "fingerprint": 1.25}
        python = {"workload": "prefill", "fingerprint": 1.25001}
        BENCHMARK.validate_pair(native, python)
        python["fingerprint"] = 2.0
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            BENCHMARK.validate_pair(native, python)

    def test_source_hashes_include_only_checkpoint_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"model")
            (root / "reference-gradients.safetensors").write_bytes(b"test-only")
            self.assertEqual(
                set(BENCHMARK.source_hashes(root)), {"config.json", "model.safetensors"}
            )


if __name__ == "__main__":
    unittest.main()
