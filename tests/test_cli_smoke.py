import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTests(unittest.TestCase):
    def test_evaluate_pipeline_cli(self) -> None:
        out = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "retrieval_strategies.cli",
                "evaluate-pipeline",
                "--profile",
                "ci-retrieval-eval",
            ],
            cwd=ROOT,
            text=True,
        )
        report = json.loads(out)
        self.assertEqual(report["eval_type"], "retrieval_pipeline")
        self.assertIn("recursive_hybrid_single", report["summary"])


if __name__ == "__main__":
    unittest.main()