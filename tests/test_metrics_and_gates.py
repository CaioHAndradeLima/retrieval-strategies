import unittest
from pathlib import Path

from retrieval_strategies.data import load_corpus, load_golden, validate_golden
from retrieval_strategies.evaluation import evaluate
from retrieval_strategies.measure_profiles import evaluate_gates, load_measure_profiles
from retrieval_strategies.pipeline_evaluation import run_retrieval_pipeline_eval

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "config/measure_profiles.json"


class MetricsGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = load_corpus(ROOT / "data/corpus.json")
        cls.cases = load_golden(ROOT / "data/golden.json")
        validate_golden(cls.corpus, cls.cases)

    def test_evaluate_four_strategies(self) -> None:
        report = evaluate(self.cases, self.corpus)
        for name in (
            "fixed_dense_single",
            "recursive_hybrid_single",
            "semantic_hybrid_tenant",
            "metadata_filter_partition",
        ):
            self.assertIn(name, report["summary"])

    def test_hybrid_beats_fixed_on_hybrid_case(self) -> None:
        report = evaluate(self.cases, self.corpus)
        hybrid = report["summary"]["recursive_hybrid_single"]["mean_recall_at_k"]
        fixed = report["summary"]["fixed_dense_single"]["mean_recall_at_k"]
        self.assertGreaterEqual(hybrid, fixed)

    def test_ci_retrieval_eval_passes(self) -> None:
        profile = load_measure_profiles(PROFILES)["ci-retrieval-eval"]
        report = run_retrieval_pipeline_eval(ROOT, profile)
        gate = evaluate_gates(report, profile)
        self.assertTrue(gate["passed"], msg=gate.get("failures"))

    def test_ci_profiles_load(self) -> None:
        profiles = load_measure_profiles(PROFILES)
        self.assertIn("ci-retrieval-eval", profiles)


if __name__ == "__main__":
    unittest.main()