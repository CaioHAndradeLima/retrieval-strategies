import unittest
from pathlib import Path

from retrieval_strategies.data import load_corpus, load_golden, validate_golden
from retrieval_strategies.retrieval_pipeline import run_retrieval
from retrieval_strategies.strategies import run_strategy

ROOT = Path(__file__).resolve().parents[1]


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = load_corpus(ROOT / "data/corpus.json")
        cls.cases = load_golden(ROOT / "data/golden.json")
        validate_golden(cls.corpus, cls.cases)

    def test_hybrid_finds_hybrid_doc(self) -> None:
        case = next(c for c in self.cases if c.id == "rag-hybrid")
        result = run_retrieval(case, self.corpus, strategy="recursive_hybrid_single")
        self.assertIn("hybrid-search::c0", result.retrieved_ids)

    def test_tenant_partition_isolates(self) -> None:
        case = next(c for c in self.cases if c.id == "tenant-b-isolation")
        result = run_retrieval(case, self.corpus, strategy="semantic_hybrid_tenant")
        self.assertTrue(result.tenant_isolated)
        self.assertIn("tenant-b-policy::c0", result.retrieved_ids)

    def test_metadata_filter_partition(self) -> None:
        case = next(c for c in self.cases if c.id == "metadata-policy-a")
        result = run_retrieval(case, self.corpus, strategy="metadata_filter_partition")
        self.assertTrue(result.metadata_respected)
        self.assertEqual(result.retrieved_ids[0], "tenant-a-policy::c0")

    def test_fixed_dense_single_runs(self) -> None:
        case = next(c for c in self.cases if c.id == "dense-retrieval")
        result = run_retrieval(case, self.corpus, strategy="fixed_dense_single")
        self.assertEqual(result.search, "dense")
        self.assertEqual(result.chunking, "fixed")

    def test_all_strategies_return_results(self) -> None:
        for strategy in (
            "fixed_dense_single",
            "recursive_hybrid_single",
            "semantic_hybrid_tenant",
            "metadata_filter_partition",
        ):
            runs = run_strategy(self.cases, self.corpus, strategy)
            self.assertEqual(len(runs), len(self.cases))
            self.assertTrue(all(r.retrieved_ids for r in runs))


if __name__ == "__main__":
    unittest.main()