import unittest
from pathlib import Path

from retrieval_strategies.data import load_corpus, load_golden, validate_golden

ROOT = Path(__file__).resolve().parents[1]


class GoldenDataTests(unittest.TestCase):
    def test_corpus_non_empty(self) -> None:
        corpus = load_corpus(ROOT / "data/corpus.json")
        self.assertGreaterEqual(len(corpus), 5)

    def test_golden_cases_labeled(self) -> None:
        cases = load_golden(ROOT / "data/golden.json")
        self.assertGreaterEqual(len(cases), 8)
        for case in cases:
            self.assertTrue(case.id)

    def test_golden_refs(self) -> None:
        corpus = load_corpus(ROOT / "data/corpus.json")
        cases = load_golden(ROOT / "data/golden.json")
        validate_golden(corpus, cases)


if __name__ == "__main__":
    unittest.main()