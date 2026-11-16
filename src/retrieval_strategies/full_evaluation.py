from pathlib import Path

from .data import load_corpus, load_golden, validate_golden
from .evaluation import evaluate, evaluate_implementation_checklist
from .implementation_flow import summarize_implementation_flow
from .maturity import lab_maturity_snapshot
from .scenario_evaluation import evaluate_scenarios
from .strategies import DEFAULT_STRATEGIES


def run_full_evaluation(root: Path) -> dict:
    corpus = load_corpus(root / "data/corpus.json")
    cases = load_golden(root / "data/golden.json")
    validate_golden(corpus, cases)

    assembly = evaluate(cases, corpus, strategies=DEFAULT_STRATEGIES)
    checklist = evaluate_implementation_checklist(cases, corpus)
    scenarios = evaluate_scenarios(cases, corpus, strategy="recursive_hybrid_single")

    return {
        "implementation_strategies": assembly,
        "implementation_flow": summarize_implementation_flow(checklist),
        "implementation_checklist": checklist,
        "maturity": lab_maturity_snapshot(),
        "scenarios": scenarios,
        "decision_based": {
            "reference_doc": "retrieval-decision-based.md",
            "recommended_default": "recursive_hybrid_single",
            "reranking_next_stage": "https://github.com/CaioHAndradeLima/reranking-strategies",
        },
        "coverage": {
            "retrieval_guides": "8/8",
            "lab_coverage_percent": 100,
            "index": "docs/LAB_COVERAGE.md",
        },
    }