"""Retrieval pipeline evaluation scorecard."""

from __future__ import annotations

from pathlib import Path

from .data import load_corpus, load_golden, validate_golden
from .measure_profiles import MeasureProfile
from .measurement_pipeline import run_measurement_pipeline
from .scenario_evaluation import evaluate_scenarios
from .strategies import select_strategies


def build_scorecard(measure_report: dict, *, primary_strategy: str = "recursive_hybrid_single") -> dict:
    core = measure_report.get("summary", {}).get(primary_strategy, {})
    return {
        "primary_strategy": primary_strategy,
        "core_retrieval": {
            "success_rate": core.get("success_rate"),
            "mean_recall_at_k": core.get("mean_recall_at_k"),
            "hybrid_recall_at_k": core.get("hybrid_recall_at_k"),
            "metadata_filter_accuracy": core.get("metadata_filter_accuracy"),
            "tenant_isolation_rate": core.get("tenant_isolation_rate"),
            "latency_ms_p95": core.get("latency_ms_p95"),
        },
        "rerank_complement": "https://github.com/CaioHAndradeLima/reranking-strategies",
        "reference_doc": "retrieval-decision-based.md",
    }


def run_retrieval_pipeline_eval(root: Path, profile: MeasureProfile) -> dict:
    corpus = load_corpus(root / "data/corpus.json")
    cases = load_golden(root / "data/golden.json")
    validate_golden(corpus, cases)

    strategies = select_strategies(list(profile.strategies))
    measure_report = run_measurement_pipeline(cases, corpus, strategies=strategies)
    primary = strategies[0] if strategies else "recursive_hybrid_single"
    scenarios = evaluate_scenarios(cases, corpus, strategy=primary)  # type: ignore[arg-type]
    hybrid_scenarios = evaluate_scenarios(cases, corpus, strategy="recursive_hybrid_single")
    scenarios["hybrid_recall_at_k"] = max(
        scenarios.get("hybrid_recall_at_k", 0.0),
        hybrid_scenarios.get("hybrid_recall_at_k", 0.0),
    )
    scenarios["metadata_filter_accuracy"] = max(
        scenarios.get("metadata_filter_accuracy", 0.0),
        hybrid_scenarios.get("metadata_filter_accuracy", 0.0),
    )

    flat_summary = measure_report["summary"]
    return {
        **measure_report,
        "eval_type": "retrieval_pipeline",
        "profile": {"name": profile.name, "description": profile.description},
        "scorecard": build_scorecard(measure_report, primary_strategy=primary),
        "scenarios": scenarios,
        "summary": flat_summary,
    }