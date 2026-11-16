from __future__ import annotations

from .evaluation import evaluate
from .models import CorpusDoc, RetrievalCase, StrategyName


def run_measurement_pipeline(
    cases: list[RetrievalCase],
    corpus: list[CorpusDoc],
    *,
    strategies: tuple[StrategyName, ...],
) -> dict:
    report = evaluate(cases, corpus, strategies=strategies)
    return {
        "eval_type": "retrieval_measure",
        "summary": report["summary"],
        "strategies": report["strategies"],
        "primary_strategy": report["primary_strategy"],
    }