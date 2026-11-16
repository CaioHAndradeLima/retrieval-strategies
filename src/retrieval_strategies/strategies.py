from __future__ import annotations

from typing import Iterable

from .models import CorpusDoc, RetrievalCase, RetrievalResult, StrategyName
from .retrieval_pipeline import run_retrieval

DEFAULT_STRATEGIES: tuple[StrategyName, ...] = (
    "fixed_dense_single",
    "recursive_hybrid_single",
    "semantic_hybrid_tenant",
    "metadata_filter_partition",
)


def select_strategies(names: list[str] | None) -> tuple[StrategyName, ...]:
    if not names:
        return DEFAULT_STRATEGIES
    allowed = set(DEFAULT_STRATEGIES)
    out: list[StrategyName] = []
    for name in names:
        if name not in allowed:
            raise ValueError(f"Unknown strategy '{name}'")
        out.append(name)  # type: ignore[arg-type]
    return tuple(out)


def run_strategy(
    cases: Iterable[RetrievalCase],
    docs: list[CorpusDoc],
    strategy: StrategyName,
) -> list[RetrievalResult]:
    return [run_retrieval(case, docs, strategy=strategy) for case in cases]