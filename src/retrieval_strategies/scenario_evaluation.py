from __future__ import annotations

from .models import CorpusDoc, RetrievalCase, StrategyName
from .strategies import run_strategy


def _subset_pass_rate(
    cases: list[RetrievalCase],
    corpus: list[CorpusDoc],
    *,
    strategy: StrategyName,
    tag: str,
) -> dict:
    subset = [c for c in cases if tag in c.tags]
    if not subset:
        return {"count": 0, "pass_rate": 1.0}
    runs = run_strategy(subset, corpus, strategy)
    passed = sum(1 for r in runs if r.checks and all(r.checks.values()))
    return {"count": len(subset), "pass_rate": round(passed / len(subset), 4)}


def evaluate_scenarios(
    cases: list[RetrievalCase],
    corpus: list[CorpusDoc],
    *,
    strategy: StrategyName = "recursive_hybrid_single",
) -> dict:
    runs = run_strategy(cases, corpus, strategy)

    hybrid_idx = [i for i, c in enumerate(cases) if "hybrid" in c.tags]
    hybrid_recall = sum(runs[i].recall_at_k for i in hybrid_idx) / max(1, len(hybrid_idx))

    meta_idx = [i for i, c in enumerate(cases) if "metadata" in c.tags]
    meta_acc = sum(1 for i in meta_idx if runs[i].metadata_respected) / max(1, len(meta_idx))

    tenant_idx = [i for i, c in enumerate(cases) if "tenant" in c.tags]
    tenant_rate = sum(1 for i in tenant_idx if runs[i].tenant_isolated) / max(1, len(tenant_idx))

    return {
        "chunking": _subset_pass_rate(cases, corpus, strategy=strategy, tag="chunking"),
        "hybrid": _subset_pass_rate(cases, corpus, strategy=strategy, tag="hybrid"),
        "dense": _subset_pass_rate(cases, corpus, strategy=strategy, tag="dense"),
        "metadata": _subset_pass_rate(cases, corpus, strategy=strategy, tag="metadata"),
        "tenant": _subset_pass_rate(cases, corpus, strategy=strategy, tag="tenant"),
        "hybrid_recall_at_k": round(hybrid_recall, 4),
        "metadata_filter_accuracy": round(meta_acc, 4),
        "tenant_isolation_rate": round(tenant_rate, 4),
    }