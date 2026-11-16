from __future__ import annotations

import statistics
from typing import Iterable

from .models import CorpusDoc, RetrievalCase, StrategyName
from .strategies import run_strategy


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[idx]


def summarize_results(cases: list[RetrievalCase], results: list) -> dict:
    n = len(results)
    if n == 0 or len(cases) != n:
        return {"count": 0}

    passed = sum(1 for r in results if r.checks and all(r.checks.values()))
    recalls = [r.recall_at_k for r in results]
    mean_recall = statistics.mean(recalls) if recalls else 0.0

    metadata_idx = [i for i, c in enumerate(cases) if c.expects.require_metadata_match or "metadata" in c.tags]
    metadata_ok = sum(1 for i in metadata_idx if results[i].metadata_respected) / max(1, len(metadata_idx))

    tenant_idx = [i for i, c in enumerate(cases) if c.expects.require_tenant_isolation or "tenant" in c.tags]
    tenant_ok = sum(1 for i in tenant_idx if results[i].tenant_isolated) / max(1, len(tenant_idx))

    hybrid_idx = [i for i, c in enumerate(cases) if "hybrid" in c.tags]
    hybrid_recall = (
        sum(results[i].recall_at_k for i in hybrid_idx) / max(1, len(hybrid_idx)) if hybrid_idx else mean_recall
    )

    chunk_idx = [i for i, c in enumerate(cases) if "chunking" in c.tags]
    chunk_pass = sum(1 for i in chunk_idx if results[i].checks.get("chunks_indexed", True)) / max(1, len(chunk_idx))

    latencies = [r.latency_ms for r in results]

    return {
        "count": n,
        "pass_rate": round(passed / n, 4),
        "success_rate": round(passed / n, 4),
        "mean_recall_at_k": round(mean_recall, 4),
        "hybrid_recall_at_k": round(hybrid_recall, 4),
        "metadata_filter_accuracy": round(metadata_ok, 4),
        "tenant_isolation_rate": round(tenant_ok, 4),
        "chunking_pass_rate": round(chunk_pass, 4),
        "latency_ms_p95": round(_percentile(latencies, 95), 3),
        "latency_ms_mean": round(statistics.mean(latencies) if latencies else 0.0, 3),
        "mean_chunks_indexed": round(statistics.mean([r.chunks_indexed for r in results]), 2),
    }


def compare_strategies(
    cases: list[RetrievalCase],
    corpus: list[CorpusDoc],
    strategies: Iterable[StrategyName],
) -> dict:
    per_strategy: dict[str, dict] = {}
    for name in strategies:
        results = run_strategy(cases, corpus, name)
        per_strategy[name] = {
            "summary": summarize_results(cases, results),
            "failures": [
                {"case_id": r.case_id, "reason": r.reason, "recall_at_k": r.recall_at_k}
                for r in results
                if r.checks and not all(r.checks.values())
            ][:5],
        }
    primary = list(strategies)[-1]
    return {
        "summary": {k: v["summary"] for k, v in per_strategy.items()},
        "strategies": per_strategy,
        "primary_strategy": primary,
    }