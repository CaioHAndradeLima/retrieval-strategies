from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ChunkingName = Literal["fixed", "recursive", "semantic"]
SearchName = Literal["dense", "hybrid", "metadata_filter"]
StorageName = Literal["single_index", "tenant_partition"]

StrategyName = Literal[
    "fixed_dense_single",
    "recursive_hybrid_single",
    "semantic_hybrid_tenant",
    "metadata_filter_partition",
]


@dataclass
class CorpusDoc:
    doc_id: str
    tenant_id: str
    title: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    tenant_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalExpectations:
    relevant_chunk_ids: list[str] = field(default_factory=list)
    min_recall_at_k: float = 0.5
    require_metadata_match: bool = False
    require_tenant_isolation: bool = False
    max_latency_ms: float | None = None
    min_chunks_indexed: int | None = None


@dataclass
class RetrievalCase:
    id: str
    query: str
    tenant_id: str = "default"
    top_k: int = 5
    metadata_filter: dict[str, Any] = field(default_factory=dict)
    expects: RetrievalExpectations = field(default_factory=RetrievalExpectations)
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    case_id: str
    strategy: StrategyName
    retrieved_ids: list[str] = field(default_factory=list)
    recall_at_k: float = 0.0
    chunks_indexed: int = 0
    latency_ms: float = 0.0
    chunking: ChunkingName = "fixed"
    search: SearchName = "dense"
    storage: StorageName = "single_index"
    metadata_respected: bool = True
    tenant_isolated: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    reason: str = ""


STRATEGY_COMPONENTS: dict[StrategyName, tuple[ChunkingName, SearchName, StorageName]] = {
    "fixed_dense_single": ("fixed", "dense", "single_index"),
    "recursive_hybrid_single": ("recursive", "hybrid", "single_index"),
    "semantic_hybrid_tenant": ("semantic", "hybrid", "tenant_partition"),
    "metadata_filter_partition": ("recursive", "metadata_filter", "tenant_partition"),
}