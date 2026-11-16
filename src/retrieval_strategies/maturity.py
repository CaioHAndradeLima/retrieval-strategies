from __future__ import annotations


def lab_maturity_snapshot() -> dict:
    return {
        "prototype": {
            "recommended_strategy": "fixed_dense_single",
            "in_repo": True,
            "notes": "Fixed-size chunks and dense-only search on a single shared index.",
        },
        "growing_production": {
            "recommended_strategy": "recursive_hybrid_single",
            "in_repo": True,
            "notes": "Paragraph-aware recursive chunking with BM25+dense RRF hybrid retrieval.",
        },
        "multi_tenant_saas": {
            "recommended_strategy": "semantic_hybrid_tenant",
            "in_repo": True,
            "notes": "Sentence-boundary semantic heuristic chunking with per-tenant index partitions.",
        },
        "regulated_metadata": {
            "recommended_strategy": "metadata_filter_partition",
            "in_repo": True,
            "notes": "Hard metadata filters with tenant-scoped indexes; ci-retrieval-eval gate.",
        },
        "production_deployment_only": [
            "Managed vector database with HNSW tuning",
            "Live embedding model versioning and rollback",
            "Cross-region index replication",
            "Handoff to reranking-strategies for candidate rerank",
            "OTel spans with chunk_id and retrieval mode",
        ],
    }