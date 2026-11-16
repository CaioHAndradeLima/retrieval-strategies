from __future__ import annotations

CHECKLIST_ITEMS = (
    "ingestion",
    "fixed_chunking",
    "recursive_chunking",
    "semantic_chunking",
    "dense_search",
    "hybrid_search",
    "metadata_filter",
    "tenant_partition",
    "regression_gates",
)


def summarize_implementation_flow(checklist_eval: dict) -> dict:
    by_tag = checklist_eval.get("by_tag", {})
    covered = [item for item in CHECKLIST_ITEMS if _item_covered(item, by_tag, checklist_eval)]
    return {
        "checklist_items": list(CHECKLIST_ITEMS),
        "covered_in_repo": covered,
        "coverage_fraction": round(len(covered) / len(CHECKLIST_ITEMS), 3),
        "reference_doc": "retrieval-implementation-flow.md",
    }


def _item_covered(item: str, by_tag: dict, checklist_eval: dict) -> bool:
    mapping = {
        "ingestion": checklist_eval.get("overall_pass_rate", 0) >= 0.7,
        "fixed_chunking": "chunking" in by_tag,
        "recursive_chunking": "chunking" in by_tag,
        "semantic_chunking": "chunking" in by_tag,
        "dense_search": "dense" in by_tag,
        "hybrid_search": "hybrid" in by_tag,
        "metadata_filter": "metadata" in by_tag,
        "tenant_partition": "tenant" in by_tag,
        "regression_gates": checklist_eval.get("checklist_complete", False),
    }
    return mapping.get(item, False)