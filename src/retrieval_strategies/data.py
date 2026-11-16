from __future__ import annotations

import json
from pathlib import Path

from .models import CorpusDoc, RetrievalCase, RetrievalExpectations


def load_corpus(path: Path) -> list[CorpusDoc]:
    raw = json.loads(path.read_text())
    docs: list[CorpusDoc] = []
    for row in raw["documents"]:
        docs.append(
            CorpusDoc(
                doc_id=row["doc_id"],
                tenant_id=row.get("tenant_id", "default"),
                title=row.get("title", ""),
                body=row["body"],
                metadata=row.get("metadata", {}),
            )
        )
    return docs


def _load_expectations(row: dict) -> RetrievalExpectations:
    exp = row.get("expects", {})
    if not exp:
        return RetrievalExpectations()
    allowed = {f.name for f in RetrievalExpectations.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in exp.items() if k in allowed}
    return RetrievalExpectations(**filtered)


def load_golden(path: Path) -> list[RetrievalCase]:
    raw = json.loads(path.read_text())
    cases: list[RetrievalCase] = []
    for row in raw["cases"]:
        cases.append(
            RetrievalCase(
                id=row["id"],
                query=row["query"],
                tenant_id=row.get("tenant_id", "default"),
                top_k=int(row.get("top_k", 5)),
                metadata_filter=row.get("metadata_filter", {}),
                expects=_load_expectations(row),
                tags=row.get("tags", []),
                extra=row.get("extra", {}),
            )
        )
    return cases


def validate_golden(corpus: list[CorpusDoc], cases: list[RetrievalCase]) -> None:
    doc_ids = {d.doc_id for d in corpus}
    chunk_prefixes = {d.doc_id for d in corpus}
    for case in cases:
        for cid in case.expects.relevant_chunk_ids:
            doc_part = cid.split("::")[0]
            if doc_part not in chunk_prefixes and doc_part not in doc_ids:
                raise ValueError(f"case {case.id} references unknown chunk doc {doc_part}")