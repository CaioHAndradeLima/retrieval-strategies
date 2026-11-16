from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, defaultdict

from .models import (
    Chunk,
    ChunkingName,
    CorpusDoc,
    RetrievalCase,
    RetrievalExpectations,
    RetrievalResult,
    SearchName,
    StorageName,
    StrategyName,
    STRATEGY_COMPONENTS,
)

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _stable_hash_vec(text: str, dim: int = 16) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [(digest[i % len(digest)] / 255.0) * 2 - 1 for i in range(dim)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def chunk_document(
    doc: CorpusDoc,
    *,
    chunking: ChunkingName,
    fixed_size: int = 120,
    overlap: int = 20,
) -> list[Chunk]:
    body = doc.body.strip()
    if not body:
        return []

    if chunking == "fixed":
        segments: list[str] = []
        start = 0
        while start < len(body):
            end = min(len(body), start + fixed_size)
            segments.append(body[start:end])
            if end >= len(body):
                break
            start = max(0, end - overlap)
    elif chunking == "recursive":
        segments = []
        for block in re.split(r"\n\n+", body):
            block = block.strip()
            if not block:
                continue
            if len(block) <= fixed_size * 2:
                segments.append(block)
            else:
                for para in re.split(r"(?<=[.!?])\s+", block):
                    if para.strip():
                        segments.append(para.strip())
    else:  # semantic heuristic — break when adjacent sentence vectors diverge
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
        if not sentences:
            return []
        segments = []
        buf = sentences[0]
        prev_vec = _stable_hash_vec(sentences[0])
        for sent in sentences[1:]:
            vec = _stable_hash_vec(sent)
            sim = _cosine(prev_vec, vec)
            if sim < 0.35 or len(buf) + len(sent) > fixed_size * 3:
                segments.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}"
            prev_vec = vec
        if buf:
            segments.append(buf)
    chunks: list[Chunk] = []
    for idx, text in enumerate(segments):
        chunk_id = f"{doc.doc_id}::c{idx}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                tenant_id=doc.tenant_id,
                text=text,
                metadata=dict(doc.metadata),
            )
        )
    return chunks


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], avg_dl: float, k1: float = 1.2, b: float = 0.75) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in tf:
            continue
        freq = tf[term]
        idf = math.log(1 + (1.0 / (0.5 + freq / max(1, len(doc_tokens)))))
        denom = freq + k1 * (1 - b + b * dl / max(1.0, avg_dl))
        score += idf * (freq * (k1 + 1)) / max(1e-9, denom)
    return score


def _index_chunks(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


def _rank_dense(query: str, pool: list[Chunk]) -> list[tuple[str, float]]:
    qvec = _stable_hash_vec(query)
    out: list[tuple[str, float]] = []
    for ch in pool:
        score = _cosine(qvec, _stable_hash_vec(ch.text))
        out.append((ch.chunk_id, score))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _rank_bm25(query: str, pool: list[Chunk]) -> list[tuple[str, float]]:
    q_tokens = _tokens(query)
    all_lens = [len(_tokens(c.text)) for c in pool] or [1]
    avg_dl = sum(all_lens) / len(all_lens)
    scored: list[tuple[str, float]] = []
    for ch in pool:
        scored.append((ch.chunk_id, _bm25_score(q_tokens, _tokens(ch.text), avg_dl)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _rrf_merge(dense: list[tuple[str, float]], sparse: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    ranks: dict[str, float] = defaultdict(float)
    for i, (cid, _) in enumerate(dense):
        ranks[cid] += 1.0 / (k + i + 1)
    for i, (cid, _) in enumerate(sparse):
        ranks[cid] += 1.0 / (k + i + 1)
    merged = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return merged


def _apply_metadata_filter(pool: list[Chunk], filt: dict) -> list[Chunk]:
    if not filt:
        return pool
    out: list[Chunk] = []
    for ch in pool:
        meta = {**ch.metadata, "tenant_id": ch.tenant_id, "doc_id": ch.doc_id}
        if all(str(meta.get(k)) == str(v) for k, v in filt.items()):
            out.append(ch)
    return out


def _search(
    case: RetrievalCase,
    pool: list[Chunk],
    *,
    search: SearchName,
    top_k: int,
) -> list[str]:
    if search == "metadata_filter":
        filtered = _apply_metadata_filter(pool, case.metadata_filter)
        ranked = _rank_bm25(case.query, filtered)
        if not ranked and filtered:
            ranked = [(c.chunk_id, 1.0) for c in filtered[:top_k]]
        return [cid for cid, _ in ranked[:top_k]]

    if search == "dense":
        ranked = _rank_dense(case.query, pool)
        return [cid for cid, _ in ranked[:top_k]]

    dense = _rank_dense(case.query, pool)
    sparse = _rank_bm25(case.query, pool)
    merged = _rrf_merge(dense, sparse)
    return [cid for cid, _ in merged[:top_k]]


def _score_result(case: RetrievalCase, result: RetrievalResult) -> dict[str, bool]:
    exp: RetrievalExpectations = case.expects
    checks: dict[str, bool] = {}
    checks["recall"] = result.recall_at_k >= exp.min_recall_at_k
    if exp.relevant_chunk_ids:
        hits = set(result.retrieved_ids) & set(exp.relevant_chunk_ids)
        checks["relevant_hit"] = len(hits) > 0
    if exp.require_metadata_match and case.metadata_filter:
        checks["metadata_filter"] = result.metadata_respected
    if exp.require_tenant_isolation:
        checks["tenant_isolation"] = result.tenant_isolated
    if exp.max_latency_ms is not None:
        checks["latency_budget"] = result.latency_ms <= exp.max_latency_ms
    if exp.min_chunks_indexed is not None:
        checks["chunks_indexed"] = result.chunks_indexed >= exp.min_chunks_indexed
    return checks


def build_chunks_for_corpus(
    docs: list[CorpusDoc],
    *,
    chunking: ChunkingName,
    storage: StorageName,
    tenant_scope: str | None = None,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        if storage == "tenant_partition" and tenant_scope and doc.tenant_id != tenant_scope:
            continue
        all_chunks.extend(chunk_document(doc, chunking=chunking))
    return all_chunks


def run_retrieval(
    case: RetrievalCase,
    docs: list[CorpusDoc],
    *,
    strategy: StrategyName,
) -> RetrievalResult:
    chunking, search, storage = STRATEGY_COMPONENTS[strategy]
    t0 = time.perf_counter()

    scope = case.tenant_id if storage == "tenant_partition" else None
    chunks = build_chunks_for_corpus(docs, chunking=chunking, storage=storage, tenant_scope=scope)
    index = _index_chunks(chunks)

    pool = list(index.values())
    if search != "metadata_filter" and case.metadata_filter:
        pool = _apply_metadata_filter(pool, case.metadata_filter)

    retrieved = _search(case, pool, search=search, top_k=case.top_k)
    latency_ms = (time.perf_counter() - t0) * 1000 + len(chunks) * 0.01 + len(case.query) * 0.02

    relevant = set(case.expects.relevant_chunk_ids)
    hits = set(retrieved) & relevant
    recall = len(hits) / max(1, len(relevant)) if relevant else (1.0 if retrieved else 0.0)

    metadata_respected = True
    if case.metadata_filter:
        for cid in retrieved:
            ch = index.get(cid)
            if not ch:
                metadata_respected = False
                break
            meta = {**ch.metadata, "tenant_id": ch.tenant_id, "doc_id": ch.doc_id}
            if not all(str(meta.get(k)) == str(v) for k, v in case.metadata_filter.items()):
                metadata_respected = False
                break

    tenant_isolated = True
    if storage == "tenant_partition":
        for cid in retrieved:
            ch = index.get(cid)
            if ch and ch.tenant_id != case.tenant_id:
                tenant_isolated = False
                break

    result = RetrievalResult(
        case_id=case.id,
        strategy=strategy,
        retrieved_ids=retrieved,
        recall_at_k=round(recall, 4),
        chunks_indexed=len(chunks),
        latency_ms=round(latency_ms, 3),
        chunking=chunking,
        search=search,
        storage=storage,
        metadata_respected=metadata_respected,
        tenant_isolated=tenant_isolated,
    )
    result.checks = _score_result(case, result)
    failed = [k for k, v in result.checks.items() if not v]
    result.reason = f"failed: {', '.join(failed)}" if failed else "ok"
    return result