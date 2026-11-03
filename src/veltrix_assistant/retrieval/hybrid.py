"""Authorization-first hybrid retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from veltrix_assistant.policy.applicability import get_superseded_document_ids
from veltrix_assistant.retrieval.bm25 import BM25Retriever
from veltrix_assistant.retrieval.filters import metadata_matches
from veltrix_assistant.schemas.documents import (
    DocumentChunk,
    DocumentMetadata,
    RetrievedChunk,
)


_LOGGER = logging.getLogger("veltrix_assistant.retrieval.hybrid")
_MIN_VECTOR_SIMILARITY = 0.1


class VectorSearchStore(Protocol):
    """Vector index boundary consumed by hybrid retrieval."""

    def search(
        self,
        query: str,
        *,
        allowed_doc_ids: Sequence[str],
        limit: int,
        metadata_filter: Mapping[str, str | int] | None = None,
    ) -> list[tuple[str, float]]: ...


class HybridRetriever:
    """Fuse BM25 and persistent Chroma rankings without weakening access filters."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        vector_store: VectorSearchStore,
        bm25_top_k: int = 10,
        vector_top_k: int = 10,
        hybrid_top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        for value, name in (
            (bm25_top_k, "bm25_top_k"),
            (vector_top_k, "vector_top_k"),
            (hybrid_top_k, "hybrid_top_k"),
            (rrf_k, "rrf_k"),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        self._chunks = tuple(chunks)
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._bm25 = BM25Retriever(chunks)
        self._vector_store = vector_store
        self._bm25_top_k = bm25_top_k
        self._vector_top_k = vector_top_k
        self._hybrid_top_k = hybrid_top_k
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        is_allowed: Callable[[DocumentMetadata], bool],
        limit: int | None = None,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """Filter first, retrieve from both indexes, then fuse unique chunk IDs."""
        if not query.strip():
            return []
        requested_limit = self._hybrid_top_k if limit is None else limit
        if requested_limit < 1:
            raise ValueError("limit must be positive")

        visible_chunks = [
            chunk
            for chunk in self._chunks
            if is_allowed(chunk.document)
            and metadata_matches(chunk.document, metadata_filter)
        ]
        visible_documents = tuple(
            {
                chunk.document.doc_id: chunk.document
                for chunk in visible_chunks
            }.values()
        )
        superseded_ids = get_superseded_document_ids(visible_documents)
        visible_chunks = [
            chunk
            for chunk in visible_chunks
            if chunk.document.doc_id not in superseded_ids
        ]
        if not visible_chunks:
            return []

        allowed_chunk_ids = {chunk.chunk_id for chunk in visible_chunks}
        allowed_doc_ids = sorted(
            {chunk.document.doc_id for chunk in visible_chunks}
        )

        bm25_started = time.perf_counter()
        bm25_results = self._bm25.search(
            query,
            is_allowed=is_allowed,
            limit=max(self._bm25_top_k, requested_limit),
            metadata_filter=metadata_filter,
        )
        bm25_duration = time.perf_counter() - bm25_started

        vector_started = time.perf_counter()
        vector_results = self._vector_store.search(
            query,
            allowed_doc_ids=allowed_doc_ids,
            limit=max(self._vector_top_k, requested_limit),
            metadata_filter=_vector_metadata_filter(metadata_filter),
        )
        vector_duration = time.perf_counter() - vector_started
        vector_results = [
            (chunk_id, score)
            for chunk_id, score in vector_results
            if chunk_id in allowed_chunk_ids and score >= _MIN_VECTOR_SIMILARITY
        ]

        fused: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}
        bm25_scores: dict[str, float] = {}
        vector_ranks: dict[str, int] = {}
        vector_scores: dict[str, float] = {}
        for rank, match in enumerate(bm25_results, start=1):
            chunk_id = match.chunk.chunk_id
            bm25_ranks[chunk_id] = rank
            bm25_scores[chunk_id] = match.score
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (
                self._rrf_k + rank
            )
        for rank, (chunk_id, score) in enumerate(vector_results, start=1):
            vector_ranks[chunk_id] = rank
            vector_scores[chunk_id] = score
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (
                self._rrf_k + rank
            )

        ordered_ids = sorted(
            fused,
            key=lambda chunk_id: (-fused[chunk_id], chunk_id),
        )
        output = [
            RetrievedChunk(
                chunk=self._chunk_by_id[chunk_id],
                score=fused[chunk_id],
                retrieval_method=(
                    "hybrid"
                    if chunk_id in bm25_ranks and chunk_id in vector_ranks
                    else ("bm25" if chunk_id in bm25_ranks else "vector")
                ),
                bm25_rank=bm25_ranks.get(chunk_id),
                vector_rank=vector_ranks.get(chunk_id),
                bm25_score=bm25_scores.get(chunk_id),
                vector_score=vector_scores.get(chunk_id),
            )
            for chunk_id in ordered_ids[:requested_limit]
        ]

        _LOGGER.info(
            "hybrid_retrieval bm25=%d vector=%d fused=%d bm25_ms=%d vector_ms=%d",
            len(bm25_results),
            len(vector_results),
            len(output),
            int(bm25_duration * 1000),
            int(vector_duration * 1000),
        )
        return output


def _vector_metadata_filter(
    filters: Mapping[str, object] | None,
) -> dict[str, str | int] | None:
    if not filters:
        return None
    authority = filters.get("authority")
    mapped: dict[str, str | int] = {}
    doc_type = filters.get("doc_type")
    if isinstance(doc_type, str):
        mapped["doc_type"] = doc_type
    policy_scope = filters.get("policy_scope")
    if isinstance(policy_scope, str):
        mapped["policy_scope"] = policy_scope
    if authority == "reference":
        mapped["authority"] = 0
    elif authority == "authoritative":
        mapped["authority"] = 1
    return mapped or None
