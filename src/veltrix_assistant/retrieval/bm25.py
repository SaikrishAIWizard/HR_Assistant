"""Deterministic local BM25 retrieval with pre-ranking metadata filtering."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

from veltrix_assistant.ingestion.schemas import DocumentChunk, DocumentMetadata, RetrievedChunk


_TOKEN_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


class BM25Retriever:
    """Small-corpus BM25 search that evaluates access filters before scoring."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self._chunks = tuple(chunks)
        self._k1 = k1
        self._b = b

    def search(
        self,
        query: str,
        *,
        is_allowed: Callable[[DocumentMetadata], bool],
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """Return ranked matches only from documents approved by ``is_allowed``."""
        if limit < 1:
            raise ValueError("limit must be positive")
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        visible_chunks = [
            chunk for chunk in self._chunks if is_allowed(chunk.document)
        ]
        if not visible_chunks:
            return []

        tokenized_documents = [_tokenize(chunk.text) for chunk in visible_chunks]
        document_frequencies: Counter[str] = Counter()
        for tokens in tokenized_documents:
            document_frequencies.update(set(tokens))

        document_count = len(visible_chunks)
        average_length = (
            sum(map(len, tokenized_documents)) / document_count if document_count else 0.0
        )
        scored: list[RetrievedChunk] = []

        for chunk, tokens in zip(visible_chunks, tokenized_documents, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for term in query_tokens:
                term_frequency = frequencies[term]
                if not term_frequency:
                    continue
                inverse_document_frequency = math.log(
                    1
                    + (document_count - document_frequencies[term] + 0.5)
                    / (document_frequencies[term] + 0.5)
                )
                denominator = term_frequency + self._k1 * (
                    1 - self._b
                    + self._b * len(tokens) / max(average_length, 1.0)
                )
                score += inverse_document_frequency * (
                    term_frequency * (self._k1 + 1) / denominator
                )
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score))

        return sorted(
            scored,
            key=lambda result: (-result.score, result.chunk.chunk_id),
        )[:limit]
