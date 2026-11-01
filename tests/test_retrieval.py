import logging

import pytest
import tiktoken

from veltrix_assistant.ingestion.schemas import DocumentChunk, DocumentMetadata
from veltrix_assistant.retrieval.bm25 import BM25Retriever
from veltrix_assistant.retrieval.context import (
    MAX_CORPUS_TOKENS,
    build_corpus_context,
    log_llm_corpus_tokens,
)


def _chunk(doc_id: str, text: str, access_level: str = "all") -> DocumentChunk:
    metadata = DocumentMetadata(
        doc_id=doc_id,
        file=f"{doc_id}.md",
        title=doc_id,
        doc_type="policy",
        regions=("GLOBAL",),
        effective_date="2026-01-01",
        access_level=access_level,
    )
    return DocumentChunk(
        chunk_id=doc_id,
        document=metadata,
        page_number=1,
        locator="p. 1",
        text=text,
    )


def test_retrieval_applies_access_filter_before_results() -> None:
    retriever = BM25Retriever(
        [
            _chunk("public", "The annual leave policy applies."),
            _chunk("restricted", "The annual leave exception is confidential.", "hr"),
        ]
    )

    results = retriever.search(
        "annual leave",
        is_allowed=lambda metadata: metadata.access_level == "all",
        limit=10,
    )

    assert results
    assert all(result.chunk.document.doc_id == "public" for result in results)


def test_retrieval_is_deterministic_for_equal_scores() -> None:
    retriever = BM25Retriever(
        [_chunk("second", "leave policy"), _chunk("first", "leave policy")]
    )
    results = retriever.search("leave", is_allowed=lambda _: True, limit=2)
    assert [result.chunk.chunk_id for result in results] == ["first", "second"]


def test_context_never_exceeds_hard_token_ceiling() -> None:
    chunks = [_chunk("first", "annual leave " * 1200), _chunk("second", "holiday dates")]
    context = build_corpus_context(chunks)
    encoding = tiktoken.get_encoding("cl100k_base")

    assert context.corpus_tokens <= MAX_CORPUS_TOKENS
    assert context.corpus_tokens == len(encoding.encode(context.text))
    assert "[first | p. 1]" in context.formatted_text()


def test_context_rejects_budget_above_assessment_limit() -> None:
    with pytest.raises(ValueError, match="between 0 and 3000"):
        build_corpus_context([], max_tokens=MAX_CORPUS_TOKENS + 1)


def test_token_count_is_logged_for_llm_call(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="veltrix_assistant.llm"):
        log_llm_corpus_tokens(17)
    assert "llm_call corpus_tokens=17" in caplog.text
