"""Corpus-only token budgeting and per-call token-count logging."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

from veltrix_assistant.ingestion.schemas import Citation, DocumentChunk


MAX_CORPUS_TOKENS = 3000
_ENCODING_NAME = "cl100k_base"
_LOGGER = logging.getLogger("veltrix_assistant.llm")


class CorpusContext(BaseModel):
    """Corpus text and source metadata selected for one model call."""

    model_config = ConfigDict(frozen=True)

    text: str
    corpus_tokens: int = Field(ge=0, le=MAX_CORPUS_TOKENS)
    citations: tuple[Citation, ...]

    def formatted_text(self) -> str:
        """Render source labels alongside the corpus text for a model prompt."""
        blocks = self.text.split("\n\n")
        labels = [
            f"[{citation.doc_id} | {citation.locator}]"
            for citation in self.citations
        ]
        if len(blocks) != len(labels):
            return self.text
        return "\n\n".join(
            f"{label}\n{block}" for label, block in zip(labels, blocks, strict=True)
        )


def build_corpus_context(
    chunks: Sequence[DocumentChunk],
    *,
    max_tokens: int = MAX_CORPUS_TOKENS,
) -> CorpusContext:
    """Select whole chunks while guaranteeing no more than 3,000 corpus tokens."""
    if not 0 <= max_tokens <= MAX_CORPUS_TOKENS:
        raise ValueError(f"max_tokens must be between 0 and {MAX_CORPUS_TOKENS}")

    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    selected_text: list[str] = []
    citations: list[Citation] = []

    for chunk in chunks:
        candidate_text = "\n\n".join((*selected_text, chunk.text))
        token_count = len(encoding.encode(candidate_text))
        if token_count > max_tokens:
            continue
        selected_text.append(chunk.text)
        citations.append(Citation(doc_id=chunk.document.doc_id, locator=chunk.locator))

    corpus_text = "\n\n".join(selected_text)
    return CorpusContext(
        text=corpus_text,
        corpus_tokens=len(encoding.encode(corpus_text)),
        citations=tuple(citations),
    )


def log_llm_corpus_tokens(
    corpus_tokens: int,
    *,
    logger: logging.Logger = _LOGGER,
) -> None:
    """Record the corpus-token count immediately before an LLM call."""
    if not 0 <= corpus_tokens <= MAX_CORPUS_TOKENS:
        raise ValueError(f"corpus_tokens must be between 0 and {MAX_CORPUS_TOKENS}")
    logger.info("llm_call corpus_tokens=%d", corpus_tokens)
