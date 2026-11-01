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

    blocks: tuple[tuple[Citation, str], ...]
    corpus_tokens: int = Field(ge=0, le=MAX_CORPUS_TOKENS)

    @property
    def text(self) -> str:
        return "\n\n".join(text for _, text in self.blocks)

    @property
    def citations(self) -> tuple[Citation, ...]:
        unique_citations = dict.fromkeys(citation for citation, _ in self.blocks)
        return tuple(unique_citations)

    def formatted_text(self) -> str:
        """Render source labels alongside the corpus text for a model prompt."""
        return "\n\n".join(
            f"[{citation.doc_id} | {citation.locator}]\n{text}"
            for citation, text in self.blocks
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
    selected_blocks: list[tuple[Citation, str]] = []

    for chunk in chunks:
        candidate_text = "\n\n".join(
            (text for _, text in selected_blocks)
        )
        candidate_text = "\n\n".join(
            part for part in (candidate_text, chunk.text) if part
        )
        token_count = len(encoding.encode(candidate_text))
        if token_count > max_tokens:
            continue
        selected_blocks.append(
            (
                Citation(doc_id=chunk.document.doc_id, locator=chunk.locator),
                chunk.text,
            )
        )

    corpus_text = "\n\n".join(text for _, text in selected_blocks)
    return CorpusContext(
        blocks=tuple(selected_blocks),
        corpus_tokens=len(encoding.encode(corpus_text)),
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
