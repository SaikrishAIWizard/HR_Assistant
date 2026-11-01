"""Validated retrieval and model-context contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from veltrix_assistant.schemas.documents import Citation


MAX_CORPUS_TOKENS = 3000


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
