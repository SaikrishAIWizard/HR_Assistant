"""Validated document and ingestion data structures."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DocumentType = Literal["policy", "addendum", "memo", "handbook", "calendar", "faq", "wiki"]
AccessLevel = Literal["all", "managers", "hr"]


class DocumentMetadata(BaseModel):
    """Manifest metadata attached to every ingested document and chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1)
    file: str = Field(min_length=1)
    title: str = Field(min_length=1)
    doc_type: DocumentType
    regions: tuple[str, ...] = Field(min_length=1)
    effective_date: date
    access_level: AccessLevel

    @field_validator("doc_id", "title", "file")
    @classmethod
    def values_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class ExtractedSection(BaseModel):
    """Text extracted from a source page or logical section."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    locator: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DocumentChunk(BaseModel):
    """A token-bounded text chunk with source metadata for retrieval and citations."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document: DocumentMetadata
    page_number: int = Field(ge=1)
    locator: str = Field(min_length=1)
    text: str = Field(min_length=1)


class IngestedCorpus(BaseModel):
    """Immutable output of a complete corpus ingestion run."""

    model_config = ConfigDict(frozen=True)

    documents: tuple[DocumentMetadata, ...]
    chunks: tuple[DocumentChunk, ...]


class RetrievedChunk(BaseModel):
    """A ranked chunk returned by local retrieval."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float


class Citation(BaseModel):
    """Citation fields exposed by the assistant response contract."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    locator: str
