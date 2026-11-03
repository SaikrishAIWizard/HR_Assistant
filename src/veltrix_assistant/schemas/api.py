"""Validated HTTP request and response models."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from veltrix_assistant.schemas.chat import SessionId
from veltrix_assistant.schemas.documents import DocumentType


class SearchRequest(BaseModel):
    """Request body for the local retrieval test endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)
    reference_date: date | None = None
    session_id: SessionId | None = None
    doc_type: DocumentType | None = None
    doc_ids: tuple[str, ...] | None = Field(default=None, max_length=50)
    policy_scope: Literal["GLOBAL", "REGIONAL"] | None = None
    authority: Literal["authoritative", "reference"] | None = None


class SearchResult(BaseModel):
    """A retrieved passage with source information for inspecting RAG results."""

    doc_id: str
    title: str
    locator: str
    text: str
    score: float


class SearchResponse(BaseModel):
    """Retrieval results for one request."""

    results: list[SearchResult]
    upcoming: list[SearchResult] = Field(default_factory=list)
    reference_date: date
