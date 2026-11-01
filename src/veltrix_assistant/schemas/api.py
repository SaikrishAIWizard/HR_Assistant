"""Validated HTTP request and response models."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    """Request body for the local retrieval test endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)
    reference_date: date | None = None


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
