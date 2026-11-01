"""FastAPI endpoints for readiness and access-filtered retrieval testing."""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from veltrix_assistant.ingestion.pipeline import ingest_corpus
from veltrix_assistant.policy.applicability import (
    DocumentAudience,
    Role,
    get_upcoming_winners,
    is_document_applicable,
    is_document_readable,
    parse_role,
)
from veltrix_assistant.retrieval.bm25 import BM25Retriever
from veltrix_assistant.tools.leave import LeaveToolFactory


_LOGGER = logging.getLogger("veltrix_assistant.api")


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


def _read_reference_date() -> date:
    value = os.getenv("APP_TODAY")
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("APP_TODAY must use YYYY-MM-DD format") from exc


def _read_employee(database_path: Path, employee_id: str) -> tuple[str, Role] | None:
    try:
        with sqlite3.connect(database_path, timeout=5) as connection:
            row = connection.execute(
                "SELECT region, role FROM employees WHERE employee_id = ?",
                (employee_id,),
            ).fetchone()
    except sqlite3.Error:
        _LOGGER.exception("Could not read employee identity from the configured database")
        raise HTTPException(
            status_code=503,
            detail="Employee directory is temporarily unavailable",
        ) from None

    if row is None:
        return None
    region, role = row
    valid_role = parse_role(role)
    if valid_role is None or not isinstance(region, str):
        _LOGGER.error("Employee record has invalid access-control attributes")
        raise HTTPException(
            status_code=503,
            detail="Employee directory contains an invalid access record",
        )
    return region, valid_role


def create_app(
    *,
    corpus_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    database_path: str | Path | None = None,
) -> FastAPI:
    """Create the local testing API and asynchronously ingest its corpus."""
    load_dotenv(override=False)
    configured_corpus = Path(corpus_dir or os.getenv("CORPUS_DIR", "./data/corpus"))
    configured_manifest = Path(
        manifest_path or os.getenv("MANIFEST_PATH", "./data/manifest.json")
    )
    configured_database = Path(database_path or os.getenv("DB_PATH", "./data/veltrix.db"))
    today = _read_reference_date()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.readiness = "loading"
        application.state.retriever = None
        ingestion_task = asyncio.create_task(_load_corpus(application))
        try:
            yield
        finally:
            await ingestion_task

    async def _load_corpus(application: FastAPI) -> None:
        try:
            corpus = await asyncio.to_thread(
                ingest_corpus,
                configured_corpus,
                configured_manifest,
            )
            application.state.retriever = BM25Retriever(corpus.chunks)
            application.state.leave_tool_factory = LeaveToolFactory(
                configured_database,
                corpus.chunks,
                today=today,
            )
            application.state.readiness = "ready"
            _LOGGER.info(
                "ingestion_ready documents=%d chunks=%d",
                len(corpus.documents),
                len(corpus.chunks),
            )
        except Exception:
            application.state.readiness = "failed"
            _LOGGER.exception("Corpus ingestion failed")

    application = FastAPI(
        title="Veltrix PeopleOps Retrieval Test API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    async def health(request: Request, response: Response) -> dict[str, str]:
        if getattr(request.app.state, "readiness", "loading") != "ready":
            response.status_code = 503
            return {"status": "unavailable"}
        return {"status": "ok"}

    @application.post("/search", response_model=SearchResponse)
    async def search(
        search_request: SearchRequest,
        request: Request,
        x_employee_id: Annotated[str | None, Header(alias="X-Employee-Id")] = None,
    ) -> SearchResponse:
        if not x_employee_id:
            raise HTTPException(status_code=401, detail="A valid employee ID is required")
        if getattr(request.app.state, "readiness", "loading") != "ready":
            raise HTTPException(status_code=503, detail="Document ingestion is not ready")

        identity = _read_employee(configured_database, x_employee_id)
        if identity is None:
            raise HTTPException(status_code=401, detail="A valid employee ID is required")
        region, role = identity
        retriever: BM25Retriever = request.app.state.retriever
        reference_date = search_request.reference_date or today
        audience = DocumentAudience(region=region, role=role)
        candidate_limit = max(search_request.limit * 10, 50)
        applicable_candidates = retriever.search(
            search_request.query,
            is_allowed=lambda metadata: is_document_applicable(
                metadata,
                audience,
                reference_date,
            ),
            limit=candidate_limit,
        )
        upcoming_candidates = retriever.search(
            search_request.query,
            is_allowed=lambda metadata: (
                is_document_readable(metadata, audience)
                and metadata.effective_date > reference_date
            ),
            limit=candidate_limit,
        )
        relevant_metadata = tuple(
            dict.fromkeys(
                match.chunk.document
                for match in (*applicable_candidates, *upcoming_candidates)
            )
        )
        upcoming_ids = {
            document.doc_id
            for document in get_upcoming_winners(
                relevant_metadata,
                audience,
                reference_date,
            )
        }
        applicable_matches = applicable_candidates[: search_request.limit]
        upcoming_matches = [
            match
            for match in upcoming_candidates
            if match.chunk.document.doc_id in upcoming_ids
        ][: search_request.limit]
        return SearchResponse(
            results=[
                SearchResult(
                    doc_id=match.chunk.document.doc_id,
                    title=match.chunk.document.title,
                    locator=match.chunk.locator,
                    text=match.chunk.text,
                    score=match.score,
                )
                for match in applicable_matches
            ],
            upcoming=[
                SearchResult(
                    doc_id=match.chunk.document.doc_id,
                    title=match.chunk.document.title,
                    locator=match.chunk.locator,
                    text=match.chunk.text,
                    score=match.score,
                )
                for match in upcoming_matches
            ],
            reference_date=reference_date,
        )

    return application


def run() -> None:
    """Run the local API server."""
    load_dotenv(override=False)
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


app = create_app()
