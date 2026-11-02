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
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from veltrix_assistant.conversation import respond_to_message
from veltrix_assistant.llm import LLMError, create_llm_provider
from veltrix_assistant.ingestion.pipeline import ingest_corpus
from veltrix_assistant.policy.applicability import (
    get_upcoming_winners,
    is_document_applicable,
    is_document_readable,
)
from veltrix_assistant.retrieval.bm25 import BM25Retriever
from veltrix_assistant.schemas.api import SearchRequest, SearchResponse, SearchResult
from veltrix_assistant.schemas.chat import ChatRequest, ChatResponse
from veltrix_assistant.schemas.policy import DocumentAudience
from veltrix_assistant.schemas.security import EmployeeIdentity
from veltrix_assistant.security.identity import (
    IdentityDirectoryError,
    SessionBindingStore,
    lookup_employee_identity,
)
from veltrix_assistant.tools.leave import LeaveToolFactory


_LOGGER = logging.getLogger("veltrix_assistant.api")


def _read_reference_date() -> date:
    value = os.getenv("APP_TODAY")
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("APP_TODAY must use YYYY-MM-DD format") from exc


def _read_employee(database_path: Path, employee_id: str) -> EmployeeIdentity | None:
    try:
        return lookup_employee_identity(database_path, employee_id)
    except IdentityDirectoryError:
        _LOGGER.exception("Could not read employee identity from the configured database")
        raise HTTPException(
            status_code=503,
            detail="Employee directory is temporarily unavailable",
        ) from None


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
    llm_provider = create_llm_provider(
        os.getenv("LLM_PROVIDER", "none"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
    )

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
            session_store = SessionBindingStore(configured_database)
            session_store.ensure_schema()
            application.state.retriever = BM25Retriever(corpus.chunks)
            application.state.session_binding_store = session_store
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
        title="Veltrix PeopleOps Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> Response:
        if request.url.path == "/chat":
            return JSONResponse(
                status_code=400,
                content={"detail": jsonable_encoder(exc.errors())},
            )
        return await request_validation_exception_handler(request, exc)

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
        if search_request.session_id is not None:
            session_store: SessionBindingStore = request.app.state.session_binding_store
            try:
                session_belongs_to_employee = session_store.bind(
                    search_request.session_id,
                    identity.employee_id,
                )
            except IdentityDirectoryError:
                _LOGGER.exception("Could not verify session ownership")
                raise HTTPException(
                    status_code=503,
                    detail="Session storage is temporarily unavailable",
                ) from None
            if not session_belongs_to_employee:
                raise HTTPException(
                    status_code=403,
                    detail="This session belongs to another employee",
                )

        region, role = identity.region, identity.role
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

    @application.post("/chat", response_model=ChatResponse)
    async def chat(
        chat_request: ChatRequest,
        request: Request,
        x_employee_id: Annotated[str | None, Header(alias="X-Employee-Id")] = None,
    ) -> ChatResponse:
        if not x_employee_id:
            raise HTTPException(status_code=401, detail="A valid employee ID is required")
        if getattr(request.app.state, "readiness", "loading") != "ready":
            raise HTTPException(status_code=503, detail="Document ingestion is not ready")

        identity = _read_employee(configured_database, x_employee_id)
        if identity is None:
            raise HTTPException(status_code=401, detail="A valid employee ID is required")

        session_store: SessionBindingStore = request.app.state.session_binding_store
        try:
            if not session_store.bind(chat_request.session_id, identity.employee_id):
                raise HTTPException(
                    status_code=403,
                    detail="This session belongs to another employee",
                )
            leave_tool_factory: LeaveToolFactory = request.app.state.leave_tool_factory
            return await asyncio.to_thread(
                respond_to_message,
                message=chat_request.message,
                session_id=chat_request.session_id,
                identity=identity,
                retriever=request.app.state.retriever,
                tools=leave_tool_factory.for_employee(identity.employee_id),
                session_store=session_store,
                today=today,
                llm_provider=llm_provider,
            )
        except LLMError as exc:
            _LOGGER.error("Language model request failed: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="The language model is temporarily unavailable",
            ) from None
        except IdentityDirectoryError:
            _LOGGER.exception("Could not read or update chat session state")
            raise HTTPException(
                status_code=503,
                detail="Conversation state is temporarily unavailable",
            ) from None
        except sqlite3.Error:
            _LOGGER.exception("Could not check chat data access")
            raise HTTPException(
                status_code=503,
                detail="Leave data is temporarily unavailable",
            ) from None
        except ValueError:
            _LOGGER.exception("Stored chat session state is invalid")
            raise HTTPException(
                status_code=503,
                detail="Conversation state is temporarily unavailable",
            ) from None

    return application


def run() -> None:
    """Run the local API server."""
    load_dotenv(override=False)
    logging.basicConfig(level=logging.INFO)
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


app = create_app()
