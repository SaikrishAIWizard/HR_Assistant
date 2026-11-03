"""FastAPI endpoints for readiness and access-filtered retrieval testing."""

from __future__ import annotations

import asyncio
import hashlib
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

from veltrix_assistant.llm import LLMError, create_llm_provider
from veltrix_assistant.ingestion.pipeline import ingest_corpus
from veltrix_assistant.policy.applicability import (
    get_upcoming_winners,
    is_document_applicable,
    is_document_readable,
)
from veltrix_assistant.retrieval.bm25 import BM25Retriever
from veltrix_assistant.retrieval.hybrid import HybridRetriever
from veltrix_assistant.retrieval.vector_store import (
    ChromaVectorStore,
    create_embedding_provider,
)
from veltrix_assistant.provenance import (
    ProvenanceError,
    ProvenanceStore,
    ProvenanceTrace,
)
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
from veltrix_assistant.settings import AppSettings
from veltrix_assistant.workflow import ChatWorkflow, ChatWorkflowState


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
    settings = AppSettings.from_env(data_dir=configured_database.parent)
    llm_provider = create_llm_provider(
        os.getenv("LLM_PROVIDER", "none"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        pii_redaction_enabled=settings.pii_redaction_enabled,
        pii_redaction_mode=settings.pii_redaction_mode,
        pii_sensitive_terms=settings.pii_sensitive_terms,
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
            if settings.retrieval_mode == "hybrid":
                embedding_provider = create_embedding_provider(
                    settings.embedding_provider,
                    model=settings.embedding_model,
                    api_key=settings.embedding_api_key,
                    timeout_seconds=settings.embedding_timeout_seconds,
                )
                vector_store = ChromaVectorStore(
                    settings.vector_db_path,
                    embedding_provider=embedding_provider,
                    collection_name=(
                        f"veltrix_v2_{settings.embedding_provider}_"
                        f"{hashlib.sha256(settings.embedding_model.encode()).hexdigest()[:10]}"
                    ),
                    pii_redaction_enabled=settings.pii_redaction_enabled,
                    pii_redaction_mode=settings.pii_redaction_mode,
                    pii_sensitive_terms=settings.pii_sensitive_terms,
                )
                sync_stats = await asyncio.to_thread(vector_store.sync, corpus.chunks)
                retriever: BM25Retriever | HybridRetriever = HybridRetriever(
                    corpus.chunks,
                    vector_store=vector_store,
                    bm25_top_k=settings.bm25_top_k,
                    vector_top_k=settings.vector_top_k,
                    hybrid_top_k=settings.hybrid_top_k,
                    rrf_k=settings.rrf_k,
                )
                application.state.vector_store = vector_store
                application.state.vector_sync_stats = sync_stats
            else:
                retriever = BM25Retriever(corpus.chunks)
            session_store = SessionBindingStore(configured_database)
            session_store.ensure_schema()
            provenance_store = (
                ProvenanceStore(
                    configured_database,
                    version=settings.provenance_version,
                    pseudonym_key=settings.provenance_pseudonym_key,
                )
                if settings.provenance_enabled
                else None
            )
            if provenance_store is not None:
                provenance_store.ensure_schema()
            application.state.retriever = retriever
            application.state.session_binding_store = session_store
            application.state.provenance_store = provenance_store
            application.state.chat_workflow = ChatWorkflow(
                database_path=configured_database,
                enabled=settings.langgraph_enabled,
            )
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
        retriever: BM25Retriever | HybridRetriever = request.app.state.retriever
        reference_date = search_request.reference_date or today
        audience = DocumentAudience(region=region, role=role)
        metadata_filter: dict[str, object] = {
            "role": role,
            "region": region,
            "doc_type": search_request.doc_type,
            "doc_ids": search_request.doc_ids,
            "policy_scope": search_request.policy_scope,
            "authority": search_request.authority,
        }
        metadata_filter = {
            key: value
            for key, value in metadata_filter.items()
            if value is not None
        }
        applicable_filter = {
            **metadata_filter,
            "effective_on_or_before": reference_date,
        }
        candidate_limit = max(search_request.limit * 10, 50)
        applicable_candidates = retriever.search(
            search_request.query,
            is_allowed=lambda metadata: is_document_applicable(
                metadata,
                audience,
                reference_date,
            ),
            limit=candidate_limit,
            metadata_filter=applicable_filter,
        )
        upcoming_candidates = retriever.search(
            search_request.query,
            is_allowed=lambda metadata: (
                is_document_readable(metadata, audience)
                and metadata.effective_date > reference_date
            ),
            limit=candidate_limit,
            metadata_filter=metadata_filter,
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
            trace = ProvenanceTrace(
                retrieval_mode=settings.retrieval_mode,
                llm_provider=getattr(llm_provider, "provider_name", "none"),
                llm_model=getattr(llm_provider, "model_name", "none"),
            )
            workflow_state: ChatWorkflowState = {
                "message": chat_request.message,
                "session_id": chat_request.session_id,
                "employee_id": identity.employee_id,
                "identity": identity,
                "retriever": request.app.state.retriever,
                "tools": leave_tool_factory.for_employee(identity.employee_id),
                "session_store": session_store,
                "today": today,
                "llm_provider": llm_provider,
                "trace": trace,
                "pii_redaction_enabled": settings.pii_redaction_enabled,
                "pii_redaction_mode": settings.pii_redaction_mode,
                "pii_sensitive_terms": settings.pii_sensitive_terms,
            }

            async def persist_trace(
                response: ChatResponse | None,
                *,
                response_status: str,
                error_code: str | None = None,
            ) -> None:
                provenance_store: ProvenanceStore | None = (
                    request.app.state.provenance_store
                )
                if provenance_store is None:
                    return
                try:
                    await asyncio.to_thread(
                        provenance_store.save,
                        trace,
                        employee_id=identity.employee_id,
                        session_id=chat_request.session_id,
                        response=response,
                        response_status=response_status,
                        error_code=error_code,
                    )
                except ProvenanceError:
                    _LOGGER.exception("Could not persist chat provenance")
                    raise HTTPException(
                        status_code=503,
                        detail="Conversation audit storage is temporarily unavailable",
                    ) from None

            workflow: ChatWorkflow = request.app.state.chat_workflow
            try:
                chat_response = await asyncio.to_thread(workflow.invoke, workflow_state)
            except LLMError as exc:
                await persist_trace(
                    None,
                    response_status="error",
                    error_code=type(exc).__name__,
                )
                _LOGGER.error("Language model request failed: %s", exc)
                detail = "The language model is temporarily unavailable"
                if exc.error_code in {
                    "credit_balance_exhausted",
                    "insufficient_quota",
                }:
                    detail = (
                        "The configured language model account has no API "
                        "credits or quota. Add credits or configure a provider "
                        "account with available quota."
                    )
                raise HTTPException(
                    status_code=502,
                    detail=detail,
                ) from None
            except IdentityDirectoryError:
                await persist_trace(
                    None,
                    response_status="error",
                    error_code="IdentityDirectoryError",
                )
                _LOGGER.exception("Could not read or update chat session state")
                raise HTTPException(
                    status_code=503,
                    detail="Conversation state is temporarily unavailable",
                ) from None
            except sqlite3.Error:
                await persist_trace(
                    None,
                    response_status="error",
                    error_code="DatabaseError",
                )
                _LOGGER.exception("Could not check chat data access")
                raise HTTPException(
                    status_code=503,
                    detail="Leave data is temporarily unavailable",
                ) from None
            except ValueError:
                await persist_trace(
                    None,
                    response_status="error",
                    error_code="ValueError",
                )
                _LOGGER.exception("Stored chat session state is invalid")
                raise HTTPException(
                    status_code=503,
                    detail="Conversation state is temporarily unavailable",
                ) from None
            except Exception as exc:
                await persist_trace(
                    None,
                    response_status="error",
                    error_code=type(exc).__name__,
                )
                _LOGGER.error(
                    "Chat workflow failed error_code=%s",
                    type(exc).__name__,
                )
                raise

            await persist_trace(
                chat_response,
                response_status="success",
            )
            return chat_response
        except HTTPException:
            raise

    return application


def run() -> None:
    """Run the local API server."""
    load_dotenv(override=False)
    logging.basicConfig(level=logging.INFO)
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


app = create_app()
