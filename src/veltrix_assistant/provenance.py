"""Versioned, privacy-conscious provenance storage in assistant-owned SQLite tables."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veltrix_assistant.schemas.chat import ChatResponse


_FORBIDDEN_EVENT_KEYS = {
    "answer_text",
    "document_text",
    "email",
    "employee_id",
    "employee_name",
    "message",
    "phone",
    "prompt",
    "raw_prompt",
    "text",
    "tool_output",
}


class ProvenanceError(RuntimeError):
    """Raised when an audit record cannot be safely persisted."""


class ProvenanceTrace:
    """Per-request ordered events with no raw prompts or document text."""

    def __init__(
        self,
        *,
        retrieval_mode: str,
        llm_provider: str,
        llm_model: str,
    ) -> None:
        self.trace_id = str(uuid.uuid4())
        self.graph_execution_id = str(uuid.uuid4())
        self.request_id = str(uuid.uuid4())
        self.retrieval_mode = retrieval_mode
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self._events: list[dict[str, object]] = []
        self._lock = threading.Lock()

    @property
    def events(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(self._events)

    def record(
        self,
        node: str,
        event: str,
        details: dict[str, object] | None = None,
    ) -> None:
        safe_details = details or {}
        if _contains_forbidden_key(safe_details):
            raise ProvenanceError("Sensitive fields are not permitted in provenance events.")
        with self._lock:
            self._events.append(
                {
                    "sequence": len(self._events) + 1,
                    "node": node,
                    "event": event,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "details": safe_details,
                }
            )

    def set_request_id(self, request_id: str) -> None:
        self.request_id = request_id


class ProvenanceStore:
    """Persist versioned event traces, pseudonymizing employee and session references."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        version: int = 2,
        pseudonym_key: str | None = None,
    ) -> None:
        if version < 1:
            raise ValueError("Provenance version must be positive.")
        self._database_path = Path(database_path)
        self._version = version
        self._configured_key = pseudonym_key.encode("utf-8") if pseudonym_key else None

    def ensure_schema(self) -> None:
        if not self._database_path.is_file():
            raise ProvenanceError("Employee database does not exist.")
        try:
            with sqlite3.connect(self._database_path, timeout=5) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS assistant_provenance_keys (
                        key_name TEXT PRIMARY KEY,
                        key_value BLOB NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO assistant_provenance_keys (key_name, key_value)
                    VALUES ('pseudonym_key', ?)
                    """,
                    (os.urandom(32),),
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS assistant_provenance_v2 (
                        trace_id TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL,
                        request_id TEXT NOT NULL,
                        conversation_ref TEXT NOT NULL,
                        employee_ref TEXT NOT NULL,
                        graph_execution_id TEXT NOT NULL,
                        retrieval_mode TEXT NOT NULL,
                        llm_provider TEXT NOT NULL,
                        llm_model TEXT NOT NULL,
                        token_usage_json TEXT,
                        citation_validation TEXT NOT NULL,
                        response_route TEXT NOT NULL,
                        response_status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        completed_at TEXT NOT NULL,
                        events_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_assistant_provenance_request "
                    "ON assistant_provenance_v2(request_id)"
                )
        except sqlite3.Error as exc:
            raise ProvenanceError("Could not initialize provenance storage.") from exc

    def save(
        self,
        trace: ProvenanceTrace,
        *,
        employee_id: str,
        session_id: str,
        response: ChatResponse | None,
        response_status: str,
        error_code: str | None = None,
    ) -> None:
        try:
            with sqlite3.connect(self._database_path, timeout=5) as connection:
                connection.execute("BEGIN IMMEDIATE")
                key = self._configured_key
                if key is None:
                    row = connection.execute(
                        "SELECT key_value FROM assistant_provenance_keys "
                        "WHERE key_name = 'pseudonym_key'"
                    ).fetchone()
                    if row is None:
                        raise ProvenanceError("Provenance pseudonym key is unavailable.")
                    key = bytes(row[0])

                now = datetime.now(timezone.utc).isoformat()
                events = list(trace.events)
                if error_code is not None:
                    trace.record(
                        "response",
                        "failed",
                        {"error_code": error_code},
                    )
                    events = list(trace.events)
                route = response.route if response is not None else "error"
                citation_validation = _citation_validation_status(events)
                token_usage = json.dumps(
                    _token_usage(events),
                    separators=(",", ":"),
                )
                connection.execute(
                    """
                    INSERT INTO assistant_provenance_v2 (
                        trace_id, schema_version, request_id, conversation_ref,
                        employee_ref, graph_execution_id, retrieval_mode,
                        llm_provider, llm_model, token_usage_json,
                        citation_validation, response_route, response_status,
                        created_at, completed_at, events_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace.trace_id,
                        self._version,
                        trace.request_id,
                        _pseudonym(key, "conversation", session_id),
                        _pseudonym(key, "employee", employee_id),
                        trace.graph_execution_id,
                        trace.retrieval_mode,
                        trace.llm_provider,
                        trace.llm_model,
                        token_usage,
                        citation_validation,
                        route,
                        response_status,
                        now,
                        now,
                        json.dumps(events, separators=(",", ":"), ensure_ascii=False),
                    ),
                )
        except sqlite3.Error as exc:
            raise ProvenanceError("Could not persist the provenance trace.") from exc


def _pseudonym(key: bytes, purpose: str, value: str) -> str:
    digest = hmac.new(
        key,
        f"{purpose}\0{value}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key.casefold() in _FORBIDDEN_EVENT_KEYS
            or _contains_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _last_corpus_token_count(events: list[dict[str, object]]) -> int | None:
    for event in reversed(events):
        details = event.get("details")
        if isinstance(details, dict) and isinstance(details.get("corpus_tokens"), int):
            return details["corpus_tokens"]
    return None


def _citation_validation_status(events: list[dict[str, object]]) -> str:
    for event in reversed(events):
        if (
            event.get("node") == "citation_validation_node"
            and event.get("event") == "validated"
        ):
            details = event.get("details")
            if isinstance(details, dict):
                decision = details.get("decision")
                if decision in {"passed", "failed"}:
                    return str(decision)
    return "not_run"


def _token_usage(events: list[dict[str, object]]) -> dict[str, int]:
    token_usage: dict[str, int] = {}
    corpus_tokens = _last_corpus_token_count(events)
    if corpus_tokens is not None:
        token_usage["corpus_tokens"] = corpus_tokens
    for event in reversed(events):
        details = event.get("details")
        usage = details.get("token_usage") if isinstance(details, dict) else None
        if isinstance(usage, dict):
            token_usage.update(
                {
                    key: value
                    for key, value in usage.items()
                    if isinstance(key, str)
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                }
            )
            break
    return token_usage
