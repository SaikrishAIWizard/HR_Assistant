"""Validated environment configuration for optional retrieval and privacy features."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _read_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class AppSettings:
    """Application configuration validated once when the app is created."""

    retrieval_mode: str
    vector_store: str
    vector_db_path: Path
    embedding_provider: str
    embedding_model: str
    embedding_api_key: str | None
    embedding_timeout_seconds: float
    bm25_top_k: int
    vector_top_k: int
    hybrid_top_k: int
    rrf_k: int
    pii_redaction_enabled: bool
    pii_redaction_mode: Literal["redact", "mask"]
    pii_sensitive_terms: tuple[str, ...]
    provenance_enabled: bool
    provenance_version: int
    provenance_pseudonym_key: str | None
    langgraph_enabled: bool

    @classmethod
    def from_env(cls, *, data_dir: Path) -> AppSettings:
        mode = os.getenv("RETRIEVAL_MODE", "hybrid").strip().casefold()
        if not _read_bool("HYBRID_RETRIEVAL_ENABLED", True):
            mode = "bm25"
        if mode not in {"hybrid", "bm25"}:
            raise ValueError("RETRIEVAL_MODE must be 'hybrid' or 'bm25'")

        vector_store = os.getenv("VECTOR_STORE", "chroma").strip().casefold()
        if vector_store != "chroma":
            raise ValueError("VECTOR_STORE must be 'chroma'")

        embedding_provider = os.getenv(
            "EMBEDDING_PROVIDER",
            "local_hashing",
        ).strip().casefold()
        if embedding_provider not in {
            "local_hashing",
            "sentence_transformers",
            "openai",
        }:
            raise ValueError(
                "EMBEDDING_PROVIDER must be 'local_hashing', "
                "'sentence_transformers', or 'openai'"
            )

        redaction_mode_value = os.getenv(
            "PII_REDACTION_MODE",
            "redact",
        ).strip().casefold()
        if redaction_mode_value not in {"redact", "mask"}:
            raise ValueError("PII_REDACTION_MODE must be 'redact' or 'mask'")
        redaction_mode: Literal["redact", "mask"] = (
            "mask" if redaction_mode_value == "mask" else "redact"
        )

        try:
            embedding_timeout = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "20"))
        except ValueError as exc:
            raise ValueError("EMBEDDING_TIMEOUT_SECONDS must be numeric") from exc
        if embedding_timeout <= 0:
            raise ValueError("EMBEDDING_TIMEOUT_SECONDS must be greater than zero")

        sensitive_terms = tuple(
            term.strip()
            for term in os.getenv("PII_SENSITIVE_TERMS", "").split(",")
            if term.strip()
        )
        return cls(
            retrieval_mode=mode,
            vector_store=vector_store,
            vector_db_path=Path(
                os.getenv("VECTOR_DB_PATH", str(data_dir / "vector_db"))
            ),
            embedding_provider=embedding_provider,
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "local-hashing-384"
                if embedding_provider == "local_hashing"
                else (
                    "text-embedding-3-small"
                    if embedding_provider == "openai"
                    else "./models/sentence-transformer"
                ),
            ).strip(),
            embedding_api_key=(
                os.getenv("EMBEDDING_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or None
            ),
            embedding_timeout_seconds=embedding_timeout,
            bm25_top_k=_read_int("BM25_TOP_K", 10),
            vector_top_k=_read_int("VECTOR_TOP_K", 10),
            hybrid_top_k=_read_int("HYBRID_TOP_K", 10),
            rrf_k=_read_int("RRF_K", 60),
            pii_redaction_enabled=_read_bool("PII_REDACTION_ENABLED", True),
            pii_redaction_mode=redaction_mode,
            pii_sensitive_terms=sensitive_terms,
            provenance_enabled=_read_bool("PROVENANCE_ENABLED", True),
            provenance_version=_read_int("PROVENANCE_VERSION", 2),
            provenance_pseudonym_key=os.getenv("PROVENANCE_PSEUDONYM_KEY") or None,
            langgraph_enabled=_read_bool("LANGGRAPH_ENABLED", True),
        )
