"""Persistent Chroma vector index and CPU/local or API embedding providers."""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, cast

import chromadb
import numpy as np
from openai import OpenAI
from openai import OpenAIError
from chromadb.api.types import Where

from veltrix_assistant.schemas.documents import DocumentChunk
from veltrix_assistant.security.privacy import redact_pii


_LOGGER = logging.getLogger("veltrix_assistant.retrieval.vector")
_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_CHARACTER_NGRAMS = (2, 3, 4)


class EmbeddingProvider(Protocol):
    """A provider that returns normalized vectors for source text and queries."""

    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalHashingEmbeddings:
    """Deterministic CPU-only feature embeddings that require no model download."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 64:
            raise ValueError("Local hashing embeddings require at least 64 dimensions")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        terms = [term.casefold() for term in _TOKEN_PATTERN.findall(text)]
        features = list(terms)
        for term in terms:
            features.extend(
                term[index : index + size]
                for size in _CHARACTER_NGRAMS
                for index in range(max(0, len(term) - size + 1))
            )
        vector = [0.0] * self._dimensions
        for feature in features:
            digest = hashlib.blake2b(
                feature.encode("utf-8"),
                digest_size=8,
                person=b"veltrix",
            ).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class SentenceTransformerEmbeddings:
    """Use a pre-provisioned sentence-transformers model on CPU, without downloads."""

    def __init__(self, model_path: str) -> None:
        try:
            sentence_transformers = importlib.import_module(
                "sentence_transformers"
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install the local-embeddings extra to use sentence_transformers"
            ) from exc

        model_location = Path(model_path)
        if not model_location.exists():
            raise ValueError(
                "EMBEDDING_MODEL must point to a pre-provisioned local model "
                "when EMBEDDING_PROVIDER=sentence_transformers"
            )
        sentence_transformer = getattr(
            sentence_transformers,
            "SentenceTransformer",
        )
        self._model = sentence_transformer(
            str(model_location),
            device="cpu",
            local_files_only=True,
        )
        self._dimensions = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbeddingProvider:
    """Batch documents through the configured OpenAI-compatible embedding API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout_seconds: float,
    ) -> None:
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY or OPENAI_API_KEY is required")
        known_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        if model not in known_dimensions:
            raise ValueError(
                "Unsupported OpenAI embedding model; choose text-embedding-3-small, "
                "text-embedding-3-large, or text-embedding-ada-002"
            )
        self._client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self._model = model
        self._dimensions = known_dimensions[model]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), 64):
                response = self._client.embeddings.create(
                    model=self._model,
                    input=list(texts[start : start + 64]),
                )
                vectors.extend(
                    item.embedding
                    for item in sorted(response.data, key=lambda item: item.index)
                )
        except OpenAIError as exc:
            raise RuntimeError("The configured embedding API request failed") from exc
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embedding_provider(
    provider: str,
    *,
    model: str,
    api_key: str | None,
    timeout_seconds: float,
) -> EmbeddingProvider:
    normalized = provider.strip().casefold()
    if normalized == "local_hashing":
        match = re.fullmatch(r"local-hashing-(\d+)", model)
        if match is None:
            raise ValueError("Local hashing EMBEDDING_MODEL must be local-hashing-<dimensions>")
        return LocalHashingEmbeddings(int(match.group(1)))
    if normalized == "sentence_transformers":
        return SentenceTransformerEmbeddings(model)
    if normalized == "openai":
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _chunk_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
    scope = "GLOBAL" if "GLOBAL" in chunk.document.regions else "REGIONAL"
    return {
        "doc_id": chunk.document.doc_id,
        "title": chunk.document.title,
        "doc_type": chunk.document.doc_type,
        "regions": ",".join(chunk.document.regions),
        "effective_date": chunk.document.effective_date.isoformat(),
        "access_level": chunk.document.access_level,
        "policy_scope": scope,
        "authority": int(
            chunk.document.doc_type not in {"faq", "wiki"}
        ),
        "locator": chunk.locator,
        "page_number": chunk.page_number,
        "source_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
    }


class ChromaVectorStore:
    """Incrementally synchronize and query a local persistent Chroma collection."""

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "veltrix_chunks_v2",
        pii_redaction_enabled: bool = True,
        pii_redaction_mode: Literal["redact", "mask"] = "redact",
        pii_sensitive_terms: tuple[str, ...] = (),
    ) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_provider = embedding_provider
        self._pii_redaction_enabled = pii_redaction_enabled
        self._pii_redaction_mode: Literal["redact", "mask"] = pii_redaction_mode
        self._pii_sensitive_terms = pii_sensitive_terms

    def close(self) -> None:
        """Close the Chroma client and release its persistent database handles."""
        self._client.close()

    def _safe_embedding_text(self, text: str) -> str:
        if not self._pii_redaction_enabled:
            return text
        return redact_pii(
            text,
            mode=self._pii_redaction_mode,
            sensitive_terms=self._pii_sensitive_terms,
        )

    def sync(self, chunks: Sequence[DocumentChunk]) -> dict[str, int]:
        """Add new chunks, update changed metadata, and remove stale chunk IDs."""
        started = time.perf_counter()
        current_chunks = {chunk.chunk_id: chunk for chunk in chunks}
        existing = self._collection.get(include=["metadatas"])
        existing_ids = set(existing["ids"])
        expected_ids = set(current_chunks)
        stale_ids = sorted(existing_ids - expected_ids)

        existing_metadata = dict(
            zip(existing["ids"], existing.get("metadatas") or [], strict=True)
        )
        new_ids = sorted(expected_ids - existing_ids)
        changed_metadata = sorted(
            chunk_id
            for chunk_id in expected_ids & existing_ids
            if existing_metadata.get(chunk_id) != _chunk_metadata(current_chunks[chunk_id])
        )

        for start in range(0, len(new_ids), 128):
            batch_ids = new_ids[start : start + 128]
            batch_chunks = [current_chunks[chunk_id] for chunk_id in batch_ids]
            vectors = self._embedding_provider.embed_documents(
                [self._safe_embedding_text(chunk.text) for chunk in batch_chunks]
            )
            self._collection.add(
                ids=batch_ids,
                embeddings=np.asarray(vectors, dtype=np.float32),
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[_chunk_metadata(chunk) for chunk in batch_chunks],
            )

        for start in range(0, len(changed_metadata), 128):
            batch_ids = changed_metadata[start : start + 128]
            batch_chunks = [current_chunks[chunk_id] for chunk_id in batch_ids]
            vectors = self._embedding_provider.embed_documents(
                [self._safe_embedding_text(chunk.text) for chunk in batch_chunks]
            )
            self._collection.update(
                ids=batch_ids,
                embeddings=np.asarray(vectors, dtype=np.float32),
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[
                    _chunk_metadata(current_chunks[chunk_id])
                    for chunk_id in batch_ids
                ],
            )

        for start in range(0, len(stale_ids), 1000):
            self._collection.delete(ids=stale_ids[start : start + 1000])

        stats = {
            "added": len(new_ids),
            "metadata_updated": len(changed_metadata),
            "deleted": len(stale_ids),
            "total": len(expected_ids),
        }
        _LOGGER.info(
            "vector_index_synced added=%d metadata_updated=%d deleted=%d total=%d duration_ms=%d",
            stats["added"],
            stats["metadata_updated"],
            stats["deleted"],
            stats["total"],
            int((time.perf_counter() - started) * 1000),
        )
        return stats

    def search(
        self,
        query: str,
        *,
        allowed_doc_ids: Sequence[str],
        limit: int,
        metadata_filter: Mapping[str, str | int] | None = None,
    ) -> list[tuple[str, float]]:
        """Search only within authorized document IDs using Chroma prefilters."""
        if limit < 1:
            raise ValueError("limit must be positive")
        permitted_ids = sorted(set(allowed_doc_ids))
        if not permitted_ids:
            return []

        # Chroma accepts this $in filter, but its recursive Where alias omits it.
        where = cast(Where, {"doc_id": {"$in": permitted_ids}})
        if metadata_filter:
            conditions: list[Where] = [where]
            for key, value in metadata_filter.items():
                conditions.append({key: value})
            where = cast(Where, {"$and": conditions})

        vector = self._embedding_provider.embed_query(
            self._safe_embedding_text(query)
        )
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=limit,
            where=where,
            include=["distances"],
        )
        result_ids = result.get("ids")
        result_distances = result.get("distances")
        if result_ids is None or result_distances is None:
            raise RuntimeError("The Chroma query omitted IDs or distances.")
        if not result_ids or not result_distances:
            return []
        ids = result_ids[0]
        distances = result_distances[0]
        return [
            (chunk_id, 1.0 - float(distance))
            for chunk_id, distance in zip(ids, distances, strict=True)
        ]
