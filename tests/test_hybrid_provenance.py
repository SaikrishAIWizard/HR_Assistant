import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import date

from veltrix_assistant.provenance import (
    ProvenanceError,
    ProvenanceStore,
    ProvenanceTrace,
)
from veltrix_assistant.retrieval.hybrid import HybridRetriever
from veltrix_assistant.retrieval.vector_store import (
    ChromaVectorStore,
    LocalHashingEmbeddings,
)
from veltrix_assistant.schemas.chat import ChatCitation, ChatResponse
from veltrix_assistant.schemas.documents import (
    AccessLevel,
    DocumentChunk,
    DocumentMetadata,
)


def _chunk(
    doc_id: str,
    text: str,
    *,
    access_level: AccessLevel = "all",
    effective_date: date = date(2026, 1, 1),
    supersedes_doc_ids: tuple[str, ...] = (),
) -> DocumentChunk:
    metadata = DocumentMetadata(
        doc_id=doc_id,
        file=f"{doc_id}.md",
        title=doc_id,
        doc_type="policy",
        regions=("GLOBAL",),
        effective_date=effective_date,
        access_level=access_level,
        supersedes_doc_ids=supersedes_doc_ids,
    )
    return DocumentChunk(
        chunk_id=doc_id,
        document=metadata,
        page_number=1,
        locator="Annual leave",
        text=text,
    )


def test_chroma_sync_adds_updates_and_deletes_chunks(tmp_path) -> None:
    store = ChromaVectorStore(
        tmp_path / "vectors",
        embedding_provider=LocalHashingEmbeddings(),
        collection_name="sync_test",
    )
    original = _chunk("public", "Annual leave is accrued each month.")

    assert store.sync([original]) == {
        "added": 1,
        "metadata_updated": 0,
        "deleted": 0,
        "total": 1,
    }
    assert store.sync([original]) == {
        "added": 0,
        "metadata_updated": 0,
        "deleted": 0,
        "total": 1,
    }

    changed = original.model_copy(
        update={"text": "Annual leave is accrued every month."}
    )
    assert store.sync([changed]) == {
        "added": 0,
        "metadata_updated": 1,
        "deleted": 0,
        "total": 1,
    }
    stored = store._collection.get(ids=["public"], include=["documents"])
    assert stored["documents"] == ["Annual leave is accrued every month."]
    reopened = ChromaVectorStore(
        tmp_path / "vectors",
        embedding_provider=LocalHashingEmbeddings(),
        collection_name="sync_test",
    )
    assert reopened.search(
        "leave every month",
        allowed_doc_ids=["public"],
        limit=1,
    )[0][0] == "public"

    assert store.sync([]) == {
        "added": 0,
        "metadata_updated": 0,
        "deleted": 1,
        "total": 0,
    }


def test_chroma_sanitizes_external_embedding_inputs(tmp_path) -> None:
    class CapturingEmbeddings(LocalHashingEmbeddings):
        def __init__(self) -> None:
            super().__init__()
            self.document_inputs: list[str] = []
            self.query_input = ""

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            self.document_inputs = list(texts)
            return super().embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            self.query_input = text
            return super().embed_query(text)

    embedding_provider = CapturingEmbeddings()
    store = ChromaVectorStore(
        tmp_path / "vectors",
        embedding_provider=embedding_provider,
        collection_name="pii_test",
        pii_redaction_enabled=True,
    )
    chunk = _chunk(
        "sensitive",
        "Contact vijay@example.test about annual leave for E1001.",
    )

    store.sync([chunk])
    store.search(
        "annual leave for E1001 at vijay@example.test",
        allowed_doc_ids=["sensitive"],
        limit=1,
    )

    assert "vijay@example.test" not in embedding_provider.document_inputs[0]
    assert "E1001" not in embedding_provider.document_inputs[0]
    assert "vijay@example.test" not in embedding_provider.query_input
    assert "E1001" not in embedding_provider.query_input


def test_hybrid_retrieval_fuses_candidates_after_authorization() -> None:
    public = _chunk("public", "Annual leave is carried forward each year.")
    restricted = _chunk(
        "restricted",
        "Annual leave exception for HR only.",
        access_level="hr",
    )

    class FakeVectorStore:
        def __init__(self) -> None:
            self.allowed_doc_ids: list[str] = []

        def search(
            self,
            query: str,
            *,
            allowed_doc_ids: Sequence[str],
            limit: int,
            metadata_filter: Mapping[str, str | int] | None = None,
        ) -> list[tuple[str, float]]:
            del query, limit, metadata_filter
            self.allowed_doc_ids = list(allowed_doc_ids)
            return [("public", 0.9), ("restricted", 1.0)]

    vector_store = FakeVectorStore()
    retriever = HybridRetriever(
        [public, restricted],
        vector_store=vector_store,
        bm25_top_k=5,
        vector_top_k=5,
        hybrid_top_k=5,
    )

    results = retriever.search(
        "annual leave",
        is_allowed=lambda metadata: metadata.access_level == "all",
        limit=5,
    )

    assert vector_store.allowed_doc_ids == ["public"]
    assert [item.chunk.chunk_id for item in results] == ["public"]
    assert results[0].retrieval_method == "hybrid"
    assert results[0].bm25_rank == 1
    assert results[0].vector_rank == 1


def test_hybrid_retrieval_excludes_superseded_documents_from_both_indexes() -> None:
    old = _chunk(
        "POLICY-V1",
        "Annual leave carry forward deadline is 30 June.",
        effective_date=date(2025, 1, 1),
    )
    current = _chunk(
        "POLICY-V2",
        "Annual leave carry forward deadline is 31 March.",
        effective_date=date(2026, 4, 1),
        supersedes_doc_ids=("POLICY-V1",),
    )

    class FakeVectorStore:
        def __init__(self) -> None:
            self.allowed_doc_ids: list[str] = []

        def search(
            self,
            query: str,
            *,
            allowed_doc_ids: Sequence[str],
            limit: int,
            metadata_filter: Mapping[str, str | int] | None = None,
        ) -> list[tuple[str, float]]:
            del query, limit, metadata_filter
            self.allowed_doc_ids = list(allowed_doc_ids)
            return [("POLICY-V1", 0.99), ("POLICY-V2", 0.8)]

    vector_store = FakeVectorStore()
    retriever = HybridRetriever([old, current], vector_store=vector_store)
    results = retriever.search(
        "annual leave carry forward deadline",
        is_allowed=lambda _: True,
        limit=5,
    )

    assert vector_store.allowed_doc_ids == ["POLICY-V2"]
    assert {match.chunk.document.doc_id for match in results} == {"POLICY-V2"}


def test_provenance_pseudonymizes_identity_and_validates_citations(tmp_path) -> None:
    database_path = tmp_path / "assistant.db"
    with sqlite3.connect(database_path):
        pass
    store = ProvenanceStore(database_path)
    store.ensure_schema()
    trace = ProvenanceTrace(
        retrieval_mode="hybrid",
        llm_provider="none",
        llm_model="none",
    )
    trace.record(
        "answer_generation_node",
        "evidence_selected",
        {
            "citations": [{"doc_id": "POLICY-1", "locator": "Annual leave"}],
            "corpus_tokens": 23,
        },
    )
    trace.record(
        "citation_validation_node",
        "validated",
        {"decision": "passed", "citation_count": 1},
    )
    response = ChatResponse(
        request_id="request-1",
        answer="Annual leave may be carried forward.",
        route="rag",
        citations=[ChatCitation(doc_id="POLICY-1", locator="Annual leave")],
    )

    store.save(
        trace,
        employee_id="E1001",
        session_id="private-session",
        response=response,
        response_status="success",
    )

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT employee_ref, conversation_ref, citation_validation,
                   token_usage_json, events_json
            FROM assistant_provenance_v2
            """
        ).fetchone()

    assert row is not None
    employee_ref, conversation_ref, validation, token_usage, events = row
    serialized = json.dumps([token_usage, events])
    assert employee_ref != "E1001"
    assert conversation_ref != "private-session"
    assert validation == "passed"
    assert json.loads(token_usage)["corpus_tokens"] == 23
    assert "private-session" not in serialized
    assert "E1001" not in serialized


def test_provenance_rejects_sensitive_event_fields() -> None:
    trace = ProvenanceTrace(
        retrieval_mode="bm25",
        llm_provider="none",
        llm_model="none",
    )

    try:
        trace.record("request_guard_node", "evaluated", {"message": "raw prompt"})
    except ProvenanceError as exc:
        assert "Sensitive fields" in str(exc)
    else:
        raise AssertionError("Raw prompt field was accepted into provenance")
