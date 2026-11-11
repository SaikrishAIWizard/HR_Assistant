import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import tiktoken

from scripts.benchmark_chunking import (
    _overlapping_split_text,
    _recursive_split_text,
    _sentence_split_text,
    write_chunk_size_chart,
)
from veltrix_assistant.ingestion.chunking import ENCODING_NAME
from veltrix_assistant.ingestion.chunking import _sentence_greedy_split_text
from veltrix_assistant.ingestion.chunking import chunk_sections
from veltrix_assistant.retrieval.vector_store import (
    ChromaVectorStore,
    LocalHashingEmbeddings,
)
from veltrix_assistant.schemas.documents import DocumentMetadata, ExtractedSection


def test_recursive_separator_split_stays_within_token_limit_and_keeps_words() -> None:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    text = (
        "Annual leave can be carried forward.\n\n"
        "Employees must submit a request to their manager. "
        "The effective date is 1 January 2026. "
    ) * 30

    chunks = _recursive_split_text(text, encoding, max_tokens=25)

    assert chunks
    assert all(len(encoding.encode(chunk)) <= 25 for chunk in chunks)
    assert re.findall(r"[\w'-]+", " ".join(chunks)) == re.findall(r"[\w'-]+", text)


def test_recursive_separator_split_does_not_split_short_text() -> None:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    text = "Short policy section.\n\nSecond paragraph."

    assert _recursive_split_text(text, encoding, max_tokens=50) == (text,)


def test_sentence_split_preserves_sentence_boundaries_when_possible() -> None:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    text = (
        "Annual leave is approved. Managers review requests. "
        "Employees receive confirmation."
    )

    chunks = _sentence_split_text(text, encoding, max_tokens=8)

    assert chunks
    assert all(len(encoding.encode(chunk)) <= 8 for chunk in chunks)
    assert all(chunk.endswith(".") for chunk in chunks)
    assert re.findall(r"[\w'-]+", " ".join(chunks)) == re.findall(r"[\w'-]+", text)


def test_sentence_greedy_overlap_stays_within_token_limit_and_repeats_context() -> None:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    text = " ".join(
        f"Policy sentence number {index} explains annual leave requirements."
        for index in range(50)
    )

    for overlap_percent in (10, 20):
        chunks = _sentence_greedy_split_text(
            text,
            encoding,
            max_tokens=50,
            overlap_percent=overlap_percent,
        )

        assert len(chunks) > 2
        assert all(len(encoding.encode(chunk)) <= 50 for chunk in chunks)
        expected_overlap = 50 * overlap_percent // 100
        first_chunk_tokens = encoding.encode(chunks[0])
        overlap_text = encoding.decode(first_chunk_tokens[-expected_overlap:])
        assert overlap_text.strip()
        assert any(
            chunks[1].startswith(
                encoding.decode(first_chunk_tokens[-token_count:]).strip()
            )
            for token_count in range(1, expected_overlap + 1)
            if encoding.decode(first_chunk_tokens[-token_count:]).strip()
        )


def test_production_default_uses_sentence_greedy_with_20_percent_overlap() -> None:
    document = DocumentMetadata(
        doc_id="TEST",
        file="test.md",
        title="Test policy",
        doc_type="policy",
        regions=("GLOBAL",),
        effective_date="2026-01-01",
        access_level="all",
    )
    section = ExtractedSection(
        page_number=1,
        locator="Annual leave",
        text=" ".join(
            f"Policy sentence number {index} explains annual leave requirements."
            for index in range(50)
        ),
    )
    default_chunks = chunk_sections(document, (section,), max_tokens=50)
    expected_chunks = chunk_sections(
        document,
        (section,),
        max_tokens=50,
        strategy="sentence_greedy",
        overlap_percent=20,
    )
    encoding = tiktoken.get_encoding(ENCODING_NAME)

    assert default_chunks == expected_chunks
    assert len(default_chunks) > 2
    assert all(len(encoding.encode(chunk.text)) <= 50 for chunk in default_chunks)


def test_overlapping_token_windows_respect_limit_and_keep_unicode() -> None:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    text = "Policy for café employees 👩‍💻. " * 30

    chunks = _overlapping_split_text(text, encoding, max_tokens=20)

    assert len(chunks) > 1
    assert all(len(encoding.encode(chunk)) <= 20 for chunk in chunks)
    assert all("\ufffd" not in chunk for chunk in chunks)
    assert chunks[0].startswith(text.strip()[:10])
    assert chunks[-1].endswith(text.strip()[-10:])


def test_benchmark_query_set_contains_30_unique_questions() -> None:
    query_path = Path("benchmarks/chunking_queries.json")
    query_set = json.loads(query_path.read_text(encoding="utf-8"))
    queries = query_set["queries"]

    assert len(queries) == 30
    assert len({query["id"] for query in queries}) == 30
    assert len({query["question"] for query in queries}) == 30


def test_chunk_size_chart_includes_all_production_size_comparisons(tmp_path: Path) -> None:
    output_path = tmp_path / "chunking_sizes.svg"
    report = {
        "evaluation": {"top_k_chunks": 20, "query_count": 30},
        "configurations": [
            {
                "strategy": "sentence_greedy_overlap_20pct",
                "chunk_tokens": size,
                "chunk_count": count,
                "mean_recall_at_20": 1.0,
                "mrr": 0.8,
                "bm25_mean_recall_at_20": 1.0,
                "bm25_mrr": 0.75,
            }
            for size, count in ((250, 165), (500, 161), (750, 161), (1000, 161))
        ],
    }

    write_chunk_size_chart(report, output_path)
    chart = output_path.read_text(encoding="utf-8")

    assert "20% overlap: token size comparison" in chart
    assert all(f">{size}</text>" in chart for size in (250, 500, 750, 1000))
    assert "Hybrid (Chroma + BM25/RRF)" in chart
    assert "BM25 baseline" in chart


def test_chroma_vector_store_close_releases_temporary_database() -> None:
    with TemporaryDirectory() as vector_db_path:
        vector_store = ChromaVectorStore(
            vector_db_path,
            embedding_provider=LocalHashingEmbeddings(),
            collection_name="chunking_benchmark_close_test",
        )
        vector_store.close()
