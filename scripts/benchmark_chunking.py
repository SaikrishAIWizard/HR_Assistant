"""Compare corpus chunking strategies with a small labeled hybrid query set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import tiktoken

from veltrix_assistant.conversation import _has_relevant_query_term
from veltrix_assistant.ingestion.chunking import (
    DEFAULT_CHUNK_OVERLAP_PERCENT,
    DEFAULT_CHUNKING_STRATEGY,
    ENCODING_NAME,
    _sentence_greedy_split_text,
    _split_text,
    chunk_sections,
)
from veltrix_assistant.ingestion.loaders import load_document
from veltrix_assistant.policy.applicability import is_document_applicable
from veltrix_assistant.retrieval.bm25 import BM25Retriever
from veltrix_assistant.retrieval.hybrid import HybridRetriever
from veltrix_assistant.retrieval.vector_store import (
    ChromaVectorStore,
    LocalHashingEmbeddings,
)
from veltrix_assistant.schemas.documents import (
    DocumentChunk,
    DocumentMetadata,
    ExtractedSection,
)
from veltrix_assistant.schemas.policy import DocumentAudience


DEFAULT_SIZES = (250, 500, 750, 1000)
RECURSIVE_SEPARATORS = ("\n\n", "\n", ". ", "; ", " ")
CHUNKING_STRATEGIES = (
    "token_whitespace",
    "recursive_separators",
    "sentence_greedy",
    "sentence_greedy_overlap_10pct",
    "sentence_greedy_overlap_20pct",
    "token_overlap_10pct",
)
PRODUCTION_STRATEGY = (
    f"{DEFAULT_CHUNKING_STRATEGY}_overlap_"
    f"{DEFAULT_CHUNK_OVERLAP_PERCENT}pct"
)


def _recursive_split_text(
    text: str,
    encoding: tiktoken.Encoding,
    max_tokens: int,
    separators: Sequence[str] = RECURSIVE_SEPARATORS,
) -> tuple[str, ...]:
    """Split at increasingly smaller text boundaries, then enforce token size."""
    remaining = text.strip()
    if not remaining:
        return ()
    if len(encoding.encode(remaining)) <= max_tokens:
        return (remaining,)
    if not separators:
        return _split_text(remaining, encoding, max_tokens)

    separator = separators[0]
    if separator not in remaining:
        return _recursive_split_text(
            remaining,
            encoding,
            max_tokens,
            separators[1:],
        )

    pieces = remaining.split(separator)
    units = [piece + separator for piece in pieces[:-1]]
    units.append(pieces[-1])
    output: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            output.append(current.strip())
        current = ""

    for unit in units:
        if not unit:
            continue
        if len(encoding.encode(unit)) > max_tokens:
            flush()
            output.extend(
                _recursive_split_text(
                    unit,
                    encoding,
                    max_tokens,
                    separators[1:],
                )
            )
            continue
        candidate = current + unit
        if len(encoding.encode(candidate)) <= max_tokens:
            current = candidate
        else:
            flush()
            current = unit

    flush()
    return tuple(output)


def _sentence_split_text(
    text: str,
    encoding: tiktoken.Encoding,
    max_tokens: int,
) -> tuple[str, ...]:
    """Greedily pack whole sentences, splitting only overlong sentences."""
    return _sentence_greedy_split_text(text, encoding, max_tokens)


def _overlapping_split_text(
    text: str,
    encoding: tiktoken.Encoding,
    max_tokens: int,
) -> tuple[str, ...]:
    """Split token windows with a 10% overlap, avoiding partial UTF-8 bytes."""
    token_ids = encoding.encode(text.strip())
    if len(token_ids) <= max_tokens:
        return (text.strip(),) if text.strip() else ()

    token_bytes = [encoding.decode_single_token_bytes(token) for token in token_ids]
    byte_offsets = [0]
    for token in token_bytes:
        byte_offsets.append(byte_offsets[-1] + len(token))
    text_bytes = b"".join(token_bytes)
    valid_boundaries = {
        index
        for index, offset in enumerate(byte_offsets)
        if offset == len(text_bytes) or text_bytes[offset] & 0xC0 != 0x80
    }

    overlap = max(1, max_tokens // 10)
    stride = max(1, max_tokens - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        while end > start:
            if end in valid_boundaries:
                candidate = text_bytes[byte_offsets[start] : byte_offsets[end]].decode(
                    "utf-8"
                ).strip()
                if candidate and len(encoding.encode(candidate)) <= max_tokens:
                    break
            end -= 1
        if end == start:
            raise ValueError("Unable to split text within the overlapping chunk token limit")
        chunks.append(candidate)
        if end == len(token_ids):
            break
        next_start = min(start + stride, end)
        start = min(
            (
                boundary
                for boundary in valid_boundaries
                if next_start <= boundary <= end
            ),
            default=end,
        )
    return tuple(chunks)


def _chunk_sections(
    strategy: str,
    document: DocumentMetadata,
    sections: tuple[ExtractedSection, ...],
    max_tokens: int,
    encoding: tiktoken.Encoding,
) -> tuple[DocumentChunk, ...]:
    chunk_text = {
        "recursive_separators": lambda text: _recursive_split_text(
            text, encoding, max_tokens
        ),
        "sentence_greedy": lambda text: _sentence_split_text(
            text, encoding, max_tokens
        ),
        "sentence_greedy_overlap_10pct": lambda text: _sentence_greedy_split_text(
            text, encoding, max_tokens, overlap_percent=10
        ),
        "sentence_greedy_overlap_20pct": lambda text: _sentence_greedy_split_text(
            text, encoding, max_tokens, overlap_percent=20
        ),
        "token_overlap_10pct": lambda text: _overlapping_split_text(
            text, encoding, max_tokens
        ),
    }
    if strategy == "token_whitespace":
        return chunk_sections(
            document,
            sections,
            max_tokens=max_tokens,
            strategy="token_whitespace",
        )
    if strategy == "sentence_greedy":
        return chunk_sections(
            document,
            sections,
            max_tokens=max_tokens,
            strategy="sentence_greedy",
            overlap_percent=0,
        )
    if strategy == "sentence_greedy_overlap_10pct":
        return chunk_sections(
            document,
            sections,
            max_tokens=max_tokens,
            strategy="sentence_greedy",
            overlap_percent=10,
        )
    if strategy == "sentence_greedy_overlap_20pct":
        return chunk_sections(
            document,
            sections,
            max_tokens=max_tokens,
            strategy="sentence_greedy",
            overlap_percent=20,
        )

    chunks: list[DocumentChunk] = []
    split_text = chunk_text[strategy]
    for section in sections:
        for part_number, text in enumerate(split_text(section.text), start=1):
            chunk_id = hashlib.sha256(
                (
                    f"{document.doc_id}\0{section.page_number}\0{section.locator}"
                    f"\0{part_number}\0{text}"
                ).encode("utf-8")
            ).hexdigest()
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document=document,
                    page_number=section.page_number,
                    locator=section.locator,
                    text=text,
                )
            )
    return tuple(chunks)


def _load_source_sections(
    corpus_dir: Path,
    manifest_path: Path,
) -> tuple[tuple[DocumentMetadata, tuple[ExtractedSection, ...]], ...]:
    raw_manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, list) or not raw_manifest:
        raise ValueError(f"{manifest_path} must contain a non-empty JSON array")

    corpus_root = corpus_dir.resolve()
    documents: list[tuple[DocumentMetadata, tuple[ExtractedSection, ...]]] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    for raw_document in raw_manifest:
        metadata = DocumentMetadata.model_validate(raw_document)
        if metadata.doc_id in seen_ids:
            raise ValueError(f"Duplicate document ID in manifest: {metadata.doc_id}")
        if metadata.file.casefold() in seen_files:
            raise ValueError(f"Duplicate document file in manifest: {metadata.file}")
        seen_ids.add(metadata.doc_id)
        seen_files.add(metadata.file.casefold())

        source_path = (corpus_root / metadata.file).resolve()
        if not source_path.is_relative_to(corpus_root):
            raise ValueError(f"Manifest path escapes corpus directory: {metadata.file}")
        if not source_path.is_file():
            raise FileNotFoundError(f"Manifest document does not exist: {source_path}")
        documents.append((metadata, load_document(source_path)))
    return tuple(documents)


def _percentile_95(values: Sequence[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def _evaluate_queries(
    retriever: BM25Retriever | HybridRetriever,
    queries: Sequence[dict[str, Any]],
    *,
    top_k: int,
) -> tuple[dict[str, float], list[dict[str, Any]], float]:
    per_query: list[dict[str, Any]] = []
    total_recall = 0.0
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for item in queries:
        audience = DocumentAudience(
            role=item["role"],
            region=item["region"],
        )
        reference_date = date.fromisoformat(item["reference_date"])
        question = item["question"]
        started = time.perf_counter()
        results = retriever.search(
            question,
            is_allowed=lambda metadata: is_document_applicable(
                metadata,
                audience,
                reference_date,
            ),
            limit=top_k,
            candidate_filter=lambda chunk: _has_relevant_query_term(
                question,
                chunk,
            ),
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)

        expected = set(item["relevant_doc_ids"])
        ranked_doc_ids: list[str] = []
        for result in results:
            doc_id = result.chunk.document.doc_id
            if doc_id not in ranked_doc_ids:
                ranked_doc_ids.append(doc_id)
        found = expected.intersection(ranked_doc_ids)
        reciprocal_rank = (
            1.0 / (min(ranked_doc_ids.index(doc_id) for doc_id in found) + 1)
            if found
            else 0.0
        )
        recall = len(found) / len(expected) if expected else 1.0
        total_recall += recall
        reciprocal_ranks.append(reciprocal_rank)
        per_query.append(
            {
                "id": item["id"],
                "recall_at_k": round(recall, 4),
                "reciprocal_rank": round(reciprocal_rank, 4),
                "expected_doc_ids": sorted(expected),
                "retrieved_doc_ids": ranked_doc_ids,
                "retrieved_chunks_by_method": {
                    "hybrid": sum(
                        result.retrieval_method == "hybrid" for result in results
                    ),
                    "bm25_only": sum(
                        result.retrieval_method == "bm25" for result in results
                    ),
                    "vector_only": sum(
                        result.retrieval_method == "vector" for result in results
                    ),
                }
                if isinstance(retriever, HybridRetriever)
                else None,
            }
        )

    query_count = len(queries)
    scores = {
        f"mean_recall_at_{top_k}": (
            round(total_recall / query_count, 4) if query_count else 0.0
        ),
        "mrr": (
            round(statistics.fmean(reciprocal_ranks), 4)
            if reciprocal_ranks
            else 0.0
        ),
    }
    mean_latency_ms = statistics.fmean(latencies_ms) if latencies_ms else 0.0
    return scores, per_query, mean_latency_ms


def run_benchmark(
    corpus_dir: Path,
    manifest_path: Path,
    query_path: Path,
    sizes: Sequence[int],
    *,
    top_k: int,
) -> dict[str, Any]:
    query_data = json.loads(query_path.read_text(encoding="utf-8"))
    queries = query_data["queries"]
    if not queries:
        raise ValueError(f"{query_path} contains no evaluation queries")
    sources = _load_source_sections(corpus_dir, manifest_path)
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    configurations: list[dict[str, Any]] = []

    for strategy in CHUNKING_STRATEGIES:
        for size in sizes:
            started = time.perf_counter()
            chunks: list[DocumentChunk] = []
            for metadata, sections in sources:
                built = _chunk_sections(strategy, metadata, sections, size, encoding)
                chunks.extend(built)
            chunking_seconds = time.perf_counter() - started

            chunk_sizes = [len(encoding.encode(chunk.text)) for chunk in chunks]
            bm25_scores, bm25_per_query, bm25_mean_latency_ms = _evaluate_queries(
                BM25Retriever(chunks),
                queries,
                top_k=top_k,
            )
            with tempfile.TemporaryDirectory(
                prefix="veltrix-chunking-hybrid-"
            ) as vector_db_path:
                vector_store = ChromaVectorStore(
                    vector_db_path,
                    embedding_provider=LocalHashingEmbeddings(),
                    collection_name="chunking_benchmark",
                )
                try:
                    vector_store.sync(chunks)
                    hybrid_retriever = HybridRetriever(
                        chunks,
                        vector_store=vector_store,
                        bm25_top_k=10,
                        vector_top_k=10,
                        hybrid_top_k=10,
                        rrf_k=60,
                    )
                    retrieval_scores, per_query, mean_latency_ms = _evaluate_queries(
                        hybrid_retriever,
                        queries,
                        top_k=top_k,
                    )
                finally:
                    vector_store.close()
            configurations.append(
                {
                    "strategy": strategy,
                    "chunk_tokens": size,
                    "chunk_count": len(chunks),
                    "mean_chunk_tokens": (
                        round(statistics.fmean(chunk_sizes), 2)
                        if chunk_sizes
                        else 0
                    ),
                    "p95_chunk_tokens": _percentile_95(chunk_sizes),
                    "max_chunk_tokens": max(chunk_sizes, default=0),
                    "chunking_seconds": round(chunking_seconds, 4),
                    "mean_query_ms": round(mean_latency_ms, 3),
                    "bm25_mean_query_ms": round(bm25_mean_latency_ms, 3),
                    f"bm25_mean_recall_at_{top_k}": bm25_scores.get(
                        f"mean_recall_at_{top_k}", 0.0
                    ),
                    "bm25_mrr": bm25_scores["mrr"],
                    "bm25_queries": bm25_per_query,
                    **retrieval_scores,
                    "queries": per_query,
                }
            )

    return {
        "benchmark": "chunking-retrieval-hybrid",
        "corpus_dir": str(corpus_dir),
        "manifest_path": str(manifest_path),
        "query_set": str(query_path),
        "query_set_note": query_data.get("note", ""),
        "evaluation": {
            "retriever": (
                "Production HybridRetriever: Chroma cosine search plus BM25, "
                "combined by RRF; BM25-only baseline also reported."
            ),
            "embedding_provider": "LocalHashingEmbeddings, 384 dimensions",
            "production_chunking": {
                "strategy": DEFAULT_CHUNKING_STRATEGY,
                "overlap_percent": DEFAULT_CHUNK_OVERLAP_PERCENT,
                "max_tokens": 500,
            },
            "vector_store": (
                "Chroma in a temporary directory per configuration; "
                "the repository's persistent vector database is not modified."
            ),
            "hybrid_settings": {
                "bm25_top_k_setting": 10,
                "vector_top_k_setting": 10,
                "hybrid_top_k_setting": 10,
                "rrf_k": 60,
                "evaluation_return_limit": top_k,
                "candidate_filter": (
                    "Production _has_relevant_query_term on the original query"
                ),
                "vector_similarity_floor": 0.1,
            },
            "query_count": len(queries),
            "query_labels": "seed labels; manually review before interpreting as ground truth",
            "access_filter": (
                "Role, region, and reference date are applied. "
                "The production chat keyword-relevance filter is also applied."
            ),
            "metric_unit": "unique relevant document IDs among top-k retrieved chunks",
            "top_k_chunks": top_k,
            "metrics": [
                "hybrid mean_recall_at_k",
                "hybrid mrr",
                "bm25_mean_recall_at_k",
                "bm25_mrr",
                "chunk_count",
                "hybrid mean_query_ms",
            ],
        },
        "configurations": configurations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(os.environ.get("CORPUS_DIR", "./corpus")),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(os.environ.get("MANIFEST_PATH", "./manifest.json")),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("benchmarks/chunking_queries.json"),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Chunk token limits to compare (default: 250 500 750 1000).",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--chart-output",
        type=Path,
        help="Write an SVG comparison chart for the selected --chart-size.",
    )
    parser.add_argument(
        "--size-chart-output",
        type=Path,
        help="Write an SVG comparing token limits for the production splitter.",
    )
    parser.add_argument(
        "--chart-size",
        type=int,
        default=250,
        help="Chunk size shown in the SVG chart (default: 250).",
    )
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes):
        parser.error("--sizes values must all be positive")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.chart_size < 1:
        parser.error("--chart-size must be positive")

    report = run_benchmark(
        args.corpus_dir,
        args.manifest_path,
        args.queries,
        args.sizes,
        top_k=args.top_k,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote benchmark report to {args.output}")
    if args.chart_output is not None:
        write_benchmark_chart(report, args.chart_output, args.chart_size)
    if args.size_chart_output is not None:
        write_chunk_size_chart(report, args.size_chart_output)


def write_benchmark_chart(
    report: dict[str, Any],
    output_path: Path,
    chunk_size: int,
) -> None:
    configurations = [
        item
        for item in report["configurations"]
        if item["chunk_tokens"] == chunk_size
    ]
    if not configurations:
        raise ValueError(f"No benchmark configurations found for {chunk_size} tokens")

    width = 1000
    height = 550
    strategies = [item["strategy"] for item in configurations]
    colors = ("#2563eb", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777")
    recall_key = f"mean_recall_at_{report['evaluation']['top_k_chunks']}"
    panels = (
        (
            f"Mean Recall@{report['evaluation']['top_k_chunks']}",
            lambda item: item[recall_key],
            1.0,
        ),
        ("MRR", lambda item: item["mrr"], 1.0),
        ("Chunk count", lambda item: item["chunk_count"], max(item["chunk_count"] for item in configurations)),
    )
    panel_width = 300
    panel_gap = 25
    left = 25
    top = 72
    plot_height = 300
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-family="Arial,sans-serif" font-size="21" font-weight="700" fill="#111827">Chunking strategies with hybrid retrieval — {chunk_size}-token limit</text>',
    ]
    for panel_index, (title, value_of, maximum) in enumerate(panels):
        x = left + panel_index * (panel_width + panel_gap)
        plot_top = top + 28
        bar_width = 43
        group_gap = (panel_width - len(strategies) * bar_width) / (len(strategies) + 1)
        elements.append(
            f'<text x="{x + panel_width / 2}" y="{top + 8}" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" font-weight="700" fill="#111827">{escape(title)}</text>'
        )
        elements.append(
            f'<line x1="{x}" y1="{plot_top + plot_height}" x2="{x + panel_width}" y2="{plot_top + plot_height}" stroke="#9ca3af"/>'
        )
        for index, item in enumerate(configurations):
            value = value_of(item)
            bar_height = (value / maximum if maximum else 0) * plot_height
            bar_x = x + group_gap * (index + 1) + bar_width * index
            bar_y = plot_top + plot_height - bar_height
            label = f"{value:.4f}" if isinstance(value, float) else str(value)
            elements.extend(
                (
                    f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[index % len(colors)]}"/>',
                    f'<text x="{bar_x + bar_width / 2:.1f}" y="{bar_y - 7:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#111827">{label}</text>',
                )
            )
    legend_y = 448
    legend_step = width / 3
    for index, strategy in enumerate(strategies):
        legend_x = (index % 3) * legend_step + 24
        row_y = legend_y + (index // 3) * 24
        elements.extend(
            (
                f'<rect x="{legend_x}" y="{row_y - 11}" width="12" height="12" fill="{colors[index % len(colors)]}"/>',
                f'<text x="{legend_x + 19}" y="{row_y}" font-family="Arial,sans-serif" font-size="11" fill="#374151">{escape(strategy)}</text>',
            )
        )
    note_y = 510
    elements.append(
        f'<text x="{width / 2}" y="{note_y}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#4b5563">Chroma + BM25 fused with RRF · {report["evaluation"]["query_count"]} provisional queries · access and keyword filters applied</text>'
    )
    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_chunk_size_chart(report: dict[str, Any], output_path: Path) -> None:
    configurations = sorted(
        (
            item
            for item in report["configurations"]
            if item["strategy"] == PRODUCTION_STRATEGY
        ),
        key=lambda item: item["chunk_tokens"],
    )
    if not configurations:
        raise ValueError(
            f"No {PRODUCTION_STRATEGY} configurations found in benchmark report"
        )

    recall_key = f"mean_recall_at_{report['evaluation']['top_k_chunks']}"
    panels = (
        (
            f"Mean Recall@{report['evaluation']['top_k_chunks']}",
            recall_key,
            f"bm25_mean_recall_at_{report['evaluation']['top_k_chunks']}",
            1.0,
        ),
        ("MRR", "mrr", "bm25_mrr", 1.0),
        (
            "Total corpus chunks",
            "chunk_count",
            None,
            max(item["chunk_count"] for item in configurations),
        ),
    )
    width = 1050
    height = 560
    panel_width = 320
    panel_gap = 25
    top = 72
    plot_height = 310
    bar_width = 22
    size_colors = ("#2563eb", "#16a34a", "#d97706", "#7c3aed")
    hybrid_color = "#2563eb"
    bm25_color = "#9ca3af"
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-family="Arial,sans-serif" font-size="21" font-weight="700" fill="#111827">Production sentence-greedy 20% overlap: token size comparison</text>',
    ]
    for panel_index, (title, hybrid_key, bm25_key, maximum) in enumerate(panels):
        x = 25 + panel_index * (panel_width + panel_gap)
        plot_top = top + 28
        group_width = panel_width / len(configurations)
        elements.extend(
            (
                f'<text x="{x + panel_width / 2}" y="{top + 8}" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" font-weight="700" fill="#111827">{escape(title)}</text>',
                f'<line x1="{x}" y1="{plot_top + plot_height}" x2="{x + panel_width}" y2="{plot_top + plot_height}" stroke="#9ca3af"/>',
            )
        )
        for index, item in enumerate(configurations):
            group_center = x + group_width * (index + 0.5)
            if bm25_key is None:
                values = ((item[hybrid_key], size_colors[index]),)
            else:
                values = (
                    (item[hybrid_key], hybrid_color),
                    (item[bm25_key], bm25_color),
                )
            offsets = (
                (0,)
                if len(values) == 1
                else (-bar_width / 2, bar_width / 2)
            )
            for (value, color), offset in zip(values, offsets, strict=True):
                bar_height = (value / maximum if maximum else 0) * plot_height
                bar_x = group_center + offset - bar_width / 2
                bar_y = plot_top + plot_height - bar_height
                label = f"{value:.4f}" if isinstance(value, float) else str(value)
                elements.extend(
                    (
                        f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>',
                        f'<text x="{bar_x + bar_width / 2:.1f}" y="{bar_y - 7:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="9" fill="#111827">{label}</text>',
                    )
                )
            elements.append(
                f'<text x="{group_center:.1f}" y="{plot_top + plot_height + 22}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#374151">{item["chunk_tokens"]}</text>'
            )
    legend_y = 465
    elements.extend(
        (
            f'<rect x="390" y="{legend_y - 11}" width="12" height="12" fill="{hybrid_color}"/>',
            f'<text x="408" y="{legend_y}" font-family="Arial,sans-serif" font-size="11" fill="#374151">Hybrid (Chroma + BM25/RRF)</text>',
            f'<rect x="610" y="{legend_y - 11}" width="12" height="12" fill="{bm25_color}"/>',
            f'<text x="628" y="{legend_y}" font-family="Arial,sans-serif" font-size="11" fill="#374151">BM25 baseline</text>',
        )
    )
    elements.append(
        f'<text x="{width / 2}" y="495" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#4b5563">Fixed strategy: {PRODUCTION_STRATEGY} · {report["evaluation"]["query_count"]} provisional queries</text>'
    )
    elements.append(
        f'<text x="{width / 2}" y="520" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#4b5563">Recall/MRR are query-set quality measures; chunk count is corpus-wide. Higher limit is not automatically better.</text>'
    )
    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
