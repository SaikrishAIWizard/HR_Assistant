"""Manifest-driven ingestion orchestration."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any

from pydantic import ValidationError

from veltrix_assistant.ingestion.chunking import chunk_sections
from veltrix_assistant.ingestion.loaders import IngestionError, load_document
from veltrix_assistant.schemas.documents import DocumentMetadata, IngestedCorpus


class ManifestError(ValueError):
    """Raised when manifest contents do not safely describe the corpus."""


def ingest_corpus(
    corpus_dir: str | Path,
    manifest_path: str | Path,
    *,
    chunk_tokens: int = 500,
) -> IngestedCorpus:
    """Load every manifest-listed document and return metadata-aware chunks."""
    corpus_root = Path(corpus_dir).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not corpus_root.is_dir():
        raise IngestionError(f"Corpus directory does not exist: {corpus_root}")
    if not manifest_file.is_file():
        raise ManifestError(f"Manifest does not exist: {manifest_file}")

    manifest_data = _read_manifest(manifest_file)
    documents = _validate_manifest(manifest_data)
    _ensure_all_supported_files_are_manifested(corpus_root, documents)

    chunks = []
    for metadata in documents:
        source_path = _resolve_document_path(corpus_root, metadata.file)
        sections = load_document(source_path)
        document_chunks = chunk_sections(metadata, sections, max_tokens=chunk_tokens)
        if not document_chunks:
            raise IngestionError(f"No chunks were created for {metadata.file}")
        chunks.extend(document_chunks)

    return IngestedCorpus(documents=tuple(documents), chunks=tuple(chunks))


def _read_manifest(manifest_path: Path) -> Any:
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read manifest {manifest_path}: {exc}") from exc


def _validate_manifest(data: Any) -> list[DocumentMetadata]:
    if not isinstance(data, list):
        raise ManifestError("Manifest must be a JSON array")

    try:
        documents = [DocumentMetadata.model_validate(item) for item in data]
    except (ValidationError, TypeError) as exc:
        raise ManifestError(f"Manifest contains invalid document metadata: {exc}") from exc

    if not documents:
        raise ManifestError("Manifest must contain at least one document")

    ids = [document.doc_id for document in documents]
    files = [document.file.casefold() for document in documents]
    if len(ids) != len(set(ids)):
        raise ManifestError("Manifest contains duplicate doc_id values")
    if len(files) != len(set(files)):
        raise ManifestError("Manifest contains duplicate file values")
    return documents


def _resolve_document_path(corpus_root: Path, relative_file: str) -> Path:
    candidate_value = Path(relative_file)
    if (
        candidate_value.is_absolute()
        or PureWindowsPath(relative_file).is_absolute()
        or ".." in candidate_value.parts
        or ".." in PureWindowsPath(relative_file).parts
    ):
        raise ManifestError(f"Manifest file must stay inside the corpus: {relative_file}")

    source_path = (corpus_root / candidate_value).resolve()
    if not source_path.is_relative_to(corpus_root):
        raise ManifestError(f"Manifest file must stay inside the corpus: {relative_file}")
    if source_path.suffix.casefold() not in {".pdf", ".docx", ".md", ".markdown", ".html", ".htm"}:
        raise ManifestError(f"Unsupported file in manifest: {relative_file}")
    if not source_path.is_file():
        raise ManifestError(f"Manifest document is missing: {relative_file}")
    return source_path


def _ensure_all_supported_files_are_manifested(
    corpus_root: Path, documents: list[DocumentMetadata]
) -> None:
    manifested = {document.file.casefold() for document in documents}
    unlisted = sorted(
        path.relative_to(corpus_root).as_posix()
        for path in corpus_root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in {".pdf", ".docx", ".md", ".markdown", ".html", ".htm"}
        and path.relative_to(corpus_root).as_posix().casefold() not in manifested
    )
    if unlisted:
        raise ManifestError(
            "Supported corpus files are missing from the manifest: " + ", ".join(unlisted)
        )
