"""Automatic document ingestion and chunking."""

from veltrix_assistant.ingestion.pipeline import ingest_corpus
from veltrix_assistant.schemas.documents import DocumentChunk, DocumentMetadata, IngestedCorpus

__all__ = ["DocumentChunk", "DocumentMetadata", "IngestedCorpus", "ingest_corpus"]
