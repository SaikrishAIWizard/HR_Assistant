"""Compatibility re-exports; Pydantic model definitions live in schemas/."""

from veltrix_assistant.schemas.documents import (
    AccessLevel,
    Citation,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    ExtractedSection,
    IngestedCorpus,
    RetrievedChunk,
)

__all__ = [
    "AccessLevel",
    "Citation",
    "DocumentChunk",
    "DocumentMetadata",
    "DocumentType",
    "ExtractedSection",
    "IngestedCorpus",
    "RetrievedChunk",
]
