"""Token-bounded chunk creation with stable source locators."""

from __future__ import annotations

import hashlib

import tiktoken

from veltrix_assistant.ingestion.schemas import DocumentChunk, DocumentMetadata, ExtractedSection


ENCODING_NAME = "cl100k_base"
DEFAULT_CHUNK_TOKENS = 500


def chunk_sections(
    document: DocumentMetadata,
    sections: tuple[ExtractedSection, ...],
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> tuple[DocumentChunk, ...]:
    """Split extracted sections without losing their document, page, or heading."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    chunks: list[DocumentChunk] = []

    for section in sections:
        for part_number, text in enumerate(
            _split_text(section.text, encoding, max_tokens), start=1
        ):
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


def _split_text(text: str, encoding: tiktoken.Encoding, max_tokens: int) -> tuple[str, ...]:
    token_ids = encoding.encode(text)
    if len(token_ids) <= max_tokens:
        return (text.strip(),) if text.strip() else ()

    chunks: list[str] = []
    for offset in range(0, len(token_ids), max_tokens):
        part = encoding.decode(token_ids[offset : offset + max_tokens]).strip()
        while part and len(encoding.encode(part)) > max_tokens:
            part = encoding.decode(encoding.encode(part)[:-1]).strip()
        if part:
            chunks.append(part)
    return tuple(chunks)
