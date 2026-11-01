"""Token-bounded chunk creation with stable source locators."""

from __future__ import annotations

import hashlib

import tiktoken

from veltrix_assistant.schemas.documents import (
    DocumentChunk,
    DocumentMetadata,
    ExtractedSection,
)


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
    remaining = text.strip()
    chunks: list[str] = []
    while remaining:
        if len(encoding.encode(remaining)) <= max_tokens:
            chunks.append(remaining)
            break

        lower = 1
        upper = len(remaining)
        while lower < upper:
            midpoint = (lower + upper + 1) // 2
            if len(encoding.encode(remaining[:midpoint])) <= max_tokens:
                lower = midpoint
            else:
                upper = midpoint - 1

        boundary = max(
            remaining.rfind(" ", 0, lower),
            remaining.rfind("\n", 0, lower),
            remaining.rfind("\t", 0, lower),
        )
        cut = boundary if boundary > 0 else lower
        part = remaining[:cut].strip()
        while part and len(encoding.encode(part)) > max_tokens:
            cut -= 1
            part = remaining[:cut].strip()
        if cut < 1 or not part:
            raise ValueError("Unable to split source text within the chunk token budget")
        chunks.append(part)
        remaining = remaining[cut:].strip()
    return tuple(chunks)
