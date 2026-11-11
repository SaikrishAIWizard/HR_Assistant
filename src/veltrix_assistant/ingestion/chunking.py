"""Token-bounded chunk creation with stable source locators."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

import tiktoken

from veltrix_assistant.schemas.documents import (
    DocumentChunk,
    DocumentMetadata,
    ExtractedSection,
)


ENCODING_NAME = "cl100k_base"
DEFAULT_CHUNK_TOKENS = 500
ChunkingStrategy = Literal["token_whitespace", "sentence_greedy"]
DEFAULT_CHUNKING_STRATEGY: ChunkingStrategy = "sentence_greedy"
DEFAULT_CHUNK_OVERLAP_PERCENT = 20


def chunk_sections(
    document: DocumentMetadata,
    sections: tuple[ExtractedSection, ...],
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    strategy: ChunkingStrategy = DEFAULT_CHUNKING_STRATEGY,
    overlap_percent: int | None = None,
) -> tuple[DocumentChunk, ...]:
    """Split sections while preserving source metadata and citation locators."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if strategy not in {"token_whitespace", "sentence_greedy"}:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")
    if overlap_percent is None:
        overlap_percent = (
            DEFAULT_CHUNK_OVERLAP_PERCENT
            if strategy == "sentence_greedy"
            else 0
        )
    if not 0 <= overlap_percent < 100:
        raise ValueError("overlap_percent must be between 0 and 99")
    if strategy == "token_whitespace" and overlap_percent:
        raise ValueError("overlap_percent requires the sentence_greedy strategy")

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    chunks: list[DocumentChunk] = []

    for section in sections:
        if strategy == "token_whitespace":
            split_text = _split_text(section.text, encoding, max_tokens)
        else:
            split_text = _sentence_greedy_split_text(
                section.text,
                encoding,
                max_tokens,
                overlap_percent=overlap_percent,
            )
        for part_number, text in enumerate(
            split_text, start=1
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


def _sentence_greedy_split_text(
    text: str,
    encoding: tiktoken.Encoding,
    max_tokens: int,
    *,
    overlap_percent: int = 0,
) -> tuple[str, ...]:
    """Pack sentences to a token budget, with optional token-based overlap."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if not 0 <= overlap_percent < 100:
        raise ValueError("overlap_percent must be between 0 and 99")
    remaining_budget = max_tokens - (max_tokens * overlap_percent // 100)
    if remaining_budget < 1:
        raise ValueError("overlap leaves no room for new sentence content")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    fresh_chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(encoding.encode(sentence)) > remaining_budget:
            if current:
                fresh_chunks.append(current)
                current = ""
            fresh_chunks.extend(_split_text(sentence, encoding, remaining_budget))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(encoding.encode(candidate)) > remaining_budget:
            fresh_chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        fresh_chunks.append(current)

    if overlap_percent == 0:
        return tuple(fresh_chunks)
    if not fresh_chunks:
        return ()

    overlap_tokens = max_tokens * overlap_percent // 100
    chunks = [fresh_chunks[0]]
    for previous, fresh_chunk in zip(fresh_chunks, fresh_chunks[1:], strict=False):
        prefix = _text_suffix_by_tokens(previous, encoding, overlap_tokens)
        while prefix:
            combined = f"{prefix} {fresh_chunk}".strip()
            if len(encoding.encode(combined)) <= max_tokens:
                break
            prefix = _text_suffix_by_tokens(
                previous,
                encoding,
                max(0, len(encoding.encode(prefix)) - 1),
            )
        chunks.append(f"{prefix} {fresh_chunk}".strip() if prefix else fresh_chunk)
    return tuple(chunks)


def _text_suffix_by_tokens(
    text: str,
    encoding: tiktoken.Encoding,
    max_tokens: int,
) -> str:
    if max_tokens < 1:
        return ""
    token_ids = encoding.encode(text)
    token_bytes = [
        encoding.decode_single_token_bytes(token_id) for token_id in token_ids
    ]
    for start in range(max(0, len(token_ids) - max_tokens), len(token_ids)):
        try:
            suffix = b"".join(token_bytes[start:]).decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if suffix and len(encoding.encode(suffix)) <= max_tokens:
            return suffix
    return ""


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
