"""File-format loaders for the assessment corpus."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pdfplumber
import pymupdf
from bs4 import BeautifulSoup, Tag
from docx import Document
from docx.document import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from veltrix_assistant.schemas.documents import ExtractedSection


class IngestionError(RuntimeError):
    """Raised when an input document cannot be ingested reliably."""


_NUMBERED_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*[.)]\s+\S.{1,100}$")
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_MIN_TEXT_BEFORE_OCR = 40
_SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".html", ".htm"}


def load_document(path: Path) -> tuple[ExtractedSection, ...]:
    """Extract readable sections from a supported document."""
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        sections = _load_pdf(path)
    elif suffix == ".docx":
        sections = _load_docx(path)
    elif suffix in {".md", ".markdown"}:
        sections = _load_markdown(path)
    elif suffix in {".html", ".htm"}:
        sections = _load_html(path)
    else:
        raise IngestionError(f"Unsupported document format: {suffix or '(none)'}")

    if not any(section.text.strip() for section in sections):
        raise IngestionError(f"No text could be extracted from {path.name}")
    return tuple(section for section in sections if section.text.strip())


def _load_pdf(path: Path) -> list[ExtractedSection]:
    sections: list[ExtractedSection] = []
    current_heading: str | None = None

    try:
        with pymupdf.open(path) as layout_document, pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(layout=False) or ""
                table_blocks = [
                    _format_table(table)
                    for table in (page.extract_tables() or [])
                    if table and any(any(cell for cell in row) for row in table)
                ]
                if table_blocks:
                    text = "\n\n".join(part for part in (text, *table_blocks) if part.strip())

                if len(text.strip()) < _MIN_TEXT_BEFORE_OCR:
                    ocr_text = _ocr_pdf_page(path, page_number)
                    if ocr_text.strip():
                        text = "\n\n".join(part for part in (text, ocr_text) if part.strip())

                heading_candidates = _pdf_typographic_headings(
                    layout_document.load_page(page_number - 1)
                )
                page_sections, current_heading = _split_plain_text(
                    text,
                    page_number=page_number,
                    initial_heading=current_heading,
                    fallback_locator=f"p. {page_number}",
                    heading_candidates=heading_candidates,
                )
                sections.extend(page_sections)
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Failed to read PDF {path.name}: {exc}") from exc

    return sections


def _pdf_typographic_headings(page: pymupdf.Page) -> frozenset[str]:
    """Find likely PDF title/heading lines from font size and weight."""
    lines: list[tuple[str, list[dict[str, object]]]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(str(span.get("text", "")) for span in spans).strip()
            if text and len(text) <= 120:
                lines.append((text, spans))

    font_size_weights: Counter[float] = Counter()
    for _, spans in lines:
        for span in spans:
            text = str(span.get("text", "")).strip()
            if text and "size" in span:
                font_size_weights[float(span["size"])] += len(text)
    total_characters = sum(font_size_weights.values())
    if not total_characters:
        return frozenset()

    midpoint = (total_characters + 1) // 2
    weighted_characters = 0
    typical_size = 0.0
    for font_size, character_count in sorted(font_size_weights.items()):
        weighted_characters += character_count
        if weighted_characters >= midpoint:
            typical_size = font_size
            break

    candidates = set()
    for text, spans in lines:
        letters = [character for character in text if character.isalpha()]
        if len(letters) < 3:
            continue
        largest_size = max(
            (float(span.get("size", 0.0)) for span in spans),
            default=0.0,
        )
        is_bold = any(
            "bold" in str(span.get("font", "")).casefold()
            or bool(int(span.get("flags", 0)) & 16)
            for span in spans
        )
        if largest_size >= typical_size * 1.15 and (
            is_bold or largest_size >= typical_size * 1.4
        ):
            candidates.add(text)
    return frozenset(candidates)


def _ocr_pdf_page(path: Path, page_number: int) -> str:
    executable = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not executable:
        raise IngestionError(
            f"{path.name} page {page_number} needs OCR, but Tesseract was not found"
        )

    image_path: str | None = None
    try:
        with pymupdf.open(path) as document:
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
                image_path = image_file.name
            pixmap.save(image_path)

        result = subprocess.run(
            [executable, image_path, "stdout", "--psm", "6"],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as exc:
        raise IngestionError(
            f"OCR timed out for {path.name} page {page_number}"
        ) from exc
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise IngestionError(
            f"OCR failed for {path.name} page {page_number}: {exc}"
        ) from exc
    finally:
        if image_path:
            Path(image_path).unlink(missing_ok=True)

    return result.stdout.strip()


def _load_docx(path: Path) -> list[ExtractedSection]:
    try:
        document = Document(path)
        blocks = list(_iter_docx_blocks(document))
    except Exception as exc:
        raise IngestionError(f"Failed to read DOCX {path.name}: {exc}") from exc

    sections: list[ExtractedSection] = []
    active_heading: str | None = None
    grouped_text: list[str] = []

    def flush() -> None:
        if grouped_text:
            sections.append(
                ExtractedSection(
                    page_number=1,
                    locator=active_heading or "p. 1",
                    text="\n".join(grouped_text).strip(),
                )
            )
            grouped_text.clear()

    for block in blocks:
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            if block.style and block.style.name.casefold().startswith("heading"):
                flush()
                active_heading = text
                grouped_text.append(text)
            else:
                grouped_text.append(text)
        else:
            table_text = _format_table(
                [[cell.text.strip() for cell in row.cells] for row in block.rows]
            )
            if table_text.strip():
                grouped_text.append(table_text)

    flush()
    return sections


def _iter_docx_blocks(document: DocxDocument) -> Iterable[Paragraph | Table]:
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def _load_markdown(path: Path) -> list[ExtractedSection]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise IngestionError(f"Failed to read Markdown {path.name}: {exc}") from exc

    sections: list[ExtractedSection] = []
    active_heading: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(
                ExtractedSection(
                    page_number=1,
                    locator=active_heading or "p. 1",
                    text=content,
                )
            )
            current_lines.clear()

    for line in text.splitlines():
        match = _MARKDOWN_HEADING.match(line)
        if match:
            flush()
            active_heading = match.group(1).strip()
            current_lines.append(line.strip())
        else:
            current_lines.append(line)
    flush()
    return sections


def _load_html(path: Path) -> list[ExtractedSection]:
    try:
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    except (OSError, UnicodeError) as exc:
        raise IngestionError(f"Failed to read HTML {path.name}: {exc}") from exc

    sections: list[ExtractedSection] = []
    active_heading: str | None = None
    current_text: list[str] = []
    body = soup.body or soup

    def flush() -> None:
        content = "\n".join(current_text).strip()
        if content:
            sections.append(
                ExtractedSection(
                    page_number=1,
                    locator=active_heading or "p. 1",
                    text=content,
                )
            )
            current_text.clear()

    for element in body.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]
    ):
        if element.name in {"p", "li"} and element.find_parent(
            ["p", "li", "table"]
        ):
            continue
        if element.name and re.fullmatch(r"h[1-6]", element.name):
            flush()
            active_heading = element.get_text(" ", strip=True)
            if active_heading:
                current_text.append(active_heading)
        elif element.name == "table":
            rows = [
                [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                for row in element.find_all("tr")
            ]
            table_text = _format_table(rows)
            if table_text.strip():
                current_text.append(table_text)
        else:
            text = element.get_text(" ", strip=True)
            if text:
                current_text.append(text)

    if not current_text:
        fallback_text = body.get_text("\n", strip=True)
        if fallback_text:
            current_text.append(fallback_text)
    flush()
    return sections


def _split_plain_text(
    text: str,
    *,
    page_number: int,
    initial_heading: str | None,
    fallback_locator: str,
    heading_candidates: frozenset[str] = frozenset(),
) -> tuple[list[ExtractedSection], str | None]:
    sections: list[ExtractedSection] = []
    active_heading = initial_heading
    current_lines: list[str] = []

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(
                ExtractedSection(
                    page_number=page_number,
                    locator=active_heading or fallback_locator,
                    text=content,
                )
            )
            current_lines.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped and (
            _looks_like_heading(stripped) or stripped in heading_candidates
        ):
            flush()
            active_heading = stripped
            current_lines.append(stripped)
        else:
            current_lines.append(line)
    flush()
    return sections, active_heading


def _looks_like_heading(line: str) -> bool:
    if len(line) > 120:
        return False
    if line.rstrip(":").casefold() == "table":
        return False
    if _NUMBERED_HEADING.match(line):
        return True
    letters = [character for character in line if character.isalpha()]
    return bool(letters) and len(letters) <= 80 and all(
        character.isupper() for character in letters
    )


def _format_table(rows: Iterable[Iterable[str | None]]) -> str:
    formatted_rows = [
        " | ".join((cell or "").replace("\n", " ").strip() for cell in row)
        for row in rows
    ]
    return "TABLE:\n" + "\n".join(row for row in formatted_rows if row.strip())
