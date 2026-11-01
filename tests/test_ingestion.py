import json

import fitz
from docx import Document

from veltrix_assistant.ingestion.loaders import IngestionError
from veltrix_assistant.ingestion.pipeline import ManifestError, ingest_corpus


def _manifest_entry(doc_id: str, filename: str, doc_type: str = "policy") -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "file": filename,
        "title": f"Test {doc_id}",
        "doc_type": doc_type,
        "regions": ["GLOBAL"],
        "effective_date": "2026-01-01",
        "access_level": "all",
    }


def _write_pdf(path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Policy heading\nAnnual leave details.")
    document.save(path)
    document.close()


def _write_docx(path) -> None:
    document = Document()
    document.add_heading("Benefits", level=1)
    document.add_paragraph("Leave can be requested by employees.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Leave type"
    table.cell(0, 1).text = "Days"
    table.cell(1, 0).text = "Annual"
    table.cell(1, 1).text = "Twenty"
    document.save(path)


def test_ingests_supported_formats_and_preserves_metadata(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "policy.md").write_text(
        "# Eligibility\nEmployees may request leave.\n", encoding="utf-8"
    )
    (corpus_dir / "holiday.html").write_text(
        "<h1>Calendar</h1><table><tr><th>Date</th><th>Name</th></tr>"
        "<tr><td>2026-01-01</td><td>Holiday</td></tr></table>",
        encoding="utf-8",
    )
    _write_docx(corpus_dir / "benefits.docx")
    _write_pdf(corpus_dir / "policy.pdf")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                _manifest_entry("MD", "policy.md"),
                _manifest_entry("HTML", "holiday.html", "calendar"),
                _manifest_entry("DOCX", "benefits.docx"),
                _manifest_entry("PDF", "policy.pdf"),
            ]
        ),
        encoding="utf-8",
    )

    corpus = ingest_corpus(corpus_dir, manifest_path)

    assert len(corpus.documents) == 4
    assert corpus.chunks
    assert {chunk.document.doc_id for chunk in corpus.chunks} == {
        "MD",
        "HTML",
        "DOCX",
        "PDF",
    }
    assert any(chunk.locator == "Eligibility" for chunk in corpus.chunks)
    assert any("2026-01-01 | Holiday" in chunk.text for chunk in corpus.chunks)
    assert any("Leave type | Days" in chunk.text for chunk in corpus.chunks)
    assert all(chunk.document.effective_date.isoformat() == "2026-01-01" for chunk in corpus.chunks)


def test_rejects_manifest_paths_outside_corpus(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([_manifest_entry("ESCAPE", "..\\secret.md")]),
        encoding="utf-8",
    )

    try:
        ingest_corpus(corpus_dir, manifest_path)
    except ManifestError as exc:
        assert "stay inside the corpus" in str(exc)
    else:
        raise AssertionError("path traversal should be rejected")


def test_fails_when_a_supported_file_is_not_in_manifest(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "unlisted.md").write_text("# Not listed\nText", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]", encoding="utf-8")

    try:
        ingest_corpus(corpus_dir, manifest_path)
    except ManifestError as exc:
        assert "at least one document" in str(exc)
    else:
        raise AssertionError("an empty manifest should be rejected")


def test_unsupported_file_format_fails_explicitly(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "policy.txt").write_text("Policy", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([_manifest_entry("TXT", "policy.txt")]),
        encoding="utf-8",
    )

    try:
        ingest_corpus(corpus_dir, manifest_path)
    except ManifestError as exc:
        assert "Unsupported file" in str(exc)
    else:
        raise AssertionError("unsupported format should be rejected")


def test_pdf_without_text_requires_ocr(tmp_path, monkeypatch) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    path = corpus_dir / "scan.pdf"
    document = fitz.open()
    document.new_page()
    document.save(path)
    document.close()

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([_manifest_entry("SCAN", "scan.pdf")]),
        encoding="utf-8",
    )

    def fail_ocr(*args):
        raise IngestionError("OCR fallback called")

    monkeypatch.setattr("veltrix_assistant.ingestion.loaders._ocr_pdf_page", fail_ocr)
    try:
        ingest_corpus(corpus_dir, manifest_path)
    except IngestionError as exc:
        assert "OCR fallback called" in str(exc)
    else:
        raise AssertionError("textless PDF should trigger OCR")
