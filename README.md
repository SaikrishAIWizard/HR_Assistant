"# Veltrix PeopleOps Assistant

This repository implements the Veltrix HR-assistant assessment in small, testable
steps. The current implementation covers **Requirement 1: ingestion and
retrieval**; the HTTP API, document-precedence policy, leave tools, and
conversation workflow remain subsequent requirements.

## Requirement 1 plan and implementation

1. Validate every manifest record and safely resolve each source path.
2. Extract PDF text and tables, OCR textless PDF pages with Tesseract, preserve
   DOCX paragraphs/tables, and parse HTML and Markdown headings/content.
3. Split extracted content into token-bounded chunks while retaining document
   metadata, page number, and citation locator.
4. Search locally with BM25. The retriever requires an `is_allowed` metadata
   filter and applies it before ranking, so later access-control logic can
   prevent restricted content from reaching a model.
5. Build model context under the assessment's hard 3,000-token corpus-text
   ceiling using `cl100k_base`; log that count immediately before each model
   call.

The implementation uses a `src/` package layout:

- `src/veltrix_assistant/ingestion/` — schema validation, file loaders,
  chunking, and manifest-driven orchestration.
- `src/veltrix_assistant/retrieval/` — filtered BM25 search and context budget.
- `tests/` — focused ingestion, retrieval, and token-budget tests.

## Setup and run ingestion

Python 3.10 or later is required. PDF OCR uses the Tesseract executable; install
it in the runtime image/environment and ensure it is available on `PATH` (or
set `TESSERACT_CMD`).

```powershell
python -m pip install -e ".[dev]"
veltrix-ingest --corpus-dir .\corpus --manifest-path .\manifest.json
python -m pytest
```

By default, the command uses `CORPUS_DIR=./data/corpus` and
`MANIFEST_PATH=./data/manifest.json`, matching the assessment's runtime data
layout. The command reports document and chunk counts; it does not print
document contents.

The search API requires callers to pass a document-eligibility predicate. The
later service layer must derive that predicate from the employee record and
the assessment's region, date, and access rules."
