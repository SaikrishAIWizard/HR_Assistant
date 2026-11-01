"# Veltrix PeopleOps Assistant

This repository implements the Veltrix HR-assistant assessment in small, testable
steps. The current implementation covers **Requirement 1: ingestion and
retrieval**, **Requirement 2: document applicability and precedence**, and
**Requirement 3: leave tools and rules**. Identity/session protections and the
final conversation workflow remain subsequent requirements.

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

- `src/veltrix_assistant/ingestion/` — file loaders, chunking, and
  manifest-driven orchestration.
- `src/veltrix_assistant/schemas/` — all Pydantic models, grouped by document,
  policy, retrieval, API, and tool contracts. Business-rule decisions remain
  with their owning domain modules.
- `src/veltrix_assistant/policy/` — role/region/date eligibility, source
  precedence, tied-value conflict retention, and upcoming-change checks.
- `src/veltrix_assistant/tools/` — DB-backed leave tools, permission checks,
  calendar-based working-day counts, and atomic request/balance updates.
- `src/veltrix_assistant/retrieval/` — filtered BM25 search and context budget.
- `tests/` — focused ingestion, retrieval, and token-budget tests.

## Setup and run ingestion

Python 3.10 or later is required. PDF OCR uses the Tesseract executable; install
it in the runtime image/environment and ensure it is available on `PATH` (or
set `TESSERACT_CMD`). The checked-in `.env.example` uses the assessment's
`./data` layout; the local, ignored `.env` points to the supplied root-level
corpus and database.

```powershell
uv sync
uv run pytest
uv run veltrix-ingest --corpus-dir .\corpus --manifest-path .\manifest.json
uv run veltrix-api
```

`uv sync` creates and maintains the project `.venv`. The API starts ingestion
in the background; `GET /health` returns 503 while loading and 200 once ready.
`POST /search` accepts `{"query": "annual leave", "limit": 5}` and requires an
`X-Employee-Id` header found in the configured SQLite database. It applies
role, region, and reference-date document filters before retrieval. This
endpoint is for testing retrieval, not a replacement for the assessment's
required `POST /chat`. Its response separates currently applicable `results`
from `upcoming` changes that would win precedence on their effective date. The
decision rules and current extraction boundary are documented in
[`TASK2_TECHNICAL_IMPLEMENTATION.md`](./TASK2_TECHNICAL_IMPLEMENTATION.md).
The five callable leave tools are bound to an authenticated employee through
`app.state.leave_tool_factory.for_employee(employee_id)` for the future chat
handler; they are not exposed as separate HTTP routes. Their access rules,
atomicity, and date handling are documented in
[`TASK3_TECHNICAL_IMPLEMENTATION.md`](./TASK3_TECHNICAL_IMPLEMENTATION.md).

The ingestion command reports document and chunk counts without printing
document contents. Never put real API keys in `.env.example` or commit a local
`.env` file."
