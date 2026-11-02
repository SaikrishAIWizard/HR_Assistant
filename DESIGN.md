# Design

## Ingestion, retrieval, and policy decisions

- `ingestion/loaders.py` extracts PDF text/tables, DOCX paragraphs/tables,
  Markdown, and HTML. It renders textless PDF pages and runs Tesseract OCR;
  extraction failures fail ingestion instead of silently dropping a document.
- `ingestion/chunking.py` makes 500-token `cl100k_base` chunks with document
  metadata and source locators. Local BM25 is deterministic and appropriate
  for the supplied small corpus; `retrieval/bm25.py` filters metadata before
  ranking. `retrieval/context.py` caps each model call at 3,000 corpus-text
  tokens and logs that count before the call.
- `policy/applicability.py` enforces role, region, effective date, and
  authority/scope/recency precedence. Tied top sources remain available to
  explain conflicts. `conversation.py` uses a requested past date and labels
  readable future winners separately. The system prompt contains instructions
  only; retrieved text is supplied separately as untrusted JSON evidence.

## Access control and privacy

- `api.py` resolves `X-Employee-Id` through `lookup_employee_identity`; role
  and region never come from the message. `SessionBindingStore.bind` in
  `security/identity.py` prevents session reuse by a different employee.
- `/search` and `/chat` apply `is_document_applicable` or
  `is_document_readable` from `policy/applicability.py` before retrieval
  results can reach the model. `tools/leave.py` authorizes every read and write
  through the authenticated employee; `can_access_employee_data` checks
  employee, direct-report manager, and HR scope. Write previews do not mutate
  data; execution rechecks rules in the same SQLite transaction as the update.
- Logs exclude message bodies, names, email, phone, and retrieved text.
  Employee records and ordinary leave-tool results are not passed to the model.
  A future-balance calculation receives only its current balance, join date,
  target date, the question, and authorized policy evidence. Email, phone, and
  employee-ID patterns in the question are redacted. Model citations are
  validated against supplied evidence. Offline mode returns source excerpts
  without a model call and does not forecast balances.
- `assess_user_message` in `security/guards.py` refuses common bulk-data and
  instruction-override requests. Retrieved text cannot invoke tools or change
  permissions; authorization remains in application code.

## Failure modes and scale

Missing OCR, unreadable sources, invalid identities, session conflicts, and
invalid proposals fail explicitly or fail closed. SQLite transactions prevent
partial leave/balance updates and cancellation/restoration. These checks are
covered by ingestion, access-control, leave, and chat integration tests.

With 10,000 documents, in-memory BM25 scoring over every chunk is the first
scaling limit. Replace it with a persistent local index that supports metadata
prefilters, batch indexing, and incremental re-ingestion. Preserve the same
authorization filters before evidence reaches the model.
