# Design

## Ingestion, retrieval, and policy decisions

- `ingestion/loaders.py` extracts PDF text/tables, DOCX paragraphs/tables,
  Markdown, and HTML. It renders textless PDF pages and runs Tesseract OCR;
  extraction failures fail ingestion instead of silently dropping a document.
- `ingestion/chunking.py` makes 500-token `cl100k_base` chunks with document
  metadata and source locators. `retrieval/hybrid.py` creates the authorized
  candidate set first, then retrieves from BM25 and persistent Chroma and
  fuses stable chunk IDs with Reciprocal Rank Fusion: BM25 matches exact policy
  terms, while vectors add semantic matches without requiring their scores to
  share a scale. BM25-only mode remains available.
  `retrieval/vector_store.py` idempotently adds, refreshes, and
  removes indexed chunks. The default CPU feature-hashing vectors need no model
  download; they are not semantic language embeddings. Optional providers are
  a pre-provisioned local sentence-transformer model or OpenAI embeddings.
  `retrieval/context.py` caps each model call at 3,000 corpus-text tokens and
  logs that count before the call.
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
- `security/privacy.py` centrally detects and redacts common email, phone,
  employee-ID, government-ID, and bank-account patterns, plus configured
  sensitive terms, before external LLM or embedding calls. This is not a
  general name recognizer; teams must configure known sensitive terms and
  validate rules against their own data. Redaction can be explicitly disabled
  by configuration.
- Logs exclude message bodies, names, email, phone, and retrieved text.
  Employee records and ordinary leave-tool results are not passed to the model.
  A future-balance calculation receives only its current balance, join date,
  target date, the question, and authorized policy evidence. Model citations
  are validated against supplied evidence. Offline mode returns source
  excerpts without a model call and does not forecast balances.
- `assess_user_message` in `security/guards.py` refuses common bulk-data and
  instruction-override requests. Retrieved text cannot invoke tools or change
  permissions; authorization remains in application code.
- `workflow.py` routes requests through LangGraph guard, identity, intent,
  retrieval/tool dispatch, conversation, citation-validation, and response
  nodes. Conditional branches preserve direct confirmation and refusal paths.
  Set `LANGGRAPH_ENABLED=false` to use the direct compatibility flow.
- `provenance.py` records Version 2 graph/request traces, retrieval method and
  ranks, selected source identifiers and effective dates, tool outcomes,
  citation-validation outcome, and token counts. Employee/session references
  are HMAC-pseudonymized. Events reject raw message/document/tool-output keys;
  no answer or corpus text is persisted. Without a configured
  `PROVENANCE_PSEUDONYM_KEY`, a generated key is stored in the assistant-owned
  SQLite database. Provenance write failures return an explicit service error
  rather than silently dropping the audit record.

## Failure modes and scale

Missing OCR, unreadable sources, invalid identities, session conflicts,
provider failures, provenance-storage failures, and invalid proposals fail
explicitly or fail closed. SQLite transactions prevent partial leave/balance
updates and cancellation/restoration. These checks are covered by ingestion,
access-control, retrieval, privacy, provenance, leave, and chat integration
tests.

Verified routing/retrieval failures included superseded policy versions
appearing beside current policy, carry-forward questions being sent to a
personal-balance tool, and hybrid answers repeating tool facts or inferring an
unsupported deadline. Version-lineage filtering, policy-aware routing,
separate policy-question generation, and source-grounded deadline instructions
address these cases.

BM25 remains in memory and scales linearly with the chunk set; Chroma persists
vectors and supports metadata prefilters and incremental synchronization.
For larger corpora, benchmark batch ingestion, embedding throughput, index
startup, and authorization-filtered recall. Preserve the same pre-retrieval
authorization boundary when scaling or changing providers.
