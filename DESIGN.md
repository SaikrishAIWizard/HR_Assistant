# Design

## Ingestion, chunking, and retrieval

`ingestion/pipeline.py` validates each manifest entry, loads its source, and
creates chunks in `ingestion/chunking.py` with a 500-token maximum using
sentence-greedy packing and 20% token overlap. Chunks retain document metadata,
page number, heading/section locator, and a stable content-derived ID. This
keeps citations traceable and permits index synchronization. The 500-token
value is a maximum, not a measured optimum. The offline
`scripts/benchmark_chunking.py` compares chunk sizes, token/whitespace,
recursive-separator, sentence-greedy, sentence-greedy with 10%/20% overlap,
and fixed token-window strategies. It uses the production `HybridRetriever`
(local hashing embeddings + Chroma, BM25, and RRF), alongside a BM25-only
baseline and 30 provisional query labels. Review those labels before using the
scores as evidence; the evaluation does not validate other embedding
providers or answer quality. It writes SVG charts for the strategy and
token-limit comparisons using temporary indexes, leaving the persistent app
index untouched.

`ingestion/loaders.py` extracts PDF text and tables with pdfplumber, serializing
table rows into searchable text; it also preserves DOCX tables and reads HTML
and Markdown tables. PDF pages with fewer than 40 extracted characters are
rendered with PyMuPDF and OCR'd automatically with Tesseract. Missing OCR
support, timeouts, or extraction errors fail ingestion rather than silently
discarding content. No corpus content is manually transcribed.

`api.py` builds a BM25 retriever for every run. In hybrid mode it also syncs
chunks to local persistent Chroma and combines BM25 and vector ranks using
reciprocal rank fusion in `retrieval/hybrid.py`; this combines lexical matches
with semantic candidates without comparing unlike score scales. BM25-only
mode avoids Chroma and is useful when embeddings are unavailable. The default
CPU feature-hashing embeddings need no model download, but are less semantic
than a language embedding model. Chat uses both the original question and the
planner's search terms. It filters BM25 candidates by query relevance before
ranking and removes irrelevant vector candidates before final fusion, so
unrelated matches cannot crowd exact lexical evidence out of the selected
results. `retrieval/context.py` keeps corpus text sent to each
answer-generation call at or below 3,000 tokens and logs that count.

## Applicability and precedence

`policy/applicability.py` implements the document rules. `is_document_readable`
checks access level and region; `is_document_applicable` also checks the
effective date. Chat and search filter candidates with these checks before
returning evidence to the answer model. Chat's relevance filter runs after
authorization but before final hybrid ranking is truncated.
`select_precedence_winners` orders applicable evidence by authority, region
specificity, then effective date, preserving true ties so conflicting values
are not silently chosen. Superseded versions are excluded, and
`get_upcoming_winners` allows readable future winners to be identified and
cited separately. `conversation.py` resolves requested historical reference
dates and assembles the current and upcoming evidence.

## Access control and leave actions

`api.py` requires `X-Employee-Id` and uses
`security/identity.py:lookup_employee_identity` to load role and region from
SQLite; message claims cannot set identity. `SessionBindingStore.bind` binds a
session to its first employee. `/search` and chat retrieval both filter
documents using the authenticated audience before evidence can reach a model.

Leave permissions are enforced inside `tools/leave.py`, not by the planner:
`_requester`, `can_access_employee_data`, and `_target_is_allowed` check the
authenticated employee, HR scope, and manager direct-report scope. Leave
requests and balances are accessed only through these tools. Apply/cancel
previews validate leave type, dates, balance, overlap, and working days without
mutation; working-day counts exclude weekends and the applicable regional
public holidays. Execution rechecks the rules and updates the request and
balance in one SQLite transaction. `conversation.py` requires explicit
confirmation before executing a proposed write. With an LLM configured,
`workflow.py` uses a request-planning call to select document search or one of
the five leave actions. The planner receives the message (PII-redacted before
external calls by default) and reference date, returns an action/search query
rather than executable arguments, and never receives retrieved text.
Application code detects known HR scopes, including compensation and salary
bands, and overrides an erroneous planner refusal for those in-scope requests.
It also preserves explicitly detected leave actions when the planner
misclassifies them as document searches. Incomplete leave requests trigger
clarification before a preview can be prepared. Application code parses tool
arguments, while tool code remains the permission boundary. Retrieval still
filters by authenticated role, region, and date, so recognizing a salary
question does not grant access to HR-restricted compensation sources.
Clarification state is persisted by session and authenticated employee. It
retains the original leave question and the pending choice or missing field,
allowing short replies to resolve the existing lookup without weakening tool
authorization or treating clarification replies as write confirmations.

## Personal data, prompts, and audit

`security/privacy.py:redact_pii` redacts detected email, phone, identifiers,
and configured sensitive terms before external model or embedding calls
(redaction is enabled by default and configurable). The planner receives no
retrieved content. Answer generation receives the question and only the
authorized evidence needed to answer; ordinary leave-tool results are not sent
to the model. Future-balance calculations pass only the current balance, join
date, target date, and relevant policy evidence. Application logs do not
include raw messages, answers, or document text.

`provenance.py` records ordered routing, authorization, retrieval, tool,
citation-validation, and token-usage events in assistant-owned SQLite tables.
It stores HMAC pseudonyms instead of employee/session IDs and rejects sensitive
event keys; traces retain evidence identifiers and hashes, not source text.
`api.py` persists the trace after a response or handled workflow error. A
provenance storage failure returns an explicit service error. This audit trail
helps explain which authorized evidence or tool outcome produced a response
without retaining the conversation itself.

## Failure modes and scale

Failures addressed include stale/superseded policy evidence, policy questions
being mistaken for personal-balance lookups, unsupported carry-forward
deadlines, duplicate tool facts in hybrid answers, and relevant calendar
sections being lost when retrieval ranked unrelated chunks first. Version
filtering, planner-assisted routing, bounded grounded answers, and focused
retrieval queries address these cases. Automated tests cover ingestion,
precedence, role-based access, leave rules, privacy/provenance, and `/chat`.
Provider, OCR, identity, database, and provenance failures are reported
explicitly or fail closed.

At 10,000 documents, the first likely bottlenecks are BM25's per-query scan and
retokenization of all visible chunks, memory for the in-process corpus, and
startup/index synchronization. The vector index does not remove the need to
load chunks for metadata filtering and citation context. I would measure
ingestion/startup time, memory, latency, and authorization-filtered recall;
then use batched incremental indexing and a persistent sparse index with
precomputed term statistics, retaining metadata prefilters before either
retriever. Scale the vector store independently if benchmarks show it is also
a bottleneck, and keep the same access-control boundary.
