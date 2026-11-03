# Task 4: Identity, Access Control, and Privacy

## Summary

Task 4 establishes reusable security controls for identity, session ownership,
document access, employee-data access, and privacy. The API resolves an
employee's role and region from the SQLite employee directory, not from
request-body fields. Retrieval filters documents by that trusted identity
before BM25/vector ranking. Leave tools independently enforce access against
the database, safe employee-record projection masks email and omits phone
numbers, and centralized PII rules redact outbound model/embedding inputs.
Versioned provenance records request decisions without raw messages or evidence.

The `/chat` endpoint is now implemented in Task 5 and consumes the
prompt-safety guard before retrieval or tool selection. The privacy projection
remains a reusable boundary for future profile features. Chat does not load
employee names, email, or phone. Annual-leave projections send only the
necessary current balance, join date, and target date with authorized evidence.

## Security requirements and implementation status

| Requirement | Implementation |
|---|---|
| Use the employee database as the source for role and region | `lookup_employee_identity` reads the employee row; unrecognized or invalid roles do not receive default access. |
| Bind a conversation session to its first employee | `session_id` values on `/chat` (and optional `/search` diagnostic requests) are persisted in `assistant_sessions`; another employee receives 403. |
| Filter unreadable documents before model exposure | `/search` and `/chat` apply role, region, and date predicates before results are returned or sent to the model. |
| Restrict employee leave data | Leave tools authorize self, manager/direct-report, and HR access in each operation; HR reads are one target per call. |
| Give the same denial for unauthorized and nonexistent targets | Leave tools return a fixed refusal for both cases. |
| Protect personal data | Employee-record projection masks email, omits phone, and includes names only for authorized viewers. Leave-tool output contains no names, emails, or phone numbers. |
| Treat prompt/document content as untrusted | A user-message guard returns refusal decisions for common override, hidden-data, and bulk-data patterns. Retrieved document text must remain evidence only. |
| Integrate safety decisions into `/chat` | `POST /chat` runs the guard before retrieval and tool selection; a refusal stops the request. |
| Redact sensitive values before external providers | `security/privacy.py` detects common identifiers and configured terms; LLM evidence/questions and external embedding inputs are redacted when enabled. |
| Keep an auditable, privacy-conscious trace | Provenance V2 stores pseudonymous employee/session references and safe workflow/retrieval metadata in assistant-owned tables. |

## Module-by-module

### `src/veltrix_assistant/security/identity.py`

- `lookup_employee_identity(database_path, employee_id)` queries only the
  employee's `region` and `role`, validates the role against the supported
  values (`employee`, `manager`, and `hr`), and returns an immutable
  `EmployeeIdentity`. Unknown IDs return `None`; database errors and malformed
  access records raise `IdentityDirectoryError`.
- `SessionBindingStore.ensure_schema()` creates the assistant-owned
  `assistant_sessions` table if needed. It refuses to create a missing
  database file and does not change assessment-owned tables or columns.
- `SessionBindingStore.bind(session_id, employee_id)` uses a SQLite
  `BEGIN IMMEDIATE` transaction to make first ownership persistent. The
  original owner can reuse the session; a different employee cannot replace
  that owner.

### `src/veltrix_assistant/schemas/security.py`

- `EmployeeIdentity` validates the trusted employee ID, region, and role.
- `RequestSafetyDecision` carries a `continue` or `refuse` route, an optional
  reason, and a fixed refusal message consumed by `POST /chat`.

### `src/veltrix_assistant/security/privacy.py`

- `mask_email` formats a valid address as
  `first-character***@domain`; malformed values are masked as `***`.
- `sanitize_employee_record` returns `None` for unauthorized records. It
  includes a name only for the employee themself, their direct manager, or HR;
  masks an included email; and never copies a phone field. Its
  non-personal-field allowlist avoids returning unrequested database fields.
- `detect_pii` returns only category and character spans for email, phone,
  employee-ID, government-ID, bank-account, and configured sensitive-term
  matches; it never includes matched values. `redact_pii` replaces findings
  with category labels or generic masks. Matching is intentionally
  deterministic and is not general-purpose name NER or DLP; configure
  `PII_SENSITIVE_TERMS` for known names/terms that must not leave the service.
- When enabled, retrieval content is redacted before constructing external
  LLM evidence, and `ChromaVectorStore` sanitizes documents and queries before
  invoking an external embedding provider. Local/offline use does not call an
  external model. Redaction can be controlled with
  `PII_REDACTION_ENABLED` and `PII_REDACTION_MODE`.
- The helper is an integration boundary for future profile/answer workflows;
  `/search` does not return employee profiles and therefore does not call it.

### `src/veltrix_assistant/security/guards.py`

- `assess_user_message` examines the user message, not retrieved document
  contents. It returns `route="refuse"` for detected prompt-override,
  hidden-data, or bulk employee-data patterns, with a fixed refusal message.
- Ordinary HR questions return `route="continue"`.
- This is a deterministic defense-in-depth check, not a complete natural
  language security classifier. Authorization is also enforced at retrieval
  and inside each tool. Task 5 runs the guard before retrieval and tool
  selection and stops processing when it returns `refuse`.
- Retrieved documents are untrusted evidence. Their contents must never be
  treated as instructions, used to change identity or permissions, or allowed
  to invoke tools.

### `src/veltrix_assistant/schemas/api.py`

`SearchRequest` accepts an optional Pydantic-validated `session_id` (trimmed,
non-empty, at most 128 characters). Existing clients that omit the field
continue to work.

### `src/veltrix_assistant/api.py`

- On startup, after corpus ingestion, the API initializes the session-binding
  table, synchronizes the configured retrieval index, and initializes
  provenance schema V2 when enabled. If initialization or ingestion fails,
  readiness stays unavailable.
- `POST /search` requires `X-Employee-Id`, resolves its identity from the
  employee database, and returns a generic 401 for a missing or unknown
  employee. Invalid directory data fails closed as a 503.
- When `session_id` is supplied, `/search` binds or verifies ownership after
  identity lookup. Reuse by another employee returns 403 without changing the
  owner.
- The role/region audience and reference date are used in retrieval predicates:
  current results must be readable and effective; upcoming candidates must be
  readable and future-dated. These filters are supplied to BM25 before ranking,
  and the authorized document-ID set constrains Chroma retrieval. Hybrid
  candidates are fused only after both retrieval paths are restricted.
- `/search` remains a retrieval diagnostic endpoint. The `/chat` endpoint
  executes the LangGraph guard and trusted-identity validation before retrieval
  or tool selection. `LANGGRAPH_ENABLED=false` selects a direct compatibility
  path.

### `src/veltrix_assistant/tools/leave.py`

Task 3's leave-tool authorization is an independent enforcement layer:

- The tool factory creates request-scoped tools bound to the employee
  authenticated by the API.
- Each operation loads the bound requester's role from the database and checks
  self, manager/direct-report, or HR scope before reading or writing target
  records. Caller-supplied target IDs do not establish identity.
- HR can access one employee per call. `get_team_leave` is reserved for
  managers and is limited to their direct reports.
- Unauthorized and nonexistent target IDs return the same fixed refusal.
- Returned leave data contains identifiers and the fields needed for leave
  operations, not names, email addresses, or phone numbers. Operational logs
  use generic error descriptions rather than employee personal data.

## Request and data flow

1. The client sends `X-Employee-Id`; a protected handler looks it up in the
   employee database and uses the returned role and region.
2. If a session ID is present, the handler binds it to the employee or rejects
   a cross-employee reuse.
3. Retrieval applies document role, region, and effective-date rules before
   both BM25 and vector ranking. Tool calls separately re-check the requester's
   database role and target scope.
4. Any employee profile intended for display or model context must first pass
   through `sanitize_employee_record`; only the minimum authorized fields
   should be sent onward.
5. `/chat` runs the guard and identity validation before retrieval or tool
   selection. A refusal stops the workflow. When an external provider is
   configured, authorized evidence and the question are passed through the PII
   sanitizer first; future-balance requests additionally receive only the
   minimum balance, join-date, and target-date facts.
6. Provenance V2 stores node decisions, retrieval ranks, selected citation
   identifiers/effective dates, tool outcomes, citation-validation result, and
   token counts. Employee/session IDs are HMAC-pseudonymized. Raw messages,
   answers, and document text are not stored in events. Without
   `PROVENANCE_PSEUDONYM_KEY`, the generated key is kept in an
   assistant-owned SQLite table. Persistence failure is surfaced as a service
   error rather than silently dropping the trace.

The employee ID header is an identity lookup key, not by itself a cryptographic
authentication mechanism. A deployed service must ensure that the header is
provided by a trusted authentication layer and cannot be freely asserted by an
untrusted client.

## Validation

The Task 4 behavior is covered by:

- `tests/test_security.py` — database identity resolution, invalid roles,
  persistent session ownership, safe employee-record projection, email
  masking, phone omission, PII detection/redaction, and message-guard decisions.
- `tests/test_api.py` — required employee header, generic unknown-ID rejection,
  role/region document filtering, session ownership enforcement, privacy-safe
  provenance, and session ID validation.
- `tests/test_hybrid_provenance.py` — authorization-first retrieval,
  incremental vector add/update/delete behavior, provenance pseudonymization,
  citation-validation state, token metadata, and forbidden-field rejection.
- `tests/test_leave_tools.py` — employee/manager/HR scopes, direct-report team
  access, and the same refusal for unauthorized and nonexistent employee IDs.

Run the complete project test suite from the repository root:

```powershell
uv run pytest -q
```

At the time this guide was updated, Task 5 integration tests exercise the
guard and route behavior through `POST /chat`.


-------------------------------
In short: SessionBindingStore answers "Does this authenticated employee own this session?", while lookup_employee_identity() answers "What role and region does the employee directory assign to this employee?" Both help establish trusted context before your assistant processes a request.