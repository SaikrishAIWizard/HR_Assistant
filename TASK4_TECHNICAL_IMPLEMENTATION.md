# Task 4: Identity, Access Control, and Privacy

## Summary

Task 4 establishes reusable security controls for identity, session ownership,
document access, employee-data access, and privacy. The API resolves an
employee's role and region from the SQLite employee directory, not from
request-body fields. Retrieval filters documents by that trusted identity
before ranking them. Leave tools independently enforce access against the
database, and safe employee-record projection masks email and omits phone
numbers.

The repository does not yet implement the assessment's `/chat` endpoint.
Consequently, the prompt-safety guard and privacy projection are available as
tested helpers, but Task 5 must call them in the final request workflow.
This document distinguishes those implemented foundations from the remaining
integration.

## Security requirements and implementation status

| Requirement | Implementation |
|---|---|
| Use the employee database as the source for role and region | `lookup_employee_identity` reads the employee row; unrecognized or invalid roles do not receive default access. |
| Bind a conversation session to its first employee | Optional `session_id` values on `/search` are persisted in `assistant_sessions`; another employee receives 403. |
| Filter unreadable documents before model exposure | `/search` passes role, region, and date predicates into the retriever before it returns results. |
| Restrict employee leave data | Leave tools authorize self, manager/direct-report, and HR access in each operation; HR reads are one target per call. |
| Give the same denial for unauthorized and nonexistent targets | Leave tools return a fixed refusal for both cases. |
| Protect personal data | Employee-record projection masks email, omits phone, and includes names only for authorized viewers. Leave-tool output contains no names, emails, or phone numbers. |
| Treat prompt/document content as untrusted | A user-message guard returns refusal decisions for common override, hidden-data, and bulk-data patterns. Retrieved document text must remain evidence only. |
| Integrate safety decisions into `/chat` | Pending Task 5; no `/chat` endpoint currently consumes the guard. |

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
  reason, and a fixed refusal message for use by a future chat handler.

### `src/veltrix_assistant/security/privacy.py`

- `mask_email` formats a valid address as
  `first-character***@domain`; malformed values are masked as `***`.
- `sanitize_employee_record` returns `None` for unauthorized records. It
  includes a name only for the employee themself, their direct manager, or HR;
  masks an included email; and never copies a phone field. Its
  non-personal-field allowlist avoids returning unrequested database fields.
- The helper is an integration boundary for future profile/answer workflows;
  `/search` does not return employee profiles and therefore does not call it.

### `src/veltrix_assistant/security/guards.py`

- `assess_user_message` examines the user message, not retrieved document
  contents. It returns `route="refuse"` for detected prompt-override,
  hidden-data, or bulk employee-data patterns, with a fixed refusal message.
- Ordinary HR questions return `route="continue"`.
- This is a deterministic defense-in-depth check, not a complete natural
  language security classifier. Authorization must still be enforced at
  retrieval and inside each tool. Task 5 must run the guard before retrieval
  and tool selection and stop processing when it returns `refuse`.
- Retrieved documents are untrusted evidence. Their contents must never be
  treated as instructions, used to change identity or permissions, or allowed
  to invoke tools.

### `src/veltrix_assistant/schemas/api.py`

`SearchRequest` accepts an optional Pydantic-validated `session_id` (trimmed,
non-empty, at most 128 characters). Existing clients that omit the field
continue to work.

### `src/veltrix_assistant/api.py`

- On startup, after corpus ingestion, the API initializes the session-binding
  table. If that initialization or ingestion fails, readiness stays unavailable
  and the failure is logged without employee personal data.
- `POST /search` requires `X-Employee-Id`, resolves its identity from the
  employee database, and returns a generic 401 for a missing or unknown
  employee. Invalid directory data fails closed as a 503.
- When `session_id` is supplied, `/search` binds or verifies ownership after
  identity lookup. Reuse by another employee returns 403 without changing the
  owner.
- The role/region audience and reference date are used in retrieval predicates:
  current results must be readable and effective; upcoming candidates must be
  readable and future-dated. These filters are supplied to BM25 before ranking,
  so restricted text is not returned as a result.
- `/search` is a retrieval diagnostic endpoint, not the assessment's `/chat`
  endpoint. It currently does not consume `assess_user_message`.

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
   returning evidence. Tool calls separately re-check the requester's database
   role and target scope.
4. Any employee profile intended for display or model context must first pass
   through `sanitize_employee_record`; only the minimum authorized fields
   should be sent onward.
5. In the future `/chat` workflow, `assess_user_message` must run before
   retrieval or tool selection. A refusal must stop the workflow. Only
   authorized evidence and the minimum necessary data should be passed to a
   model.

The employee ID header is an identity lookup key, not by itself a cryptographic
authentication mechanism. A deployed service must ensure that the header is
provided by a trusted authentication layer and cannot be freely asserted by an
untrusted client.

## Validation

The Task 4 behavior is covered by:

- `tests/test_security.py` — database identity resolution, invalid roles,
  persistent session ownership, safe employee-record projection, email
  masking, phone omission, and message-guard decisions.
- `tests/test_api.py` — required employee header, generic unknown-ID rejection,
  role/region document filtering, session ownership enforcement, and session
  ID validation.
- `tests/test_leave_tools.py` — employee/manager/HR scopes, direct-report team
  access, and the same refusal for unauthorized and nonexistent employee IDs.

Run the complete project test suite from the repository root:

```powershell
uv run pytest -q
```

At the time this guide was written, all 60 tests passed. The chat guard's
end-to-end route behavior remains untested until Task 5 implements and wires
`POST /chat`.
