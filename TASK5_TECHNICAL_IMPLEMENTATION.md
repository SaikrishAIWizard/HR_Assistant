# Task 5: Conversation Behavior

## Summary

`POST /chat` implements the assessment's request/response contract and routes
each message to exactly one of `rag`, `tool`, `hybrid`, `clarify`, `no_answer`,
or `refuse`. The route logic is deterministic: it uses the trusted employee
identity, the Task 4 safety guard, the access-filtered BM25 retriever, and the
existing leave tools. Write actions are validated without mutation, persisted
as one pending proposal per session, and executed only after a clear
confirmation in that same employee-owned session.

When `LLM_PROVIDER=openai`, RAG uses the OpenAI chat API with temperature 0.
Only role/region/date-filtered evidence is passed; the system prompt contains no
corpus facts, and every call is limited to 3,000 corpus-text tokens and logged.
The response schema validates citation IDs and locators against the supplied
evidence. The optional `LLM_PROVIDER=none` mode returns labeled source excerpts
without calling an external model. Leave tools and application code remain the
authority for permissions and writes in either mode.

## API contract

### `POST /chat`

Required header: `X-Employee-Id`

Request:

```json
{
  "session_id": "employee-selected-session-id",
  "message": "What does the annual leave policy say?"
}
```

Response:

```json
{
  "request_id": "unique-request-id",
  "answer": "Relevant policy evidence ...",
  "route": "rag",
  "citations": [
    {"doc_id": "POLICY-ID", "locator": "Section"}
  ],
  "tool_calls": [],
  "pending_confirmation": false
}
```

The request model rejects blank or oversized session IDs/messages and unknown
fields. Invalid chat bodies return HTTP 400. Missing or unknown employee IDs
return 401, and reuse of a session by a different employee returns 403.
Refusals are normal HTTP 200 responses with route `refuse`.

### Route contract

| Route | Behavior | Citations | Tool calls |
|---|---|---|---|
| `rag` | Return relevant, currently applicable document evidence. | At least one | Empty |
| `tool` | Return an authorized tool result, tool error, or pending write proposal. | Empty | At least one |
| `hybrid` | Return both eligible document evidence and a tool result. | At least one | At least one |
| `clarify` | Ask for all required missing or ambiguous details in one message. | Empty | Empty |
| `no_answer` | State that eligible documents do not cover the in-scope question and suggest HR. | Empty | Empty |
| `refuse` | Refuse an out-of-scope, unauthorized, bulk-data, or manipulation request. | Empty | Empty |

Every response has one route, a generated request ID, citation and tool-call
arrays, and a boolean `pending_confirmation`. Tool-call records use only the
contract's fields: `name`, `args`, and `status`.

## Implementation

### `src/veltrix_assistant/schemas/chat.py`

- Defines the Pydantic request, response, citation, tool-call, and pending
  action models.
- Restricts routes and tool-call statuses to the assessment's enumerated
  values. A session ID is trimmed, required, and limited to 128 characters.

### `src/veltrix_assistant/conversation.py`

- `respond_to_message` consumes the single pending action for the session
  before handling the new message. A clear confirmation executes it;
  cancellation discards it; any other message discards it and is handled as a
  new request.
- The Task 4 message guard runs before retrieval and tool selection. Bulk
  employee-data, hidden-data, and common instruction-override requests return
  `refuse` with no citations or calls. Out-of-scope requests are also refused.
- HR-related questions retrieve only documents that pass role, region, and
  effective-date checks. The output labels passages as source evidence and
  returns matching document IDs and locators as citations. Retrieved content
  never selects or triggers a tool.
- A relevant-query-token overlap check avoids treating a generic match on
  words such as “leave” as evidence for unrelated specifics. This is a
  deterministic fallback, not a semantic model; it can conservatively return
  `no_answer` when wording differs from the corpus.
- Balance, request-list, and team-leave questions call the existing read tools.
  Employee targets are authorized before a visible tool call; the tool checks
  access again while reading. Cross-employee denials use the same fixed
  message whether a target exists or not.
- Future annual-balance questions call the current-balance tool and retrieve
  accrual evidence. For an authorized target, the model receives only the
  current balance, join date, and target date needed for the estimate. Offline
  mode reports the current balance and explicitly does not project.
- Apply and cancellation intents call read-only preview validation first. A
  rejected action returns route `tool` with status `error` and is not proposed.
  A valid action returns a summary and a proposed call with status
  `pending_confirmation`; no balance or request row changes at this point.
- `hybrid` is returned when an authorized tool result and eligible document
  evidence both contribute to the answer.

### `src/veltrix_assistant/security/identity.py`

The assistant-owned `assistant_pending_actions` table stores a single JSON
proposal per session. `store_pending` atomically replaces the previous
proposal. `consume_pending` uses a SQLite write transaction to read and delete
it once, which prevents a second concurrent confirmation from executing the
same proposal. Pending state is consumed only after the session is bound to
the authenticated owner.

### `src/veltrix_assistant/tools/leave.py`

- `preview_apply_leave` performs the same date, calendar, overlap, balance, and
  requester checks as execution but makes no changes. The validation logic is
  shared with `apply_leave`.
- `preview_cancel_leave` verifies that the request belongs to the requester,
  is pending or approved, and starts after today. It makes no changes.
- On confirmation, the existing `apply_leave` or `cancel_leave` methods run
  their transactional checks again. Changed balances, overlaps, statuses, or
  request dates can therefore invalidate a previously valid proposal.
- `can_access_employee_data` performs a preflight authorization check for
  employee, direct-report manager, or HR scope; the actual read tool retains
  its own authorization check.

### `src/veltrix_assistant/api.py`

- Exposes `POST /chat` using the Pydantic response model and the required
  `X-Employee-Id` header.
- Binds or verifies `session_id` before using persistent session state.
- Maps invalid chat request bodies to HTTP 400 while preserving `/search`'s
  existing validation behavior.
- Keeps `/search` as a retrieval diagnostic endpoint and `GET /health` as the
  readiness check.

## Route selection and write confirmation

1. Authenticate the employee ID from the database and bind the session.
2. Consume any pending proposal. Only an explicit confirmation such as “yes,”
   “confirm,” or “go ahead” executes the stored action; “no” or “cancel”
   discards it. A different request discards it and continues through normal
   routing.
3. Apply the safety guard. Refuse on detected manipulation or bulk-data
   requests before retrieval or tools.
4. Determine whether the request is in scope, needs one read/write tool, and/or
   needs policy evidence.
5. If required information is absent, ask for all missing details in one
   `clarify` response. Do not call tools for clarification.
6. For a write, validate first. On error, return a `tool` error and do not
   create pending state. Otherwise persist the action and ask for confirmation.
7. On confirmation, re-run the tool's validations and execute its transaction.
   At most one proposal exists per session.
8. For document questions, filter eligible documents before BM25 ranking. If
   no relevant eligible evidence is found, return `no_answer` rather than
   guessing.

The implementation recognizes ISO dates and common month-name dates, with an
omitted year interpreted as 2026. “Next week” is Monday through Friday of the
following week. “Next Friday” is the first Friday after today. “Today”,
“tomorrow”, and explicit date ranges are supported. Past or unparsed dates are
clarified rather than silently shifted. Leave application constraints such as
the supported calendar year are still checked by the leave tool before a
proposal can be created.

## Validation

Run the full test suite:

```powershell
uv run pytest -q
```

`tests/test_api.py` covers route response shapes, RAG/no-answer/refusal,
invalid-body and identity HTTP codes, authorized read tools, hybrid answers,
clarification, write prevalidation, proposal immutability, confirmation,
cancellation, changed proposals, session ownership, and pending-action
discard behavior. `tests/test_leave_tools.py` continues to cover the
transactional and permission rules enforced again at execution time.
