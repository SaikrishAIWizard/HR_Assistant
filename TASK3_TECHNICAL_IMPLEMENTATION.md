# Task 3: Leave Tools and Rules

## Summary

Task 3 implements the five required callable tools and enforces access inside
their code using a request-scoped authenticated identity. Leave requests use
inclusive date ranges, count weekdays while excluding public holidays parsed
from the ingested applicable regional calendar, and update request and balance
records in a single SQLite transaction. Existing database tables are used as
provided; no existing schema is modified.

## Tool names and arguments

`LeaveTools.get_tool_functions()` exposes exactly these names and parameters:

| Tool | Arguments |
|---|---|
| `get_leave_balance` | `employee_id`, optional `leave_type` |
| `list_leave_requests` | `employee_id`, optional `status` |
| `get_team_leave` | `start_date`, `end_date` |
| `apply_leave` | `leave_type`, `start_date`, `end_date` |
| `cancel_leave` | `request_id` |

Every call returns a `LeaveToolResult` with the exact tool name, arguments,
`success` or `error` status, and either data or a specific error message.
Rejected writes return `error` and leave the database unchanged.

## Module-by-module

### `src/veltrix_assistant/tools/leave.py`

- `LeaveToolFactory` retains the configured SQLite database, ingested chunks,
  and reference date. The chat handler can call
  `app.state.leave_tool_factory.for_employee(x_employee_id)` to create tools
  bound to the identity authenticated by the request header. Employee IDs
  passed to read tools are checked against this bound identity and the current
  database role; they do not establish identity.
- Employees may read their own leave data; managers may read their own and
  direct reports' data; HR may read one employee at a time. Only managers may
  use `get_team_leave`, which returns overlapping pending/approved requests
  from direct reports. Unavailable and unauthorized target IDs receive the
  same fixed refusal. Read results expose no names, emails, or phone numbers.
- `get_public_holidays` chooses the applicable 2026 calendar using region,
  effective date, and document precedence, then parses dates from the
  ingested public/paid-holiday sections. Restricted-holiday sections are
  explicitly excluded. Missing or malformed applicable calendar data returns
  a specific tool error; there is no hand-maintained fallback list.
- `count_working_days` counts the inclusive range's Monday-Friday dates except
  for those public holidays. Weekends and holidays are not counted.
- `apply_leave` requires `annual` or `sick`, limits both dates to 2026, rejects
  past starts, reversed ranges, zero working days, insufficient balances, and
  any date overlap with the requester's pending or approved leave. It begins
  an immediate SQLite transaction, subtracts the balance, inserts a pending
  request with a UUID4 ID and working-day count, and commits both changes
  together. ISO `YYYY-MM-DD` strings from JSON tool arguments are parsed
  explicitly. A failed insert rolls back the balance debit.
- `cancel_leave` only accepts the authenticated requester's pending/approved
  request when its start date is strictly after today. It marks the request
  cancelled and restores its recorded working days in one transaction. If
  either update fails, both changes roll back.
- `get_leave_balance` reports stored balances only; it does not calculate
  future accrual. For a future-balance question, Task 5 retrieves the current
  annual balance and join date separately, then combines them with applicable
  accrual evidence through the configured LLM. Request and team queries include
  only the fields needed by the caller.

### `src/veltrix_assistant/api.py`

After ingestion, the API stores a `LeaveToolFactory` in app state. The tools
are not exposed through separate HTTP routes because the assessment expects
tool calls to be orchestrated by `POST /chat`. The chat handler creates the
per-request tools using the authenticated `X-Employee-Id`. Task 5 validates
write proposals without mutation, requests confirmation, and only then invokes
the write tool, which re-checks its rules inside the database transaction.

## Validation

`tests/test_leave_tools.py` covers exact tool names/signatures, extracted
regional holidays, restricted holidays, inclusive weekdays, date/type
boundaries, balance and overlap rejection, role permissions, team leave,
request creation, cancellation, and rollback when either side of a write
transaction fails.
