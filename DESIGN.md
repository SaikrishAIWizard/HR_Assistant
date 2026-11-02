# Security and privacy design

## Trusted identity and session ownership

- Every protected API request requires `X-Employee-Id`. The value is looked up
  in the configured SQLite `employees` table; role and region are never read
  from request-body fields or user claims.
- Unknown employee IDs receive the same generic 401 response. Invalid or
  unavailable employee-directory records fail closed rather than receiving
  default permissions.
- The optional `session_id` on `POST /search` is stored in the assistant-owned
  `assistant_sessions` table. Its first authenticated employee becomes its
  owner. Reuse by another employee returns 403, and the original ownership is
  not overwritten. The table is created without changing assessment-owned
  tables or columns.

## Document and employee-data authorization

- Role, region, effective date, and document audience are checked by the
  retrieval filter before BM25 ranks or returns content. Results therefore
  do not reveal restricted document IDs or text to an unauthorized requester.
- Leave tools are bound to the authenticated employee and re-check self,
  manager/direct-report, and HR access against the database for each operation.
  Unauthorized and nonexistent employee targets receive the same fixed
  refusal.
- Employee records should be projected through
  `sanitize_employee_record` before display or model use. It returns no record
  to unauthorized requesters, exposes names only to the employee, their
  manager, or HR, masks email to `first-character***@domain`, and never includes
  phone numbers. Leave tools do not return employee names, email addresses, or
  phone numbers.
- Logs must contain operational status and aggregate counts only. Do not log
  employee names, email addresses, phone numbers, request bodies, or retrieved
  document text.

## Untrusted content and refusal boundary

- User-provided messages are classified by `assess_user_message`. Bulk
  employee-data requests and common instruction-override/hidden-data requests
  produce a fixed refusal decision (`route="refuse"`); ordinary HR questions
  continue.
- Retrieved documents are evidence only. Document text must never select or
  invoke tools, change identity/permissions, or override system/developer
  instructions.
- The repository currently implements `/search` as a retrieval diagnostic
  endpoint, not the assessment's `/chat` endpoint. Requirement 5 must invoke
  the message guard before retrieval and tool selection, stop on a refusal,
  and pass only authorized, relevant evidence and minimum necessary data to a
  model. The guard is a deterministic defense-in-depth check, not a substitute
  for authorization checks in each tool.
