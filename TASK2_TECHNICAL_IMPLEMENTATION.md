# Task 2: Document Applicability and Precedence

## Summary

Task 2 makes policy-source selection deterministic. A document can be used
only when the employee's database role can read it, its region matches the
employee or is `GLOBAL`, and its effective date is on or before the question's
reference date. When evidence for the same rule conflicts, source authority
outranks regional specificity, which outranks recency. Documents tied at the
top rank remain together so the caller can report a conflict rather than
silently choosing one.

The reference date is supplied on a search request when asking about a past or
future date. Otherwise, the API uses `APP_TODAY` when configured and falls back
to the system date.

## Module-by-module

### `src/veltrix_assistant/policy/applicability.py`

- `DocumentAudience` carries the employee's trusted region and role. The API
  loads those values from the employee database; they are not accepted from
  search request JSON.
- `is_document_readable` applies role access and region scope independently
  from date. `is_document_applicable` adds the inclusive check
  `effective_date <= reference_date`.
  When a newer applicable manifest entry explicitly lists an older document in
  `supersedes_doc_ids`, retrieval excludes that older document for the current
  reference date. Historical queries still retain the older policy when the
  replacement was not yet effective.
- `document_precedence_key` returns a tuple ordered by authority, region
  specificity, and effective date. Official sources (`policy`, `addendum`,
  `memo`, `handbook`, and `calendar`) outrank `faq` and `wiki`; a region-only
  document outranks `GLOBAL`; and a later effective date wins after the first
  two comparisons.
- `select_precedence_winners` returns all documents with the highest key.
  It does not invent a tie-breaker such as document ID.
- `PolicyEvidence` and `resolve_policy_evidence` resolve explicitly grouped
  evidence by `rule_key`. Only applicable evidence participates. Lower-ranked
  values are excluded; multiple distinct values among tied winners are
  returned with their source metadata and locator and
  `unresolved_conflict=True`, allowing both values to be cited.
- `get_upcoming_winners` checks readable future documents at each candidate's
  effective date against the same precedence rules. A future document that
  would lose (for example, a global update against a regional addendum) is not
  considered a winning change. Equal-ranked candidates are preserved.

Precedence and upcoming checks operate on documents already identified as
evidence for the same rule. The current retrieval API does not extract policy
facts or semantically group arbitrary chunks into rule keys; that fact
extraction and final natural-language conflict response belong in the later
conversation workflow.

### `src/veltrix_assistant/api.py`

`POST /search` uses the database-backed employee identity and reference date to
filter eligible documents *before* BM25 ranking. It separately retrieves
readable future sources, checks whether they would win precedence, and returns
them in the `upcoming` array. The response includes the effective
`reference_date`; future sources are not mixed into current `results`.

This remains a retrieval test endpoint. It does not generate an answer or
extract competing policy values. The pure `resolve_policy_evidence` function
is the deterministic integration point for that later answer workflow.

## Validation

`tests/test_applicability.py` covers role, region, effective date, authority,
regional scope, recency, unresolved ties with conflicting values, and upcoming
documents that do or do not win. `tests/test_api.py` covers current versus
upcoming response separation and employee-specific region/access filtering.
