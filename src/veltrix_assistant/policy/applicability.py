"""Deterministic document eligibility, precedence, and upcoming-change logic."""

from __future__ import annotations

from datetime import date

from veltrix_assistant.schemas.documents import DocumentMetadata
from veltrix_assistant.schemas.policy import (
    DocumentAudience,
    PolicyEvidence,
    PolicyResolution,
    Role,
)

_ROLE_ACCESS: dict[Role, frozenset[str]] = {
    "employee": frozenset({"all"}),
    "manager": frozenset({"all", "managers"}),
    "hr": frozenset({"all", "managers", "hr"}),
}
_AUTHORITY_RANK: dict[str, int] = {
    "faq": 0,
    "wiki": 0,
    "policy": 1,
    "addendum": 1,
    "memo": 1,
    "handbook": 1,
    "calendar": 1,
}


def parse_role(value: object) -> Role | None:
    """Validate an untrusted employee-directory role without coercion."""
    if value == "employee":
        return "employee"
    if value == "manager":
        return "manager"
    if value == "hr":
        return "hr"
    return None


def is_document_readable(
    document: DocumentMetadata,
    audience: DocumentAudience,
) -> bool:
    """Return whether role and region permit reading this document at any date."""
    allowed_access = _ROLE_ACCESS[audience.role]
    matches_region = (
        "GLOBAL" in document.regions or audience.region in document.regions
    )
    return document.access_level in allowed_access and matches_region


def is_document_applicable(
    document: DocumentMetadata,
    audience: DocumentAudience,
    reference_date: date,
) -> bool:
    """Check role, region, and inclusive effective-date eligibility."""
    return (
        is_document_readable(document, audience)
        and document.effective_date <= reference_date
    )


def get_applicable_documents(
    documents: tuple[DocumentMetadata, ...] | list[DocumentMetadata],
    audience: DocumentAudience,
    reference_date: date,
) -> tuple[DocumentMetadata, ...]:
    """Return all documents the requester may use on the reference date."""
    return tuple(
        document
        for document in documents
        if is_document_applicable(document, audience, reference_date)
    )


def document_precedence_key(
    document: DocumentMetadata,
    audience: DocumentAudience,
) -> tuple[int, int, date]:
    """Return ordered authority, scope, and recency precedence for one rule."""
    authority = _AUTHORITY_RANK[document.doc_type]
    is_region_specific = (
        audience.region in document.regions and "GLOBAL" not in document.regions
    )
    scope = int(is_region_specific)
    return authority, scope, document.effective_date


def select_precedence_winners(
    documents: tuple[DocumentMetadata, ...] | list[DocumentMetadata],
    audience: DocumentAudience,
) -> tuple[DocumentMetadata, ...]:
    """Keep all tied top documents for a single rule; never break a true tie."""
    if not documents:
        return ()

    best_key = max(document_precedence_key(document, audience) for document in documents)
    return tuple(
        sorted(
            (
                document
                for document in documents
                if document_precedence_key(document, audience) == best_key
            ),
            key=lambda document: document.doc_id,
        )
    )


def resolve_policy_evidence(
    evidence: tuple[PolicyEvidence, ...] | list[PolicyEvidence],
    audience: DocumentAudience,
    reference_date: date,
) -> tuple[PolicyResolution, ...]:
    """Resolve precedence independently per rule and preserve tied conflicts."""
    grouped: dict[str, list[PolicyEvidence]] = {}
    for item in evidence:
        if is_document_applicable(item.document, audience, reference_date):
            grouped.setdefault(item.rule_key, []).append(item)

    resolutions: list[PolicyResolution] = []
    for rule_key, candidates in sorted(grouped.items()):
        winners = select_precedence_winners(
            [item.document for item in candidates],
            audience,
        )
        winning_ids = {document.doc_id for document in winners}
        winning_evidence = tuple(
            sorted(
                (
                    item
                    for item in candidates
                    if item.document.doc_id in winning_ids
                ),
                key=lambda item: (item.document.doc_id, item.value),
            )
        )
        values = tuple(sorted({item.value for item in winning_evidence}))
        resolutions.append(
            PolicyResolution(
                rule_key=rule_key,
                values=values,
                evidence=winning_evidence,
                unresolved_conflict=len(values) > 1,
            )
        )

    return tuple(resolutions)


def get_upcoming_winners(
    documents: tuple[DocumentMetadata, ...] | list[DocumentMetadata],
    audience: DocumentAudience,
    reference_date: date,
) -> tuple[DocumentMetadata, ...]:
    """Find readable future documents that reach top precedence when effective.

    The input documents must be candidates for the same rule (for example,
    documents retrieved as evidence for one policy question). Documents tied at
    the top precedence are all retained so a caller can explain an unresolved
    conflict rather than silently selecting one.
    """
    readable = tuple(
        document for document in documents if is_document_readable(document, audience)
    )
    future_dates = sorted(
        {
            document.effective_date
            for document in readable
            if document.effective_date > reference_date
        }
    )

    upcoming: dict[str, DocumentMetadata] = {}
    for effective_date in future_dates:
        candidates_at_date = tuple(
            document
            for document in readable
            if document.effective_date <= effective_date
        )
        winners = select_precedence_winners(candidates_at_date, audience)
        for winner in winners:
            if winner.effective_date > reference_date:
                upcoming[winner.doc_id] = winner

    return tuple(
        sorted(
            upcoming.values(),
            key=lambda document: (document.effective_date, document.doc_id),
        )
    )
