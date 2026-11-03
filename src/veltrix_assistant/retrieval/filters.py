"""Shared metadata filtering for BM25 and vector retrieval."""

from __future__ import annotations

from collections.abc import Mapping

from veltrix_assistant.policy.applicability import parse_role
from veltrix_assistant.schemas.documents import DocumentMetadata


def metadata_matches(
    document: DocumentMetadata,
    filters: Mapping[str, object] | None,
) -> bool:
    """Apply optional metadata narrowing without replacing authorization checks."""
    if not filters:
        return True

    doc_type = filters.get("doc_type")
    if doc_type is not None and document.doc_type != doc_type:
        return False
    doc_ids = filters.get("doc_ids")
    if doc_ids is not None and document.doc_id not in doc_ids:
        return False

    policy_scope = filters.get("policy_scope")
    actual_scope = "GLOBAL" if "GLOBAL" in document.regions else "REGIONAL"
    if policy_scope is not None and actual_scope != policy_scope:
        return False
    authority = filters.get("authority")
    actual_authority = (
        "reference"
        if document.doc_type in {"faq", "wiki"}
        else "authoritative"
    )
    if authority is not None and actual_authority != authority:
        return False

    region = filters.get("region")
    if region is not None and region not in document.regions and "GLOBAL" not in document.regions:
        return False
    role = parse_role(filters.get("role"))
    if role is not None:
        allowed_access = {
            "employee": {"all"},
            "manager": {"all", "managers"},
            "hr": {"all", "managers", "hr"},
        }
        if document.access_level not in allowed_access[role]:
            return False
    elif filters.get("role") is not None:
        return False

    effective_on_or_before = filters.get("effective_on_or_before")
    if effective_on_or_before is not None:
        if not hasattr(effective_on_or_before, "isoformat"):
            return False
        if document.effective_date > effective_on_or_before:
            return False
    return True
