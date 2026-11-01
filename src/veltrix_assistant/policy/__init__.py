"""Document eligibility and precedence rules."""

from veltrix_assistant.policy.applicability import (
    document_precedence_key,
    get_applicable_documents,
    get_upcoming_winners,
    is_document_applicable,
    is_document_readable,
    parse_role,
    resolve_policy_evidence,
    select_precedence_winners,
)
from veltrix_assistant.schemas.policy import (
    DocumentAudience,
    PolicyEvidence,
    PolicyResolution,
    Role,
)

__all__ = [
    "DocumentAudience",
    "PolicyEvidence",
    "PolicyResolution",
    "Role",
    "document_precedence_key",
    "get_applicable_documents",
    "get_upcoming_winners",
    "is_document_applicable",
    "is_document_readable",
    "parse_role",
    "resolve_policy_evidence",
    "select_precedence_winners",
]
