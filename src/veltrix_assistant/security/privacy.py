"""Safe projection and masking helpers for employee personal data."""

from __future__ import annotations

from collections.abc import Mapping

from veltrix_assistant.schemas.policy import Role

_NON_PERSONAL_FIELDS = (
    "employee_id",
    "region",
    "join_date",
    "level",
    "role",
    "manager_id",
)


def mask_email(email: str) -> str:
    """Mask a valid email address as first-local-character + *** + domain."""
    local, separator, domain = email.strip().partition("@")
    if not separator or not local or not domain or "@" in domain:
        return "***"
    return f"{local[0]}***@{domain}"


def sanitize_employee_record(
    record: Mapping[str, object],
    *,
    requester_id: str,
    requester_role: Role,
) -> dict[str, str] | None:
    """Project a permitted single-employee view, masking email and omitting phone."""
    employee_id = record.get("employee_id")
    manager_id = record.get("manager_id")
    if not isinstance(employee_id, str):
        return None
    may_view = (
        requester_id == employee_id
        or requester_role == "hr"
        or (
            requester_role == "manager"
            and manager_id == requester_id
        )
    )
    if not may_view:
        return None

    safe_record = {
        field: value
        for field in _NON_PERSONAL_FIELDS
        if isinstance((value := record.get(field)), str)
    }
    full_name = record.get("full_name")
    if isinstance(full_name, str):
        safe_record["full_name"] = full_name
    email = record.get("email")
    if isinstance(email, str):
        safe_record["email"] = mask_email(email)
    return safe_record
