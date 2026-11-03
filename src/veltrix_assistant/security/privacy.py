"""Safe projection and masking helpers for employee personal data."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from veltrix_assistant.schemas.policy import Role

_NON_PERSONAL_FIELDS = (
    "employee_id",
    "region",
    "join_date",
    "level",
    "role",
    "manager_id",
)
PIIKind = Literal[
    "email",
    "phone",
    "employee_id",
    "government_id",
    "bank_account",
    "configured_sensitive_term",
]

_PII_PATTERNS: tuple[tuple[PIIKind, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", re.UNICODE)),
    ("employee_id", re.compile(r"\b[A-Z][0-9]{3,}\b", re.IGNORECASE)),
    (
        "government_id",
        re.compile(
            r"\b(?:\d{3}[- ]?\d{2}[- ]?\d{4}|"
            r"\d{4}[- ]?\d{4}[- ]?\d{4}|"
            r"[A-Z]{5}\d{4}[A-Z])\b",
            re.IGNORECASE,
        ),
    ),
    (
        "bank_account",
        re.compile(r"\b\d{10,18}\b", re.UNICODE),
    ),
    (
        "phone",
        re.compile(r"(?<!\w)\+?(?:\d[\s().-]*){9,}\d(?!\w)", re.UNICODE),
    ),
)

_REDACTION_LABELS: dict[PIIKind, str] = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "employee_id": "[REDACTED_EMPLOYEE_ID]",
    "government_id": "[REDACTED_GOVERNMENT_ID]",
    "bank_account": "[REDACTED_BANK_ACCOUNT]",
    "configured_sensitive_term": "[REDACTED_PERSONAL_NAME]",
}


@dataclass(frozen=True)
class PIIFinding:
    """A PII category and source span; never include the matched value."""

    kind: PIIKind
    start: int
    end: int


def mask_email(email: str) -> str:
    """Mask a valid email address as first-local-character + *** + domain."""
    local, separator, domain = email.strip().partition("@")
    if not separator or not local or not domain or "@" in domain:
        return "***"
    return f"{local[0]}***@{domain}"


def detect_pii(
    text: str,
    *,
    sensitive_terms: tuple[str, ...] = (),
) -> tuple[PIIFinding, ...]:
    """Find supported identifiers without returning or logging their values."""
    candidates: list[PIIFinding] = []
    for kind, pattern in _PII_PATTERNS:
        candidates.extend(
            PIIFinding(kind=kind, start=match.start(), end=match.end())
            for match in pattern.finditer(text)
        )
    for term in sensitive_terms:
        normalized_term = term.strip()
        if len(normalized_term) < 3:
            continue
        pattern = re.compile(
            rf"(?<!\w){re.escape(normalized_term)}(?!\w)",
            re.IGNORECASE | re.UNICODE,
        )
        candidates.extend(
            PIIFinding(
                kind="configured_sensitive_term",
                start=match.start(),
                end=match.end(),
            )
            for match in pattern.finditer(text)
        )

    priority = {
        "employee_id": 0,
        "government_id": 1,
        "phone": 2,
        "bank_account": 3,
        "email": 4,
        "configured_sensitive_term": 5,
    }
    selected: list[PIIFinding] = []
    for finding in sorted(
        candidates,
        key=lambda item: (
            item.start,
            priority[item.kind],
            -(item.end - item.start),
        ),
    ):
        if any(
            finding.start < existing.end and finding.end > existing.start
            for existing in selected
        ):
            continue
        selected.append(finding)
    return tuple(sorted(selected, key=lambda item: item.start))


def redact_pii(
    text: str,
    *,
    mode: Literal["redact", "mask"] = "redact",
    sensitive_terms: tuple[str, ...] = (),
) -> str:
    """Redact or mask known sensitive values before an external provider call."""
    findings = detect_pii(text, sensitive_terms=sensitive_terms)
    if not findings:
        return text

    output: list[str] = []
    cursor = 0
    for finding in findings:
        output.append(text[cursor : finding.start])
        label = _REDACTION_LABELS[finding.kind]
        if mode == "mask":
            label = f"[{finding.kind.upper()}]"
        output.append(label)
        cursor = finding.end
    output.append(text[cursor:])
    return "".join(output)


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
