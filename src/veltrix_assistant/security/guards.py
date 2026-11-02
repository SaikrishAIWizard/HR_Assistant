"""Conservative deterministic guards for bulk-data and instruction-override asks."""

from __future__ import annotations

import re

from veltrix_assistant.schemas.security import RequestSafetyDecision


_PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"\bignore\b.{0,50}\b(previous|prior|above|system|developer)\b.{0,30}\binstructions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(reveal|show|print|repeat|output|disclose)\b.{0,50}"
        r"\b(system|developer|hidden|secret|confidential)\b.{0,30}"
        r"\b(prompt|instructions?|data|information)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(pretend|act)\b.{0,30}\b(as|to be)\b.{0,20}\b(hr|administrator|admin)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(i am|i'm|i work in)\b.{0,30}\b(hr|administrator|admin)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(bypass|override)\b.{0,30}\b(access|security|privacy|rules?)\b", re.IGNORECASE),
    re.compile(r"\bhidden data\b|\bsecret prompt\b", re.IGNORECASE),
    re.compile(r"\b(system|developer) prompt\b", re.IGNORECASE),
    re.compile(
        r"\b(give|tell|show|repeat|reveal|print|output)\b.{0,30}"
        r"\b(system|developer) prompt\b",
        re.IGNORECASE,
    ),
)
_BULK_EMPLOYEE_DATA_PATTERN = re.compile(
    r"\b(everyone|everybody|every employee|all employees|all staff|all workers|"
    r"entire workforce|whole company|company[- ]wide|each employee)\b"
    r".{0,80}\b(balance|leave|request|salary|email|phone|contact|employee data)\b"
    r"|\b(show|list|export|give|send)\b.{0,50}"
    r"\b(all employees|everyone|all staff|entire workforce|whole company)\b"
    r".{0,60}\b(balance|leave|request|salary|email|phone|contact|data)\b",
    re.IGNORECASE,
)


def assess_user_message(message: str) -> RequestSafetyDecision:
    """Flag requests that must be refused before retrieval or tool selection."""
    if _BULK_EMPLOYEE_DATA_PATTERN.search(message):
        return RequestSafetyDecision(
            route="refuse",
            reason="bulk_employee_data",
            message="I can't share leave or personal data about multiple employees.",
        )
    if any(pattern.search(message) for pattern in _PROMPT_INJECTION_PATTERNS):
        reason = (
            "hidden_data"
            if re.search(r"\b(hidden|secret|confidential)\b", message, re.IGNORECASE)
            else "prompt_injection"
        )
        return RequestSafetyDecision(
            route="refuse",
            reason=reason,
            message="I can't help override access controls or reveal hidden instructions or data.",
        )
    return RequestSafetyDecision(route="continue")
