"""Validated identity and request-safety contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from veltrix_assistant.schemas.policy import Role


class EmployeeIdentity(BaseModel):
    """Requester identity loaded from the employee directory, never message text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    employee_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    role: Role


class RequestSafetyDecision(BaseModel):
    """Deterministic guard result for refusal-prone user messages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: Literal["continue", "refuse"]
    reason: Literal["prompt_injection", "hidden_data", "bulk_employee_data"] | None = None
    message: str | None = None
