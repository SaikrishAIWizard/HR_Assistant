"""Validated audience and policy-evidence contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from veltrix_assistant.schemas.documents import DocumentMetadata


Role = Literal["employee", "manager", "hr"]


class DocumentAudience(BaseModel):
    """Database-backed region and role used to decide document eligibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    region: str = Field(min_length=1)
    role: Role


class PolicyEvidence(BaseModel):
    """A rule value extracted from a source and tied to its source metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    document: DocumentMetadata
    locator: str = Field(min_length=1)


class PolicyResolution(BaseModel):
    """Winning evidence for one rule, retaining conflicting top-ranked values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_key: str
    values: tuple[str, ...]
    evidence: tuple[PolicyEvidence, ...]
    unresolved_conflict: bool
