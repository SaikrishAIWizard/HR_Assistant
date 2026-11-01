"""Validated tool-call result contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


ToolStatus = Literal["success", "error"]


class LeaveToolResult(BaseModel):
    """Tool output includes a machine-readable status and specific failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    args: dict[str, object]
    status: ToolStatus
    data: dict[str, object] | None = None
    error: str | None = None
