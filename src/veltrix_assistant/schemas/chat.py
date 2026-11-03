"""Validated conversation API and session-state contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.types import StringConstraints
from pydantic.types import StringConstraints


SessionId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
ChatMessage = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
]


class ChatRequest(BaseModel):
    """Request body for one employee chat turn."""

    model_config = ConfigDict(extra="forbid")

    session_id: SessionId
    message: ChatMessage


ToolCallStatus = Literal["pending_confirmation", "success", "error"]
ChatRoute = Literal["rag", "tool", "hybrid", "clarify", "no_answer", "refuse"]


class ChatCitation(BaseModel):
    """Document citation exposed in a chat response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    locator: str


class ChatToolCall(BaseModel):
    """One tool invocation or proposed write action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    args: dict[str, object]
    status: ToolCallStatus


class ChatResponse(BaseModel):
    """The exact response contract for POST /chat."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    answer: str
    route: ChatRoute
    citations: list[ChatCitation] = Field(default_factory=list)
    tool_calls: list[ChatToolCall] = Field(default_factory=list)
    pending_confirmation: bool = False


class PendingAction(BaseModel):
    """The single write proposal awaiting confirmation in a session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    employee_id: str = Field(min_length=1)
    name: Literal["apply_leave", "cancel_leave"]
    args: dict[str, object]
    summary: str = Field(min_length=1)


class PendingClarification(BaseModel):
    """The details already provided for a leave request awaiting clarification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    employee_id: str = Field(min_length=1)
    intent: Literal["apply_leave"]
    leave_type: Literal["annual", "sick"] | None = None
    start_date: str | None = None
    end_date: str | None = None
