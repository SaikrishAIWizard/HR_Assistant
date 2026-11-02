"""Deterministic chat routing, clarification, and confirmed leave actions."""

from __future__ import annotations

import re
import uuid
from calendar import month_abbr, month_name, monthrange
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Literal

from veltrix_assistant.policy.applicability import is_document_applicable
from veltrix_assistant.policy.applicability import (
    get_upcoming_winners,
    is_document_readable,
)
from veltrix_assistant.retrieval.bm25 import BM25Retriever
from veltrix_assistant.retrieval.context import build_corpus_context
from veltrix_assistant.llm import EvidenceBlock, LLMProvider
from veltrix_assistant.schemas.chat import (
    ChatRoute,
    ChatCitation,
    ChatResponse,
    ChatToolCall,
    PendingAction,
)
from veltrix_assistant.schemas.documents import RetrievedChunk
from veltrix_assistant.schemas.policy import DocumentAudience
from veltrix_assistant.schemas.security import EmployeeIdentity
from veltrix_assistant.schemas.tools import LeaveToolResult
from veltrix_assistant.security.guards import assess_user_message
from veltrix_assistant.security.identity import SessionBindingStore
from veltrix_assistant.tools.leave import LeaveTools


_EMPLOYEE_ID_PATTERN = re.compile(r"\b[A-Z][0-9]{3,}\b", re.IGNORECASE)
_REQUEST_ID_PATTERN = re.compile(
    r"\b(?:REQ-[A-Z0-9-]+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)
_ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_PATTERN = "(?:" + "|".join(
    sorted(
        {
            month
            for index in range(1, 13)
            for month in (month_name[index], month_abbr[index])
        },
        key=len,
        reverse=True,
    )
) + ")"
_DAY_RANGE_PATTERN = re.compile(
    r"\b(?:(?P<first_day>\d{1,2})\s*(?:to|through|-)\s*"
    rf"(?P<last_day>\d{{1,2}})\s+(?P<month_after>{_MONTH_PATTERN})"
    r"(?:\s+(?P<year_after>20\d{2}))?"
    rf"|(?P<month_before>{_MONTH_PATTERN})\s+(?P<first_before>\d{{1,2}})\s*"
    r"(?:to|through|-)\s*(?P<last_before>\d{1,2})"
    r"(?:,?\s*(?P<year_before>20\d{2}))?)\b"
    ,
    re.IGNORECASE,
)
_NAMED_DATE_PATTERN = re.compile(
    rf"\b(?:(?P<month>{_MONTH_PATTERN})\s+(?P<day>\d{{1,2}})"
    r"(?:,?\s*(?P<year>20\d{2}))?"
    rf"|(?P<day_first>\d{{1,2}})\s+(?P<month_first>{_MONTH_PATTERN})"
    r"(?:\s+(?P<year_first>20\d{2}))?)\b"
    ,
    re.IGNORECASE,
)
_CONFIRMATIONS = {
    "yes",
    "yes please",
    "confirm",
    "confirmed",
    "go ahead",
    "proceed",
    "do it",
    "approve",
}
_CANCELLATIONS = {
    "no",
    "no thanks",
    "cancel",
    "cancel it",
    "do not",
    "don't",
    "stop",
}
_HR_SCOPE_TERMS = (
    "leave",
    "holiday",
    "policy",
    "employee",
    "manager",
    "workplace",
    "sick",
    "annual",
    "parental",
    "bereavement",
    "carry forward",
    "accrual",
    "balance",
    "absence",
)
_POLICY_CUES = (
    "policy",
    "carry forward",
    "carry over",
    "accrual",
    "how do i",
    "what are the rules",
    "what is the rule",
    "according to",
    "eligible",
)
_OTHER_LEAVE_TYPES = (
    "bereavement",
    "compassionate",
    "maternity",
    "parental",
    "paternity",
    "unpaid",
)
_GENERIC_QUERY_TERMS = {
    "a", "about", "an", "and", "are", "can", "do", "does", "for", "how",
    "i", "in", "is", "it", "leave", "me", "my", "of", "on", "please",
    "policy", "tell", "the",
    "to", "what", "when", "where", "which", "who", "why", "with", "you",
}


def _response(
    answer: str,
    route: ChatRoute,
    *,
    citations: list[ChatCitation] | None = None,
    tool_calls: list[ChatToolCall] | None = None,
    pending_confirmation: bool = False,
) -> ChatResponse:
    return ChatResponse(
        request_id=str(uuid.uuid4()),
        answer=answer,
        route=route,
        citations=citations or [],
        tool_calls=tool_calls or [],
        pending_confirmation=pending_confirmation,
    )


def _tool_call(result: LeaveToolResult) -> ChatToolCall:
    status = "success" if result.status == "success" else "error"
    return ChatToolCall(name=result.name, args=result.args, status=status)


def _tool_response(result: LeaveToolResult) -> ChatResponse:
    if result.status == "error":
        answer = result.error or "The requested leave action could not be completed."
    else:
        answer = _format_tool_result(result)
    return _response(answer, "tool", tool_calls=[_tool_call(result)])


def _format_tool_result(result: LeaveToolResult) -> str:
    data = result.data or {}
    if result.name == "get_leave_balance":
        balances = data.get("balances")
        if not isinstance(balances, dict):
            return "The leave balance was retrieved."
        details = ", ".join(
            f"{leave_type}: {balance} days"
            for leave_type, balance in sorted(balances.items())
        )
        return f"The leave balance is {details}."
    if result.name == "list_leave_requests":
        requests = data.get("requests")
        if not isinstance(requests, list) or not requests:
            return "There are no leave requests for the authorized employee."
        entries = [
            _format_request_item(item, include_employee_id=False)
            for item in requests
            if isinstance(item, dict)
        ]
        return "Leave requests: " + "; ".join(entries)
    if result.name == "get_team_leave":
        requests = data.get("requests")
        if not isinstance(requests, list) or not requests:
            return "No overlapping leave was found for your direct reports."
        entries = [
            _format_request_item(item, include_employee_id=True)
            for item in requests
            if isinstance(item, dict)
        ]
        return "Overlapping direct-report leave: " + "; ".join(entries)
    if result.name == "cancel_leave":
        return (
            f"Leave request {data.get('request_id')} was cancelled; "
            f"{data.get('working_days_restored')} working day(s) were restored."
        )
    if result.name == "apply_leave":
        return (
            f"Leave request {data.get('request_id')} was submitted for "
            f"{data.get('working_days')} working day(s), "
            f"{data.get('start_date')} through {data.get('end_date')}."
        )
    return "The leave request was submitted."


def _format_request_item(
    item: dict[str, object],
    *,
    include_employee_id: bool,
) -> str:
    employee_prefix = (
        f"Employee {item.get('employee_id')}: "
        if include_employee_id
        else ""
    )
    return (
        f"{employee_prefix}{item.get('request_id')} "
        f"{item.get('leave_type')} from {item.get('start_date')} "
        f"through {item.get('end_date')} "
        f"({item.get('working_days')} working day(s), {item.get('status')})"
    )


def _month_number(month: str) -> int | None:
    return next(
        (
            index
            for index, name in enumerate(month_name)
            if index and (name.casefold() == month.casefold()
                          or name[:3].casefold() == month[:3].casefold())
        ),
        None,
    )


def _parse_month_date(month: str, day: str, year: str | None) -> date | None:
    month_value = _month_number(month)
    if month_value is None:
        return None
    try:
        return date(int(year or 2026), month_value, int(day))
    except ValueError:
        return None


def _extract_dates(message: str) -> tuple[date, ...] | None:
    dates: list[tuple[int, date]] = []
    iso_spans: list[tuple[int, int]] = []
    range_spans: list[tuple[int, int]] = []
    for match in _ISO_DATE_PATTERN.finditer(message):
        try:
            dates.append((match.start(), date.fromisoformat(match.group())))
            iso_spans.append(match.span())
        except ValueError:
            return None
    for match in _DAY_RANGE_PATTERN.finditer(message):
        groups = match.groupdict()
        month = groups["month_after"] or groups["month_before"]
        first_day = groups["first_day"] or groups["first_before"]
        last_day = groups["last_day"] or groups["last_before"]
        year = groups["year_after"] or groups["year_before"]
        if month is None or first_day is None or last_day is None:
            continue
        if _month_number(month) is None:
            continue
        first_date = _parse_month_date(month, first_day, year)
        last_date = _parse_month_date(month, last_day, year)
        if first_date is None or last_date is None:
            return None
        dates.extend(((match.start(), first_date), (match.end(), last_date)))
        range_spans.append(match.span())
    for match in _NAMED_DATE_PATTERN.finditer(message):
        if any(
            match.start() < iso_end and match.end() > iso_start
            for iso_start, iso_end in iso_spans
        ) or any(
            match.start() < range_end and match.end() > range_start
            for range_start, range_end in range_spans
        ):
            continue
        groups = match.groupdict()
        month = groups["month"] or groups["month_first"]
        day = groups["day"] or groups["day_first"]
        year = groups["year"] or groups["year_first"]
        if month is None or day is None or _month_number(month) is None:
            continue
        parsed = _parse_month_date(month, day, year)
        if parsed is None:
            return None
        dates.append((match.start(), parsed))
    dates.sort(key=lambda item: item[0])
    unique: list[date] = []
    for _, parsed in dates:
        if parsed not in unique:
            unique.append(parsed)
    return tuple(unique)


def _date_range(message: str, today: date) -> tuple[date, date] | None:
    lowered = message.casefold()
    if "next week" in lowered:
        next_monday = today + timedelta(days=7 - today.weekday())
        return next_monday, next_monday + timedelta(days=4)
    if "next friday" in lowered:
        days = (4 - today.weekday()) % 7 or 7
        next_friday = today + timedelta(days=days)
        return next_friday, next_friday
    if "tomorrow" in lowered:
        tomorrow = today + timedelta(days=1)
        return tomorrow, tomorrow
    if "day after tomorrow" in lowered:
        day_after = today + timedelta(days=2)
        return day_after, day_after
    if re.search(r"\btoday\b", lowered):
        return today, today

    dates = _extract_dates(message)
    if dates is None or not dates:
        return None
    if len(dates) == 1:
        return dates[0], dates[0]
    return dates[0], dates[1]


def _is_future_balance_question(message: str) -> bool:
    lowered = message.casefold()
    if not re.search(r"\bannual\s+leave\b", lowered):
        return False
    has_target = re.search(
        rf"\b(?:by|in|at the end of)\s+(?:the\s+)?"
        rf"(?:{_MONTH_PATTERN}|\d{{4}}-\d{{2}}-\d{{2}}|20\d{{2}})\b",
        lowered,
        re.IGNORECASE,
    )
    asks_future = re.search(r"\b(?:will|would)\b", lowered)
    asks_amount = re.search(r"\b(?:how many|how much)\b", lowered)
    asks_balance = re.search(r"\b(?:balance|days|have)\b", lowered)
    return bool(
        (asks_future and (asks_amount or asks_balance))
        or (has_target and asks_amount)
    )


def _projection_target_date(message: str) -> date | None:
    dates = _extract_dates(message)
    if dates:
        return dates[-1]
    lowered = message.casefold()
    if re.search(r"\b(?:by|at the end of)\s+(?:the\s+)?(?:year|year-end)\b", lowered):
        return date(2026, 12, 31)
    month_match = re.search(
        rf"\b(?:by|in|at the end of)\s+(?:the\s+)?"
        rf"(?P<month>{_MONTH_PATTERN})(?:\s+(?P<year>20\d{{2}}))?\b",
        message,
        re.IGNORECASE,
    )
    if month_match is not None:
        month = _month_number(month_match.group("month"))
        if month is not None:
            year = int(month_match.group("year") or "2026")
            return date(year, month, monthrange(year, month)[1])
    year_match = re.search(
        r"\b(?:by|at the end of)\s+(?:the\s+)?(?:year\s+)?(20\d{2})\b",
        lowered,
    )
    if year_match is not None:
        return date(int(year_match.group(1)), 12, 31)
    return None


def _is_past_date_range(date_range: tuple[date, date], today: date) -> bool:
    return date_range[0] < today


def _leave_type(message: str) -> str | None:
    matches = [
        kind
        for kind in ("annual", "sick")
        if re.search(rf"\b{kind}\b", message, re.IGNORECASE)
    ]
    return matches[0] if len(matches) == 1 else None


def _employee_target(message: str, identity: EmployeeIdentity) -> str | None:
    match = _EMPLOYEE_ID_PATTERN.search(message)
    if match:
        return match.group().upper()
    if re.search(r"\b(my|me|mine|own|i)\b", message, re.IGNORECASE):
        return identity.employee_id
    return None


def _request_status(message: str) -> str | None:
    statuses = [
        status
        for status in ("pending", "approved", "cancelled")
        if re.search(rf"\b{status}\b", message, re.IGNORECASE)
    ]
    return statuses[0] if len(statuses) == 1 else None


def _tool_intent(message: str) -> str | None:
    lowered = re.sub(r"[-_]", " ", message.casefold())
    if _is_future_balance_question(message):
        return "get_leave_balance"
    if re.search(r"\b(cancel|withdraw)\b.{0,50}\b(leave|request)\b", lowered):
        return "cancel_leave"
    if re.search(r"\b(team leave|team members off|who is off)\b", lowered):
        return "get_team_leave"
    if re.search(
        r"\b(apply|request|book|take|want|need)\b.{0,50}\b(annual|sick)?\s*leave\b",
        lowered,
    ) and not any(cue in lowered for cue in _POLICY_CUES):
        return "apply_leave"
    if (
        re.search(r"\b(balance|remaining|left)\b", lowered)
        and re.search(r"\b(leave|annual|sick|days)\b", lowered)
        and ("carry forward" not in lowered or "balance" in lowered)
    ):
        return "get_leave_balance"
    if re.search(r"\b(list|show|check|view)\b.{0,40}\b(my )?leave requests?\b", lowered):
        return "list_leave_requests"
    if re.search(r"\b(my|our) leave requests?\b", lowered):
        return "list_leave_requests"
    return None


def _needs_retrieval(message: str, intent: str | None) -> bool:
    lowered = re.sub(r"[-_]", " ", message.casefold())
    policy_question = any(cue in lowered for cue in _POLICY_CUES)
    if intent is not None:
        return policy_question or _is_future_balance_question(message)
    return any(term in lowered for term in _HR_SCOPE_TERMS)


def _has_relevant_query_term(message: str, chunk_text: str) -> bool:
    normalized_message = re.sub(
        r"(?<=\w)'s\b|[-_]",
        " ",
        message.casefold(),
    )
    query_terms = {
        term.casefold()
        for term in re.findall(r"[\w'-]+", normalized_message)
        if term.casefold() not in _GENERIC_QUERY_TERMS
        and not re.fullmatch(r"(?:19|20)\d{2}", term)
    }
    if not query_terms:
        return True
    corpus_terms = {
        term.casefold()
        for term in re.findall(r"[\w'-]+", chunk_text)
    }
    return bool(query_terms & corpus_terms)


def _policy_reference_date(message: str, today: date) -> date:
    """Resolve an explicit historical reference date for document questions."""
    lowered = message.casefold()
    past_reference = re.search(
        r"\b(?:was|were|historically|in|during|as of|as at|on|before)\b",
        lowered,
    )
    if past_reference is None:
        return today

    dates = _extract_dates(message)
    if dates is not None:
        past_dates = [value for value in dates if value < today]
        if past_dates:
            return past_dates[0]

    year_match = re.search(
        r"\b(?:in|during|as of|as at|for)\s+(?:the year\s+)?(20\d{2})\b",
        lowered,
    )
    if year_match is not None:
        year = int(year_match.group(1))
        if year < today.year:
            return date(year, 12, 31)
    return today


def _clarification_for_tool(
    intent: str,
    message: str,
    identity: EmployeeIdentity,
    today: date,
) -> tuple[str | None, dict[str, object] | None]:
    if intent in {"get_leave_balance", "list_leave_requests"}:
        employee_id = _employee_target(message, identity)
        if employee_id is None:
            return "Which employee ID should I use? You can say 'my' to refer to yourself.", None
        if intent == "get_leave_balance":
            leave_type = _leave_type(message)
            args: dict[str, object] = {"employee_id": employee_id}
            if leave_type is not None:
                args["leave_type"] = leave_type
            else:
                other_type = next(
                    (
                        kind
                        for kind in _OTHER_LEAVE_TYPES
                        if re.search(rf"\b{kind}\s+leave\b", message, re.IGNORECASE)
                    ),
                    None,
                )
                if other_type is not None:
                    args["leave_type"] = other_type
            return None, args
        status = _request_status(message)
        args = {"employee_id": employee_id}
        if status is not None:
            args["status"] = status
        return None, args

    if intent == "get_team_leave":
        date_range = _date_range(message, today)
        if date_range is None:
            return "What date range should I check? Please provide both start and end dates.", None
        if _is_past_date_range(date_range, today):
            return "That date range has already started or passed. What future date range should I check?", None
        return None, {
            "start_date": date_range[0].isoformat(),
            "end_date": date_range[1].isoformat(),
        }

    if intent == "apply_leave":
        missing: list[str] = []
        leave_type = _leave_type(message)
        if leave_type is None:
            missing.append("leave type (annual or sick)")
        date_range = _date_range(message, today)
        if date_range is None:
            missing.append("start and end dates")
        elif _is_past_date_range(date_range, today):
            return "That date has already passed. What future date or date range should I use?", None
        if missing:
            return f"Please provide the {', and '.join(missing)}.", None
        assert leave_type is not None and date_range is not None
        return None, {
            "leave_type": leave_type,
            "start_date": date_range[0].isoformat(),
            "end_date": date_range[1].isoformat(),
        }

    match = _REQUEST_ID_PATTERN.search(message)
    if match is None:
        return "Which future leave request should I cancel? Please provide its request ID.", None
    return None, {"request_id": match.group()}


def _is_confirmation(message: str) -> bool:
    normalized = " ".join(
        re.findall(r"[\w']+", message.casefold())
    ).replace("’", "'")
    return normalized in _CONFIRMATIONS | {
        "yes confirm",
        "yes please confirm",
        "yes i confirm",
        "yes go ahead",
        "yes proceed",
        "yes do it",
        "please proceed",
    }


def _is_cancellation(message: str) -> bool:
    normalized = " ".join(
        re.findall(r"[\w']+", message.casefold())
    ).replace("’", "'")
    return normalized in _CANCELLATIONS | {
        "no cancel it",
        "no please",
        "no do not",
        "do not proceed",
        "don't proceed",
        "i changed my mind",
        "no i changed my mind",
        "never mind",
        "nevermind",
    }


def _execute_pending(
    pending: PendingAction,
    tools: LeaveTools,
    identity: EmployeeIdentity,
) -> ChatResponse:
    if pending.employee_id != identity.employee_id:
        raise ValueError("Pending action identity does not match the authenticated employee.")
    if pending.name == "apply_leave":
        expected = {"leave_type", "start_date", "end_date"}
        if set(pending.args) != expected:
            raise ValueError("Stored leave proposal has invalid arguments.")
        result = tools.apply_leave(
            leave_type=str(pending.args["leave_type"]),
            start_date=str(pending.args["start_date"]),
            end_date=str(pending.args["end_date"]),
        )
    else:
        if set(pending.args) != {"request_id"}:
            raise ValueError("Stored cancellation proposal has invalid arguments.")
        result = tools.cancel_leave(request_id=str(pending.args["request_id"]))
    return _tool_response(result)


def _proposal_response(
    *,
    session_id: str,
    identity: EmployeeIdentity,
    store: SessionBindingStore,
    name: Literal["apply_leave", "cancel_leave"],
    args: dict[str, object],
    summary: str,
) -> ChatResponse:
    pending = PendingAction(
        employee_id=identity.employee_id,
        name=name,
        args=args,
        summary=summary,
    )
    store.store_pending(
        session_id,
        identity.employee_id,
        pending.model_dump_json(),
    )
    call = ChatToolCall(
        name=name,
        args=args,
        status="pending_confirmation",
    )
    return _response(
        f"{summary} Please confirm if you want me to proceed.",
        "tool",
        tool_calls=[call],
        pending_confirmation=True,
    )


def _format_retrieval_answer(
    matches: Sequence[RetrievedChunk],
    upcoming_matches: Sequence[RetrievedChunk],
    *,
    question: str,
    reference_date: date,
    llm_provider: LLMProvider | None,
    tool_facts: dict[str, object] | None = None,
) -> tuple[str, list[ChatCitation], bool]:
    combined_matches = (*matches, *upcoming_matches)
    context = build_corpus_context([match.chunk for match in combined_matches])
    metadata_by_block = {
        (
            match.chunk.document.doc_id,
            match.chunk.locator,
            match.chunk.text,
        ): match.chunk.document
        for match in combined_matches
    }
    upcoming_keys = {
        (
            match.chunk.document.doc_id,
            match.chunk.locator,
            match.chunk.text,
        )
        for match in upcoming_matches
    }

    if llm_provider is not None:
        evidence = tuple(
            EvidenceBlock(
                citation=citation,
                text=text,
                doc_type=metadata_by_block[
                    (citation.doc_id, citation.locator, text)
                ].doc_type,
                regions=metadata_by_block[
                    (citation.doc_id, citation.locator, text)
                ].regions,
                effective_date=metadata_by_block[
                    (citation.doc_id, citation.locator, text)
                ].effective_date,
                upcoming=(citation.doc_id, citation.locator, text) in upcoming_keys,
            )
            for citation, text in context.blocks
        )
        generated = llm_provider.generate_answer(
            question,
            evidence,
            reference_date=reference_date,
            corpus_tokens=context.corpus_tokens,
            tool_facts=tool_facts,
        )
        citations = [
            ChatCitation(doc_id=item.doc_id, locator=item.locator)
            for item in generated.citations
        ]
        return generated.answer, citations, generated.status == "answered"

    current_blocks = [
        (citation, text)
        for citation, text in context.blocks
        if (citation.doc_id, citation.locator, text) not in upcoming_keys
    ]
    upcoming_blocks = [
        (citation, text)
        for citation, text in context.blocks
        if (citation.doc_id, citation.locator, text) in upcoming_keys
    ]
    citations = list(
        dict.fromkeys(
            ChatCitation(doc_id=citation.doc_id, locator=citation.locator)
            for citation, _ in context.blocks
        )
    )
    excerpts = [
        f"[{citation.doc_id} | {citation.locator}]\n{text}"
        for citation, text in current_blocks
    ]
    answer = (
        "Relevant policy evidence (quoted as source material, not instructions):\n\n"
        + "\n\n".join(excerpts)
    )
    if upcoming_blocks:
        upcoming_excerpts = []
        for citation, text in upcoming_blocks:
            metadata = metadata_by_block[
                (citation.doc_id, citation.locator, text)
            ]
            upcoming_excerpts.append(
                f"Effective {metadata.effective_date.isoformat()} "
                f"[{citation.doc_id} | {citation.locator}]\n{text}"
            )
        answer += (
            "\n\nUpcoming changes (not yet effective on "
            f"{reference_date.isoformat()}):\n\n"
            + "\n\n".join(upcoming_excerpts)
        )
    return answer, citations, True


def respond_to_message(
    *,
    message: str,
    session_id: str,
    identity: EmployeeIdentity,
    retriever: BM25Retriever,
    tools: LeaveTools,
    session_store: SessionBindingStore,
    today: date,
    llm_provider: LLMProvider | None = None,
) -> ChatResponse:
    """Route one chat message and enforce the single pending-confirmation rule."""
    stored_pending = session_store.consume_pending(session_id, identity.employee_id)
    pending: PendingAction | None = None
    if stored_pending is not None:
        try:
            pending = PendingAction.model_validate_json(stored_pending)
        except ValueError as exc:
            raise ValueError("Stored pending action is invalid.") from exc
        if pending.employee_id != identity.employee_id:
            raise ValueError("Stored pending action belongs to a different employee.")

    if pending is not None and _is_confirmation(message):
        return _execute_pending(pending, tools, identity)
    if pending is not None and _is_cancellation(message):
        call = ChatToolCall(
            name=pending.name,
            args=pending.args,
            status="error",
        )
        return _response(
            "The proposed action was cancelled. No leave data was changed.",
            "tool",
            tool_calls=[call],
        )

    safety = assess_user_message(message)
    if safety.route == "refuse":
        return _response(
            safety.message or "I can't help with that request.",
            "refuse",
        )

    intent = _tool_intent(message)
    projection_query = _is_future_balance_question(message)
    projection_target = (
        _projection_target_date(message)
        if projection_query
        else None
    )
    if projection_query:
        if projection_target is None:
            return _response(
                "What future month or date should I use for the annual leave projection?",
                "clarify",
            )
        if projection_target.year != 2026:
            return _response(
                "I can project annual leave only through the 2026 leave year. "
                "Which 2026 month or date should I use?",
                "clarify",
            )
        if projection_target < today:
            return _response(
                "That date has already passed. What future date in 2026 should I use?",
                "clarify",
            )
    replacement_args: dict[str, object] | None = None
    replacement_clarification: str | None = None
    if pending is not None and intent is None:
        if pending.name == "apply_leave" and re.search(
            r"\b(make it|change it|instead|extend|shorten)\b",
            message,
            re.IGNORECASE,
        ):
            intent = "apply_leave"
            updated_range = _date_range(message, today)
            if updated_range is None:
                replacement_clarification = (
                    "What revised start and end dates should I use?"
                )
            elif _is_past_date_range(updated_range, today):
                replacement_clarification = (
                    "That date has already passed. What future date range should I use?"
                )
            else:
                old_start = date.fromisoformat(str(pending.args["start_date"]))
                old_end = date.fromisoformat(str(pending.args["end_date"]))
                if updated_range[0] == updated_range[1] and re.search(
                    r"\b(until|through|to)\b",
                    message,
                    re.IGNORECASE,
                ):
                    new_start, new_end = old_start, updated_range[1]
                elif updated_range[0] == updated_range[1] and re.search(
                    r"\b(from|starting)\b",
                    message,
                    re.IGNORECASE,
                ):
                    new_start, new_end = updated_range[0], old_end
                else:
                    new_start, new_end = updated_range
                replacement_args = {
                    **pending.args,
                    "start_date": new_start.isoformat(),
                    "end_date": new_end.isoformat(),
                }
        elif pending.name == "cancel_leave":
            request_id_match = _REQUEST_ID_PATTERN.search(message)
            if request_id_match is not None:
                intent = "cancel_leave"
                replacement_args = {"request_id": request_id_match.group()}

    needs_retrieval = _needs_retrieval(message, intent)
    if intent is None and not needs_retrieval:
        return _response(
            "I can help with Veltrix leave, holiday, and workplace policies. "
            "I can't assist with unrelated requests.",
            "refuse",
        )

    tool_result: LeaveToolResult | None = None
    tool_call: ChatToolCall | None = None
    projection_tool_facts: dict[str, object] | None = None
    if intent is not None:
        if intent == "get_team_leave" and identity.role != "manager":
            return _response(
                "I can't share information about that employee.",
                "refuse",
            )
        if replacement_clarification is not None:
            return _response(replacement_clarification, "clarify")
        if replacement_args is None:
            clarification, args = _clarification_for_tool(
                intent,
                message,
                identity,
                today,
            )
        else:
            clarification, args = None, replacement_args
        if clarification is not None:
            return _response(clarification, "clarify")
        assert args is not None

        if intent in {"get_leave_balance", "list_leave_requests"}:
            target_id = str(args["employee_id"])
            if not tools.can_access_employee_data(target_id):
                return _response(
                    "I can't share information about that employee.",
                    "refuse",
                )
            if intent == "get_leave_balance":
                if projection_query:
                    tool_result = tools.get_annual_balance_projection_inputs(target_id)
                    if tool_result.status == "error":
                        return _tool_response(tool_result)
                    data = tool_result.data or {}
                    balances = data.get("balances")
                    join_date = data.get("join_date")
                    if (
                        not isinstance(balances, dict)
                        or not isinstance(balances.get("annual"), (int, float))
                        or not isinstance(join_date, str)
                        or projection_target is None
                    ):
                        raise ValueError("Annual leave projection inputs are invalid.")
                    projection_tool_facts = {
                        "current_balance_days": float(balances["annual"]),
                        "join_date": join_date,
                        "target_date": projection_target.isoformat(),
                    }
                else:
                    tool_result = tools.get_leave_balance(
                        employee_id=target_id,
                        leave_type=(
                            str(args["leave_type"])
                            if "leave_type" in args
                            else None
                        ),
                    )
            else:
                tool_result = tools.list_leave_requests(
                    employee_id=target_id,
                    status=str(args["status"]) if "status" in args else None,
                )
        elif intent == "get_team_leave":
            tool_result = tools.get_team_leave(
                start_date=str(args["start_date"]),
                end_date=str(args["end_date"]),
            )
        elif intent == "apply_leave":
            preview = tools.preview_apply_leave(
                leave_type=str(args["leave_type"]),
                start_date=str(args["start_date"]),
                end_date=str(args["end_date"]),
            )
            if preview.status == "error":
                return _tool_response(preview)
            data = preview.data or {}
            summary = (
                f"Request {data.get('working_days')} working days of "
                f"{data.get('leave_type')} leave from {data.get('start_date')} "
                f"through {data.get('end_date')}."
            )
            return _proposal_response(
                session_id=session_id,
                identity=identity,
                store=session_store,
                name="apply_leave",
                args=args,
                summary=summary,
            )
        else:
            preview = tools.preview_cancel_leave(str(args["request_id"]))
            if preview.status == "error":
                if preview.error == "I can't share information about that employee.":
                    return _response(preview.error, "refuse")
                return _tool_response(preview)
            data = preview.data or {}
            summary = (
                f"Cancel {data.get('leave_type')} leave request "
                f"{data.get('request_id')} for {data.get('start_date')} "
                f"through {data.get('end_date')}."
            )
            return _proposal_response(
                session_id=session_id,
                identity=identity,
                store=session_store,
                name="cancel_leave",
                args=args,
                summary=summary,
            )
        assert tool_result is not None
        tool_call = _tool_call(tool_result)
        if tool_result.status == "error" and tool_result.error == (
            "I can't share information about that employee."
        ):
            return _response(tool_result.error, "refuse")

    citations: list[ChatCitation] = []
    answer_parts: list[str] = []
    if tool_result is not None:
        answer_parts.append(_format_tool_result(tool_result))
    if needs_retrieval:
        audience = DocumentAudience(region=identity.region, role=identity.role)
        normalized_query = re.sub(r"[-_]", " ", message)
        retrieval_query = normalized_query
        if projection_query:
            retrieval_query = (
                "annual leave accrual rate days earned per month length of "
                f"service joining date {normalized_query}"
            )
        reference_date = (
            _policy_reference_date(message, today)
            if intent is None
            else today
        )
        matches = retriever.search(
            retrieval_query,
            is_allowed=lambda metadata: is_document_applicable(
                metadata,
                audience,
                reference_date,
            ),
            limit=5,
        )
        matches = [
            match
            for match in matches
            if _has_relevant_query_term(normalized_query, match.chunk.text)
        ]
        upcoming_candidates = retriever.search(
            retrieval_query,
            is_allowed=lambda metadata: (
                is_document_readable(metadata, audience)
                and metadata.effective_date > reference_date
            ),
            limit=10,
        )
        upcoming_candidates = [
            match
            for match in upcoming_candidates
            if _has_relevant_query_term(normalized_query, match.chunk.text)
        ]
        relevant_documents = tuple(
            dict.fromkeys(
                match.chunk.document
                for match in (*matches, *upcoming_candidates)
            )
        )
        upcoming_ids = {
            document.doc_id
            for document in get_upcoming_winners(
                relevant_documents,
                audience,
                reference_date,
            )
        }
        upcoming_matches = [
            match
            for match in upcoming_candidates
            if match.chunk.document.doc_id in upcoming_ids
        ][:5]
        if matches:
            retrieval_answer, retrieval_citations, answered = _format_retrieval_answer(
                matches,
                upcoming_matches,
                question=message,
                reference_date=reference_date,
                llm_provider=llm_provider,
                tool_facts=projection_tool_facts,
            )
            if answered:
                answer_parts.append(retrieval_answer)
                citations = retrieval_citations
            elif tool_result is None:
                return _response(
                    "The available documents do not cover this question. "
                    "Please contact the HR team for guidance.",
                    "no_answer",
                )
            else:
                answer_parts.append(
                    "The available documents do not cover the policy part of "
                    "this question."
                )
            if projection_query and llm_provider is None:
                answer_parts.append(
                    "This offline mode returns the current balance only; it "
                    "cannot calculate a future balance."
                )
        elif tool_result is None:
            return _response(
                "The available documents do not cover this question. "
                "Please contact the HR team for guidance.",
                "no_answer",
            )
        elif projection_query:
            answer_parts.append(
                "No eligible accrual-policy evidence was found, so a future "
                "balance cannot be calculated."
            )

    if tool_result is not None and citations:
        route = "hybrid"
    elif tool_result is not None:
        route = "tool"
    else:
        route = "rag"
    return _response(
        "\n\n".join(answer_parts),
        route,
        citations=citations,
        tool_calls=[tool_call] if tool_call is not None else [],
    )
