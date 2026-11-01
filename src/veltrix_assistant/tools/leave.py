"""Role-checked leave tools backed by the assessment SQLite database."""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from veltrix_assistant.schemas.documents import DocumentChunk
from veltrix_assistant.policy.applicability import (
    is_document_applicable,
    parse_role,
    select_precedence_winners,
)
from veltrix_assistant.schemas.policy import (
    DocumentAudience,
    Role,
)
from veltrix_assistant.schemas.tools import LeaveToolResult


_LOGGER = logging.getLogger("veltrix_assistant.tools.leave")
_UNAUTHORIZED = "I can't share information about that employee."
_ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATE_PATTERN = re.compile(r"\b\d{1,2}\s+[A-Za-z]+\s+2026\b")
_PUBLIC_HOLIDAY_LOCATOR = re.compile(
    r"\b(?:public|paid)\b.*\bholidays?\b|\bpublic and company holidays\b",
    re.IGNORECASE,
)


class HolidayCalendarError(ValueError):
    """Raised when an applicable calendar cannot be read from ingested content."""


class _TransactionAbort(RuntimeError):
    """Internal signal that an in-progress transaction must roll back."""


def _calendar_dates_from_chunk(chunk: DocumentChunk) -> set[date]:
    if not _PUBLIC_HOLIDAY_LOCATOR.search(chunk.locator):
        return set()
    dates: set[date] = set()
    for match in _DATE_PATTERN.finditer(chunk.text):
        try:
            dates.add(datetime.strptime(match.group(0), "%d %B %Y").date())
        except ValueError as exc:
            raise HolidayCalendarError(
                f"Invalid holiday date in calendar document {chunk.document.doc_id}."
            ) from exc
    return dates


def get_public_holidays(
    chunks: Sequence[DocumentChunk],
    *,
    region: str,
    reference_date: date,
    year: int = 2026,
) -> frozenset[date]:
    """Extract public holidays from the winning applicable calendar evidence."""
    audience = DocumentAudience(region=region, role="employee")
    calendar_documents = tuple(
        dict.fromkeys(
            chunk.document
            for chunk in chunks
            if chunk.document.doc_type == "calendar"
            and chunk.document.effective_date.year == year
            and is_document_applicable(chunk.document, audience, reference_date)
        )
    )
    if not calendar_documents:
        raise HolidayCalendarError(
            f"No applicable holiday calendar is available for region {region} in {year}."
        )

    winners = select_precedence_winners(calendar_documents, audience)
    winner_ids = {document.doc_id for document in winners}
    holidays = frozenset(
        holiday
        for chunk in chunks
        if chunk.document.doc_id in winner_ids
        for holiday in _calendar_dates_from_chunk(chunk)
    )
    if not holidays:
        raise HolidayCalendarError(
            f"The applicable holiday calendar for region {region} contains no readable public holidays."
        )
    return frozenset(day for day in holidays if day.year == year)


def count_working_days(
    start_date: date,
    end_date: date,
    public_holidays: frozenset[date] | set[date],
) -> int:
    """Count inclusive weekdays excluding public holidays."""
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date.")
    total = 0
    current = start_date
    while current <= end_date:
        if current.weekday() < 5 and current not in public_holidays:
            total += 1
        current += timedelta(days=1)
    return total


def _normalize_args(args: dict[str, object]) -> dict[str, object]:
    return {
        key: value.isoformat() if isinstance(value, date) else value
        for key, value in args.items()
    }


def _parse_tool_date(value: date | str) -> date | None:
    if isinstance(value, date):
        return value
    if not _ISO_DATE_PATTERN.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _success(
    name: str,
    args: dict[str, object],
    data: dict[str, object],
) -> LeaveToolResult:
    return LeaveToolResult(
        name=name,
        args=_normalize_args(args),
        status="success",
        data=data,
    )


def _error(
    name: str,
    args: dict[str, object],
    message: str,
) -> LeaveToolResult:
    return LeaveToolResult(
        name=name,
        args=_normalize_args(args),
        status="error",
        error=message,
    )


class LeaveToolFactory:
    """Creates request-scoped tool objects bound to a trusted header identity."""

    def __init__(
        self,
        database_path: str | Path,
        chunks: Sequence[DocumentChunk],
        *,
        today: date,
    ) -> None:
        self._database_path = Path(database_path)
        self._chunks = tuple(chunks)
        self._today = today

    def for_employee(self, authenticated_employee_id: str) -> LeaveTools:
        """Bind tools to the employee ID authenticated by the API request."""
        return LeaveTools(
            self._database_path,
            self._chunks,
            authenticated_employee_id=authenticated_employee_id,
            today=self._today,
        )


class LeaveTools:
    """Five callable tools whose permissions are enforced in every operation."""

    def __init__(
        self,
        database_path: str | Path,
        chunks: Sequence[DocumentChunk],
        *,
        authenticated_employee_id: str,
        today: date,
    ) -> None:
        if not authenticated_employee_id:
            raise ValueError("An authenticated employee ID is required.")
        self._database_path = Path(database_path)
        self._chunks = tuple(chunks)
        self._authenticated_employee_id = authenticated_employee_id
        self._today = today

    def get_tool_functions(self) -> dict[str, Callable[..., LeaveToolResult]]:
        """Return the exact assessment tool names mapped to their bound functions."""
        return {
            "get_leave_balance": self.get_leave_balance,
            "list_leave_requests": self.list_leave_requests,
            "get_team_leave": self.get_team_leave,
            "apply_leave": self.apply_leave,
            "cancel_leave": self.cancel_leave,
        }

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path, timeout=5)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _requester(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[str, Role, str | None] | None:
        row = connection.execute(
            "SELECT region, role, manager_id FROM employees WHERE employee_id = ?",
            (self._authenticated_employee_id,),
        ).fetchone()
        if row is None:
            return None
        region, role, manager_id = row
        valid_role = parse_role(role)
        if valid_role is None or not isinstance(region, str):
            _LOGGER.error("Authenticated employee has invalid leave-tool access attributes")
            return None
        return region, valid_role, manager_id

    @staticmethod
    def _target_is_allowed(
        connection: sqlite3.Connection,
        requester: tuple[str, Role, str | None],
        requester_id: str,
        target_id: str,
    ) -> bool:
        if not target_id:
            return False
        target = connection.execute(
            "SELECT 1 FROM employees WHERE employee_id = ?",
            (target_id,),
        ).fetchone()
        if target is None:
            return False
        if target_id == requester_id:
            return True
        _, role, _ = requester
        if role == "hr":
            return True
        if role == "manager":
            return connection.execute(
                "SELECT 1 FROM employees WHERE employee_id = ? AND manager_id = ?",
                (target_id, requester_id),
            ).fetchone() is not None
        return False

    def get_leave_balance(
        self,
        employee_id: str,
        leave_type: str | None = None,
    ) -> LeaveToolResult:
        """Return one or both current balances for an authorized employee."""
        name = "get_leave_balance"
        args: dict[str, object] = {"employee_id": employee_id}
        if leave_type is not None:
            args["leave_type"] = leave_type
        if leave_type is not None and leave_type not in {"annual", "sick"}:
            return _error(name, args, "leave_type must be 'annual' or 'sick'.")

        try:
            with self._connection() as connection:
                requester = self._requester(connection)
                if requester is None or not self._target_is_allowed(
                    connection,
                    requester,
                    self._authenticated_employee_id,
                    employee_id,
                ):
                    return _error(name, args, _UNAUTHORIZED)
                query = (
                    "SELECT leave_type, balance_days FROM leave_balances "
                    "WHERE employee_id = ?"
                )
                parameters: tuple[object, ...] = (employee_id,)
                if leave_type is not None:
                    query += " AND leave_type = ?"
                    parameters += (leave_type,)
                rows = connection.execute(query, parameters).fetchall()
        except sqlite3.Error:
            _LOGGER.exception("Leave balance lookup failed")
            return _error(name, args, "Leave data is temporarily unavailable.")

        if not rows:
            return _error(name, args, "No leave balance exists for the requested leave type.")
        balances = {kind: float(balance) for kind, balance in rows}
        return _success(
            name,
            args,
            {
                "employee_id": employee_id,
                "balances": balances,
            },
        )

    def list_leave_requests(
        self,
        employee_id: str,
        status: str | None = None,
    ) -> LeaveToolResult:
        """List requests for the employee, subject to self/manager/HR access."""
        name = "list_leave_requests"
        args: dict[str, object] = {"employee_id": employee_id}
        if status is not None:
            args["status"] = status
        if status is not None and status not in {"pending", "approved", "cancelled"}:
            return _error(
                name,
                args,
                "status must be 'pending', 'approved', or 'cancelled'.",
            )

        try:
            with self._connection() as connection:
                requester = self._requester(connection)
                if requester is None or not self._target_is_allowed(
                    connection,
                    requester,
                    self._authenticated_employee_id,
                    employee_id,
                ):
                    return _error(name, args, _UNAUTHORIZED)
                query = (
                    "SELECT request_id, employee_id, leave_type, start_date, end_date, "
                    "working_days, status FROM leave_requests WHERE employee_id = ?"
                )
                parameters: tuple[object, ...] = (employee_id,)
                if status is not None:
                    query += " AND status = ?"
                    parameters += (status,)
                query += " ORDER BY start_date, request_id"
                rows = connection.execute(query, parameters).fetchall()
        except sqlite3.Error:
            _LOGGER.exception("Leave request lookup failed")
            return _error(name, args, "Leave data is temporarily unavailable.")

        requests = [
            {
                "request_id": row[0],
                "employee_id": row[1],
                "leave_type": row[2],
                "start_date": row[3],
                "end_date": row[4],
                "working_days": row[5],
                "status": row[6],
            }
            for row in rows
        ]
        return _success(name, args, {"requests": requests})

    def get_team_leave(
        self,
        start_date: date | str,
        end_date: date | str,
    ) -> LeaveToolResult:
        """Return overlapping pending/approved leave for a manager's reports."""
        name = "get_team_leave"
        args: dict[str, object] = {"start_date": start_date, "end_date": end_date}
        parsed_start = _parse_tool_date(start_date)
        parsed_end = _parse_tool_date(end_date)
        if parsed_start is None or parsed_end is None:
            return _error(name, args, "start_date and end_date must use YYYY-MM-DD format.")
        start_date = parsed_start
        end_date = parsed_end
        if end_date < start_date:
            return _error(name, args, "end_date must be on or after start_date.")

        try:
            with self._connection() as connection:
                requester = self._requester(connection)
                if requester is None or requester[1] != "manager":
                    return _error(name, args, "Only managers may view team leave.")
                rows = connection.execute(
                    """
                    SELECT request_id, employee_id, leave_type, start_date, end_date,
                           working_days, status
                    FROM leave_requests
                    WHERE employee_id IN (
                        SELECT employee_id FROM employees WHERE manager_id = ?
                    )
                      AND status IN ('pending', 'approved')
                      AND start_date <= ?
                      AND end_date >= ?
                    ORDER BY start_date, employee_id, request_id
                    """,
                    (
                        self._authenticated_employee_id,
                        end_date.isoformat(),
                        start_date.isoformat(),
                    ),
                ).fetchall()
        except sqlite3.Error:
            _LOGGER.exception("Team leave lookup failed")
            return _error(name, args, "Leave data is temporarily unavailable.")

        requests = [
            {
                "request_id": row[0],
                "employee_id": row[1],
                "leave_type": row[2],
                "start_date": row[3],
                "end_date": row[4],
                "working_days": row[5],
                "status": row[6],
            }
            for row in rows
        ]
        return _success(name, args, {"requests": requests})

    def apply_leave(
        self,
        leave_type: str | None,
        start_date: date | str,
        end_date: date | str,
    ) -> LeaveToolResult:
        """Validate, persist, and debit leave in one SQLite transaction."""
        name = "apply_leave"
        args: dict[str, object] = {
            "leave_type": leave_type,
            "start_date": start_date,
            "end_date": end_date,
        }
        parsed_start = _parse_tool_date(start_date)
        parsed_end = _parse_tool_date(end_date)
        if parsed_start is None or parsed_end is None:
            return _error(name, args, "start_date and end_date must use YYYY-MM-DD format.")
        start_date = parsed_start
        end_date = parsed_end
        if leave_type not in {"annual", "sick"}:
            message = (
                "leave_type is required; ask the employee to choose annual or sick."
                if leave_type is None
                else "leave_type must be 'annual' or 'sick'."
            )
            return _error(name, args, message)
        if start_date.year != 2026 or end_date.year != 2026:
            return _error(name, args, "Both leave dates must be within calendar year 2026.")
        if start_date < self._today:
            return _error(name, args, "start_date must be today or later.")
        if end_date < start_date:
            return _error(name, args, "end_date must be on or after start_date.")

        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                requester = self._requester(connection)
                if requester is None:
                    return _error(name, args, _UNAUTHORIZED)
                region = requester[0]
                holidays = get_public_holidays(
                    self._chunks,
                    region=region,
                    reference_date=self._today,
                )
                working_days = count_working_days(start_date, end_date, holidays)
                if working_days == 0:
                    return _error(
                        name,
                        args,
                        "The selected date range contains zero working days.",
                    )

                overlap = connection.execute(
                    """
                    SELECT 1 FROM leave_requests
                    WHERE employee_id = ?
                      AND status IN ('pending', 'approved')
                      AND start_date <= ?
                      AND end_date >= ?
                    LIMIT 1
                    """,
                    (
                        self._authenticated_employee_id,
                        end_date.isoformat(),
                        start_date.isoformat(),
                    ),
                ).fetchone()
                if overlap is not None:
                    return _error(
                        name,
                        args,
                        "The date range overlaps an existing pending or approved request.",
                    )

                balance_row = connection.execute(
                    """
                    SELECT balance_days FROM leave_balances
                    WHERE employee_id = ? AND leave_type = ?
                    """,
                    (self._authenticated_employee_id, leave_type),
                ).fetchone()
                if balance_row is None:
                    return _error(
                        name,
                        args,
                        f"No {leave_type} leave balance exists for the requester.",
                    )
                balance = float(balance_row[0])
                if working_days > balance:
                    return _error(
                        name,
                        args,
                        f"Insufficient {leave_type} leave balance: "
                        f"{working_days} working days requested, {balance:g} available.",
                    )

                request_id = str(uuid.uuid4())
                created_at = datetime.now(timezone.utc).isoformat()
                cursor = connection.execute(
                    """
                    UPDATE leave_balances
                    SET balance_days = balance_days - ?
                    WHERE employee_id = ? AND leave_type = ? AND balance_days >= ?
                    """,
                    (
                        working_days,
                        self._authenticated_employee_id,
                        leave_type,
                        working_days,
                    ),
                )
                if cursor.rowcount != 1:
                    raise _TransactionAbort(
                        "Leave balance changed during the request; please retry."
                    )
                connection.execute(
                    """
                    INSERT INTO leave_requests (
                        request_id, employee_id, leave_type, start_date, end_date,
                        working_days, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        request_id,
                        self._authenticated_employee_id,
                        leave_type,
                        start_date.isoformat(),
                        end_date.isoformat(),
                        working_days,
                        created_at,
                    ),
                )
                remaining_balance = balance - working_days
        except HolidayCalendarError as exc:
            return _error(name, args, str(exc))
        except _TransactionAbort as exc:
            return _error(name, args, str(exc))
        except sqlite3.Error:
            _LOGGER.exception("Leave application transaction failed")
            return _error(
                name,
                args,
                "The leave request could not be saved; no balance was changed.",
            )

        return _success(
            name,
            args,
            {
                "request_id": request_id,
                "employee_id": self._authenticated_employee_id,
                "leave_type": leave_type,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "working_days": working_days,
                "status": "pending",
                "balance_remaining": remaining_balance,
            },
        )

    def cancel_leave(self, request_id: str) -> LeaveToolResult:
        """Cancel the authenticated requester's future leave and restore balance."""
        name = "cancel_leave"
        args: dict[str, object] = {"request_id": request_id}
        if not request_id:
            return _error(name, args, "request_id is required.")

        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                requester = self._requester(connection)
                if requester is None:
                    return _error(name, args, _UNAUTHORIZED)
                row = connection.execute(
                    """
                    SELECT leave_type, start_date, working_days, status
                    FROM leave_requests
                    WHERE request_id = ? AND employee_id = ?
                    """,
                    (request_id, self._authenticated_employee_id),
                ).fetchone()
                if row is None:
                    return _error(name, args, _UNAUTHORIZED)
                leave_type, start_date_text, working_days, status = row
                if status not in {"pending", "approved"}:
                    return _error(
                        name,
                        args,
                        "Only pending or approved leave requests can be cancelled.",
                    )
                if date.fromisoformat(start_date_text) <= self._today:
                    return _error(
                        name,
                        args,
                        "Only leave requests with a start_date after today can be cancelled.",
                    )
                balance_row = connection.execute(
                    """
                    SELECT balance_days FROM leave_balances
                    WHERE employee_id = ? AND leave_type = ?
                    """,
                    (self._authenticated_employee_id, leave_type),
                ).fetchone()
                if balance_row is None:
                    raise _TransactionAbort(
                        "The leave balance record is missing; the request was not cancelled."
                    )
                cursor = connection.execute(
                    """
                    UPDATE leave_requests SET status = 'cancelled'
                    WHERE request_id = ? AND employee_id = ?
                      AND status IN ('pending', 'approved')
                    """,
                    (request_id, self._authenticated_employee_id),
                )
                if cursor.rowcount != 1:
                    raise _TransactionAbort(
                        "The request changed during cancellation; please retry."
                    )
                cursor = connection.execute(
                    """
                    UPDATE leave_balances SET balance_days = balance_days + ?
                    WHERE employee_id = ? AND leave_type = ?
                    """,
                    (working_days, self._authenticated_employee_id, leave_type),
                )
                if cursor.rowcount != 1:
                    raise _TransactionAbort(
                        "The leave balance could not be restored; the request was not cancelled."
                    )
                restored_balance = float(balance_row[0]) + working_days
        except _TransactionAbort as exc:
            return _error(name, args, str(exc))
        except sqlite3.Error:
            _LOGGER.exception("Leave cancellation transaction failed")
            return _error(
                name,
                args,
                "The leave request could not be cancelled; no balance was changed.",
            )

        return _success(
            name,
            args,
            {
                "request_id": request_id,
                "status": "cancelled",
                "working_days_restored": working_days,
                "balance_remaining": restored_balance,
            },
        )
