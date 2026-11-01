import sqlite3
import uuid
from datetime import date
from inspect import signature

import pytest

from veltrix_assistant.ingestion.schemas import DocumentChunk, DocumentMetadata
from veltrix_assistant.tools.leave import (
    LeaveToolFactory,
    LeaveTools,
    count_working_days,
    get_public_holidays,
)


TODAY = date(2026, 11, 2)


def _calendar_chunk(
    doc_id: str,
    region: str,
    locator: str,
    text: str,
) -> DocumentChunk:
    metadata = DocumentMetadata(
        doc_id=doc_id,
        file=f"{doc_id}.pdf",
        title=f"{region} 2026 Holiday Calendar",
        doc_type="calendar",
        regions=(region,),
        effective_date=date(2026, 1, 1),
        access_level="all",
    )
    return DocumentChunk(
        chunk_id=f"{doc_id}-{locator}",
        document=metadata,
        page_number=1,
        locator=locator,
        text=text,
    )


@pytest.fixture
def leave_environment(tmp_path):
    database_path = tmp_path / "employees.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE employees (
                employee_id TEXT PRIMARY KEY,
                region TEXT NOT NULL,
                role TEXT NOT NULL,
                manager_id TEXT
            );
            CREATE TABLE leave_balances (
                employee_id TEXT NOT NULL,
                leave_type TEXT NOT NULL,
                balance_days REAL NOT NULL,
                PRIMARY KEY (employee_id, leave_type)
            );
            CREATE TABLE leave_requests (
                request_id TEXT PRIMARY KEY,
                employee_id TEXT NOT NULL,
                leave_type TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                working_days INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO employees VALUES (?, ?, ?, ?)",
            [
                ("E1001", "IN", "employee", "M1001"),
                ("E1002", "IN", "employee", "M1001"),
                ("E1003", "IN", "employee", "M2001"),
                ("M1001", "IN", "manager", None),
                ("M2001", "IN", "manager", None),
                ("H1001", "US", "hr", None),
            ],
        )
        connection.executemany(
            "INSERT INTO leave_balances VALUES (?, ?, ?)",
            [
                ("E1001", "annual", 8.0),
                ("E1001", "sick", 4.0),
                ("E1002", "annual", 5.0),
                ("E1002", "sick", 6.0),
                ("E1003", "annual", 7.0),
                ("M1001", "annual", 10.0),
                ("H1001", "annual", 12.0),
            ],
        )
        connection.executemany(
            "INSERT INTO leave_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "REQ-A1",
                    "E1001",
                    "annual",
                    "2026-11-16",
                    "2026-11-20",
                    5,
                    "pending",
                    "2026-01-01T00:00:00+00:00",
                ),
                (
                    "REQ-A2",
                    "E1002",
                    "annual",
                    "2026-11-10",
                    "2026-11-11",
                    2,
                    "approved",
                    "2026-01-01T00:00:00+00:00",
                ),
                (
                    "REQ-CANCELLED",
                    "E1002",
                    "annual",
                    "2026-11-12",
                    "2026-11-12",
                    1,
                    "cancelled",
                    "2026-01-01T00:00:00+00:00",
                ),
                (
                    "REQ-OTHER",
                    "E1003",
                    "annual",
                    "2026-11-10",
                    "2026-11-11",
                    2,
                    "pending",
                    "2026-01-01T00:00:00+00:00",
                ),
            ],
        )

    chunks = (
        _calendar_chunk(
            "CAL-IN-2026",
            "IN",
            "1. Public holidays 2026",
            "Date Day Holiday\n9 November 2026 Monday Deepavali\n"
            "25 December 2026 Friday Christmas Day",
        ),
        _calendar_chunk(
            "CAL-IN-2026",
            "IN",
            "2. Restricted holidays 2026",
            "Date Day Restricted holiday\n24 November 2026 Tuesday Guru Nanak Jayanti",
        ),
        _calendar_chunk(
            "CAL-US-2026",
            "US",
            "Paid holidays 2026",
            "Date Day Holiday\n26 November 2026 Thursday Thanksgiving Day",
        ),
    )
    factory = LeaveToolFactory(database_path, chunks, today=TODAY)
    return database_path, chunks, factory


def _tools(factory: LeaveToolFactory, employee_id: str = "E1001") -> LeaveTools:
    return factory.for_employee(employee_id)


def _balance(database_path, employee_id="E1001", leave_type="annual") -> float:
    with sqlite3.connect(database_path) as connection:
        return connection.execute(
            "SELECT balance_days FROM leave_balances "
            "WHERE employee_id = ? AND leave_type = ?",
            (employee_id, leave_type),
        ).fetchone()[0]


def _data(result) -> dict[str, object]:
    assert result.status == "success"
    assert result.data is not None
    return result.data


def _error_text(result) -> str:
    assert result.status == "error"
    assert result.error is not None
    return result.error


def _request_items(result) -> list[object]:
    request_items = _data(result)["requests"]
    assert isinstance(request_items, list)
    return request_items


def test_tool_registry_uses_required_names_and_argument_names(leave_environment) -> None:
    _, _, factory = leave_environment
    tools = _tools(factory)

    assert set(tools.get_tool_functions()) == {
        "get_leave_balance",
        "list_leave_requests",
        "get_team_leave",
        "apply_leave",
        "cancel_leave",
    }
    assert {
        name: tuple(signature(function).parameters)
        for name, function in tools.get_tool_functions().items()
    } == {
        "get_leave_balance": ("employee_id", "leave_type"),
        "list_leave_requests": ("employee_id", "status"),
        "get_team_leave": ("start_date", "end_date"),
        "apply_leave": ("leave_type", "start_date", "end_date"),
        "cancel_leave": ("request_id",),
    }


def test_calendar_extraction_uses_public_holidays_not_restricted_holidays(
    leave_environment,
) -> None:
    _, chunks, _ = leave_environment

    holidays = get_public_holidays(
        chunks,
        region="IN",
        reference_date=TODAY,
    )

    assert date(2026, 11, 9) in holidays
    assert date(2026, 12, 25) in holidays
    assert date(2026, 11, 24) not in holidays
    assert count_working_days(
        date(2026, 11, 9),
        date(2026, 11, 13),
        holidays,
    ) == 4
    assert count_working_days(
        date(2026, 11, 23),
        date(2026, 11, 27),
        holidays,
    ) == 5


def test_calendar_selection_is_region_specific(leave_environment) -> None:
    _, chunks, _ = leave_environment

    india_holidays = get_public_holidays(chunks, region="IN", reference_date=TODAY)
    us_holidays = get_public_holidays(chunks, region="US", reference_date=TODAY)

    assert date(2026, 11, 9) in india_holidays
    assert date(2026, 11, 26) not in india_holidays
    assert date(2026, 11, 26) in us_holidays


def test_apply_leave_counts_holidays_and_debits_balance_atomically(
    leave_environment,
) -> None:
    database_path, _, factory = leave_environment

    result = _tools(factory).apply_leave(
        "annual",
        date(2026, 11, 9),
        date(2026, 11, 13),
    )

    result_data = _data(result)
    assert result_data["working_days"] == 4
    assert result_data["balance_remaining"] == 4.0
    assert result_data["status"] == "pending"
    request_id = result_data["request_id"]
    assert isinstance(request_id, str)
    assert uuid.UUID(request_id).version == 4
    assert _balance(database_path) == 4.0
    with sqlite3.connect(database_path) as connection:
        saved = connection.execute(
            "SELECT employee_id, start_date, end_date, working_days, status "
            "FROM leave_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
    assert saved == ("E1001", "2026-11-09", "2026-11-13", 4, "pending")


def test_apply_leave_accepts_iso_dates_from_json_tool_arguments(
    leave_environment,
) -> None:
    database_path, _, factory = leave_environment

    result = _tools(factory).apply_leave(
        "annual",
        "2026-11-23",
        "2026-11-25",
    )

    result_data = _data(result)
    assert result_data["working_days"] == 3
    assert result_data["balance_remaining"] == 5.0
    assert _balance(database_path) == 5.0


def test_apply_leave_allows_starting_today(leave_environment) -> None:
    database_path, _, factory = leave_environment

    result = _tools(factory).apply_leave("sick", TODAY, TODAY)

    assert _data(result)["working_days"] == 1
    assert _balance(database_path, leave_type="sick") == 3.0


def test_apply_leave_returns_error_for_malformed_iso_dates(leave_environment) -> None:
    database_path, _, factory = leave_environment

    result = _tools(factory).apply_leave("annual", "November 23", "2026-11-25")

    assert "must use YYYY-MM-DD format" in _error_text(result)
    assert _balance(database_path) == 8.0


@pytest.mark.parametrize(
    ("leave_type", "start_date", "end_date", "error"),
    [
        ("annual", date(2026, 11, 1), date(2026, 11, 2), "start_date must be today"),
        ("annual", date(2026, 12, 31), date(2027, 1, 1), "within calendar year 2026"),
        ("annual", date(2026, 11, 4), date(2026, 11, 3), "end_date must be"),
        ("annual", date(2026, 11, 7), date(2026, 11, 8), "zero working days"),
        (None, date(2026, 11, 3), date(2026, 11, 3), "leave_type is required"),
        ("unpaid", date(2026, 11, 3), date(2026, 11, 3), "leave_type must be"),
    ],
)
def test_apply_leave_rejects_invalid_type_and_date_ranges(
    leave_environment,
    leave_type,
    start_date,
    end_date,
    error,
) -> None:
    database_path, _, factory = leave_environment

    result = _tools(factory).apply_leave(leave_type, start_date, end_date)

    assert error in _error_text(result)
    assert _balance(database_path) == 8.0


def test_apply_leave_rejects_insufficient_balance_and_overlapping_dates(
    leave_environment,
) -> None:
    database_path, _, factory = leave_environment
    tools = _tools(factory)

    insufficient = tools.apply_leave(
        "annual",
        date(2026, 11, 23),
        date(2026, 12, 4),
    )
    overlap = tools.apply_leave(
        "sick",
        date(2026, 11, 20),
        date(2026, 11, 23),
    )

    assert "Insufficient annual leave balance" in _error_text(insufficient)
    assert "overlaps an existing" in _error_text(overlap)
    assert _balance(database_path) == 8.0
    assert _balance(database_path, leave_type="sick") == 4.0


def test_failed_insert_rolls_back_balance_debit(leave_environment) -> None:
    database_path, _, factory = leave_environment
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_leave_insert
            BEFORE INSERT ON leave_requests
            BEGIN
                SELECT RAISE(ABORT, 'forced insert failure');
            END
            """
        )

    result = _tools(factory).apply_leave(
        "annual",
        date(2026, 11, 23),
        date(2026, 11, 25),
    )

    assert "no balance was changed" in _error_text(result)
    assert _balance(database_path) == 8.0
    with sqlite3.connect(database_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM leave_requests WHERE employee_id = 'E1001'"
        ).fetchone()[0]
    assert count == 1


def test_balance_and_request_tools_enforce_employee_manager_and_hr_scopes(
    leave_environment,
) -> None:
    _, _, factory = leave_environment
    employee = _tools(factory, "E1001")
    manager = _tools(factory, "M1001")
    hr = _tools(factory, "H1001")

    self_result = employee.get_leave_balance("E1001")
    manager_report = manager.get_leave_balance("E1002", "annual")
    manager_other = manager.get_leave_balance("E1003", "annual")
    hr_any = hr.get_leave_balance("E1003", "annual")
    nonexistent = employee.get_leave_balance("DOES-NOT-EXIST", "annual")

    assert _data(self_result)["balances"] == {"annual": 8.0, "sick": 4.0}
    _data(manager_report)
    assert _error_text(manager_other) == _error_text(nonexistent)
    assert _error_text(manager_other) == "I can't share information about that employee."
    _data(hr_any)
    assert manager.get_leave_balance("E1002", "personal").status == "error"

    requests = manager.list_leave_requests("E1002", "approved")
    request_items = _request_items(requests)
    assert len(request_items) == 1
    assert isinstance(request_items[0], dict)
    assert request_items[0]["request_id"] == "REQ-A2"


def test_get_team_leave_is_manager_only_direct_reports_and_overlap_filtered(
    leave_environment,
) -> None:
    _, _, factory = leave_environment
    manager = _tools(factory, "M1001")

    result = manager.get_team_leave(date(2026, 11, 10), date(2026, 11, 10))
    forbidden = _tools(factory, "E1001").get_team_leave(
        date(2026, 11, 10),
        date(2026, 11, 10),
    )

    request_items = _request_items(result)
    assert len(request_items) == 1
    assert isinstance(request_items[0], dict)
    assert request_items[0]["request_id"] == "REQ-A2"
    assert _error_text(forbidden) == "Only managers may view team leave."


def test_cancel_leave_marks_cancelled_and_restores_balance(leave_environment) -> None:
    database_path, _, factory = leave_environment
    tools = _tools(factory)
    applied = tools.apply_leave(
        "annual",
        date(2026, 11, 23),
        date(2026, 11, 25),
    )
    request_id = _data(applied)["request_id"]
    assert isinstance(request_id, str)
    before_cancel_balance = _balance(database_path)

    cancelled = tools.cancel_leave(request_id)

    cancelled_data = _data(cancelled)
    assert cancelled_data["status"] == "cancelled"
    assert cancelled_data["working_days_restored"] == 3
    assert cancelled_data["balance_remaining"] == 8.0
    assert _balance(database_path) == 8.0
    with sqlite3.connect(database_path) as connection:
        status = connection.execute(
            "SELECT status FROM leave_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()[0]
    assert status == "cancelled"
    assert before_cancel_balance == 5.0


def test_cancel_leave_allows_approved_future_requests(leave_environment) -> None:
    database_path, _, factory = leave_environment
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE leave_balances SET balance_days = 6 "
            "WHERE employee_id = 'E1001' AND leave_type = 'annual'"
        )
        connection.execute(
            "INSERT INTO leave_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "REQ-APPROVED-FUTURE",
                "E1001",
                "annual",
                "2026-11-30",
                "2026-12-01",
                2,
                "approved",
                "2026-01-01T00:00:00+00:00",
            ),
        )

    result = _tools(factory).cancel_leave("REQ-APPROVED-FUTURE")

    assert _data(result)["working_days_restored"] == 2
    assert _balance(database_path) == 8.0


def test_cancel_leave_requires_own_future_pending_or_approved_request(
    leave_environment,
) -> None:
    database_path, _, factory = leave_environment
    tools = _tools(factory)
    with sqlite3.connect(database_path) as connection:
        connection.executemany(
            "INSERT INTO leave_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "REQ-TODAY",
                    "E1001",
                    "annual",
                    "2026-11-02",
                    "2026-11-02",
                    1,
                    "pending",
                    "2026-01-01T00:00:00+00:00",
                ),
                (
                    "REQ-OTHER-OWNER",
                    "E1002",
                    "annual",
                    "2026-11-23",
                    "2026-11-23",
                    1,
                    "pending",
                    "2026-01-01T00:00:00+00:00",
                ),
                (
                    "REQ-CANCELLED-OWN",
                    "E1001",
                    "annual",
                    "2026-11-23",
                    "2026-11-23",
                    1,
                    "cancelled",
                    "2026-01-01T00:00:00+00:00",
                ),
            ],
        )

    same_day = tools.cancel_leave("REQ-TODAY")
    other_owner = tools.cancel_leave("REQ-OTHER-OWNER")
    cancelled = tools.cancel_leave("REQ-CANCELLED-OWN")

    assert "after today" in _error_text(same_day)
    assert _error_text(other_owner) == "I can't share information about that employee."
    assert "Only pending or approved" in _error_text(cancelled)
    assert _balance(database_path) == 8.0


def test_failed_balance_restore_rolls_back_request_cancellation(leave_environment) -> None:
    database_path, _, factory = leave_environment
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_balance_restore
            BEFORE UPDATE OF balance_days ON leave_balances
            WHEN NEW.balance_days > OLD.balance_days
            BEGIN
                SELECT RAISE(ABORT, 'forced balance failure');
            END
            """
        )

    result = _tools(factory).cancel_leave("REQ-A1")

    assert "no balance was changed" in _error_text(result)
    assert _balance(database_path) == 8.0
    with sqlite3.connect(database_path) as connection:
        status = connection.execute(
            "SELECT status FROM leave_requests WHERE request_id = 'REQ-A1'"
        ).fetchone()[0]
    assert status == "pending"
