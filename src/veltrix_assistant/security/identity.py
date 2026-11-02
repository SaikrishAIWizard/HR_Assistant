"""Database-backed identity lookup and persistent session ownership."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from veltrix_assistant.policy.applicability import parse_role
from veltrix_assistant.schemas.security import EmployeeIdentity


class IdentityDirectoryError(RuntimeError):
    """Raised when the employee directory is unavailable or malformed."""


class SessionBindingStore:
    """Persist the first authenticated employee associated with each session."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)

    def ensure_schema(self) -> None:
        """Create only the assistant-owned session table, preserving source tables."""
        if not self._database_path.is_file():
            raise IdentityDirectoryError("Employee database does not exist.")
        try:
            with sqlite3.connect(self._database_path, timeout=5) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS assistant_sessions (
                        session_id TEXT PRIMARY KEY,
                        employee_id TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error as exc:
            raise IdentityDirectoryError(
                "Could not initialize persistent session ownership."
            ) from exc

    def bind(self, session_id: str, employee_id: str) -> bool:
        """Bind an unused session or confirm the existing owner; false means mismatch."""
        if not session_id.strip() or not employee_id.strip():
            raise ValueError("session_id and employee_id must not be blank.")
        try:
            with sqlite3.connect(self._database_path, timeout=5) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT employee_id FROM assistant_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is not None:
                    return row[0] == employee_id
                connection.execute(
                    """
                    INSERT INTO assistant_sessions (session_id, employee_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, employee_id, datetime.now(timezone.utc).isoformat()),
                )
                return True
        except sqlite3.Error as exc:
            raise IdentityDirectoryError(
                "Could not verify session ownership."
            ) from exc


def lookup_employee_identity(
    database_path: str | Path,
    employee_id: str,
) -> EmployeeIdentity | None:
    """Resolve role and region only from the employee database."""
    try:
        with sqlite3.connect(database_path, timeout=5) as connection:
            row = connection.execute(
                "SELECT region, role FROM employees WHERE employee_id = ?",
                (employee_id,),
            ).fetchone()
    except sqlite3.Error as exc:
        raise IdentityDirectoryError("Employee directory lookup failed.") from exc

    if row is None:
        return None
    region, stored_role = row
    role = parse_role(stored_role)
    if role is None or not isinstance(region, str):
        raise IdentityDirectoryError("Employee directory contains an invalid access record.")
    try:
        return EmployeeIdentity(
            employee_id=employee_id,
            region=region,
            role=role,
        )
    except ValueError as exc:
        raise IdentityDirectoryError(
            "Employee directory contains an invalid access record."
        ) from exc
