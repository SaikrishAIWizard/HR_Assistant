import json
import sqlite3
import time

from fastapi.testclient import TestClient

from veltrix_assistant.api import create_app


def _write_fixture_files(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "global.md").write_text(
        "# Annual Leave\nGlobal annual leave policy details.\n",
        encoding="utf-8",
    )
    (corpus_dir / "restricted.md").write_text(
        "# Annual Leave\nHR annual leave confidential exception.\n",
        encoding="utf-8",
    )
    (corpus_dir / "future.md").write_text(
        "# Annual Leave\nAnnual leave future change.\n",
        encoding="utf-8",
    )
    (corpus_dir / "manager.md").write_text(
        "# Annual Leave\nManager confidential leave exception.\n",
        encoding="utf-8",
    )
    (corpus_dir / "germany.md").write_text(
        "# Annual Leave\nGermany annual leave regional policy.\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "GLOBAL",
                    "file": "global.md",
                    "title": "Global Leave Policy",
                    "doc_type": "policy",
                    "regions": ["GLOBAL"],
                    "effective_date": "2025-01-01",
                    "access_level": "all",
                },
                {
                    "doc_id": "HR",
                    "file": "restricted.md",
                    "title": "HR Leave Policy",
                    "doc_type": "policy",
                    "regions": ["GLOBAL"],
                    "effective_date": "2025-01-01",
                    "access_level": "hr",
                },
                {
                    "doc_id": "FUTURE",
                    "file": "future.md",
                    "title": "Future Leave Policy",
                    "doc_type": "policy",
                    "regions": ["GLOBAL"],
                    "effective_date": "2027-01-01",
                    "access_level": "all",
                },
                {
                    "doc_id": "MANAGER",
                    "file": "manager.md",
                    "title": "Manager Leave Policy",
                    "doc_type": "policy",
                    "regions": ["GLOBAL"],
                    "effective_date": "2025-01-01",
                    "access_level": "managers",
                },
                {
                    "doc_id": "GERMANY",
                    "file": "germany.md",
                    "title": "Germany Leave Policy",
                    "doc_type": "addendum",
                    "regions": ["DE"],
                    "effective_date": "2025-01-01",
                    "access_level": "all",
                },
            ]
        ),
        encoding="utf-8",
    )
    database_path = tmp_path / "employees.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE employees (employee_id TEXT PRIMARY KEY, region TEXT, role TEXT)"
        )
        connection.executemany(
            "INSERT INTO employees VALUES (?, ?, ?)",
            [
                ("E1001", "IN", "employee"),
                ("M1001", "IN", "manager"),
                ("H1001", "IN", "hr"),
                ("D1001", "DE", "employee"),
            ],
        )
    return corpus_dir, manifest_path, database_path


def _wait_until_ready(client: TestClient) -> None:
    for _ in range(100):
        response = client.get("/health")
        if response.status_code == 200:
            return
        time.sleep(0.05)
    raise AssertionError("API did not finish ingestion in time")


def test_health_and_search_apply_database_access_filters(tmp_path) -> None:
    corpus_dir, manifest_path, database_path = _write_fixture_files(tmp_path)
    app = create_app(
        corpus_dir=corpus_dir,
        manifest_path=manifest_path,
        database_path=database_path,
    )

    with TestClient(app) as client:
        _wait_until_ready(client)
        response = client.post(
            "/search",
            headers={"X-Employee-Id": "E1001"},
            json={"query": "annual leave"},
        )

    assert response.status_code == 200
    assert {item["doc_id"] for item in response.json()["results"]} == {"GLOBAL"}


def test_hr_can_search_hr_documents_but_unknown_employee_gets_401(tmp_path) -> None:
    corpus_dir, manifest_path, database_path = _write_fixture_files(tmp_path)
    app = create_app(
        corpus_dir=corpus_dir,
        manifest_path=manifest_path,
        database_path=database_path,
    )

    with TestClient(app) as client:
        _wait_until_ready(client)
        hr_response = client.post(
            "/search",
            headers={"X-Employee-Id": "H1001"},
            json={"query": "annual leave confidential"},
        )
        unknown_response = client.post(
            "/search",
            headers={"X-Employee-Id": "UNKNOWN"},
            json={"query": "annual leave"},
        )

    assert hr_response.status_code == 200
    assert "HR" in {item["doc_id"] for item in hr_response.json()["results"]}
    assert unknown_response.status_code == 401


def test_search_requires_employee_header(tmp_path) -> None:
    corpus_dir, manifest_path, database_path = _write_fixture_files(tmp_path)
    app = create_app(
        corpus_dir=corpus_dir,
        manifest_path=manifest_path,
        database_path=database_path,
    )

    with TestClient(app) as client:
        _wait_until_ready(client)
        response = client.post("/search", json={"query": "annual leave"})

    assert response.status_code == 401


def test_search_filters_manager_documents_and_regions(tmp_path) -> None:
    corpus_dir, manifest_path, database_path = _write_fixture_files(tmp_path)
    app = create_app(
        corpus_dir=corpus_dir,
        manifest_path=manifest_path,
        database_path=database_path,
    )

    with TestClient(app) as client:
        _wait_until_ready(client)
        employee_response = client.post(
            "/search",
            headers={"X-Employee-Id": "E1001"},
            json={"query": "manager confidential leave"},
        )
        manager_response = client.post(
            "/search",
            headers={"X-Employee-Id": "M1001"},
            json={"query": "manager confidential leave"},
        )
        regional_response = client.post(
            "/search",
            headers={"X-Employee-Id": "E1001"},
            json={"query": "Germany annual leave regional"},
        )

    employee_doc_ids = {
        item["doc_id"] for item in employee_response.json()["results"]
    }
    manager_doc_ids = {
        item["doc_id"] for item in manager_response.json()["results"]
    }
    regional_doc_ids = {
        item["doc_id"] for item in regional_response.json()["results"]
    }
    assert "MANAGER" not in employee_doc_ids
    assert "MANAGER" in manager_doc_ids
    assert "GERMANY" not in regional_doc_ids


def test_search_excludes_documents_not_yet_effective(tmp_path) -> None:
    corpus_dir, manifest_path, database_path = _write_fixture_files(tmp_path)
    app = create_app(
        corpus_dir=corpus_dir,
        manifest_path=manifest_path,
        database_path=database_path,
    )

    with TestClient(app) as client:
        _wait_until_ready(client)
        response = client.post(
            "/search",
            headers={"X-Employee-Id": "E1001"},
            json={
                "query": "future change annual leave",
                "reference_date": "2026-12-31",
            },
        )

    assert "FUTURE" not in {
        item["doc_id"] for item in response.json()["results"]
    }
