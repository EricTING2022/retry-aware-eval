"""Read-only export of complete solver/verifier workflow records."""

from __future__ import annotations

from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Any


def _decode(value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _call_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    for source, target in (
        ("request_json", "request"),
        ("response_json", "response"),
        ("usage_json", "usage"),
    ):
        if source in record:
            record[target] = _decode(record.pop(source))
    return record


def _totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    reached = [row for row in calls if row["status"] in ("executed", "failed")]
    by_role: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "reported_cost_usd": 0.0,
            "calls_with_unknown_cost": 0,
            "elapsed_seconds": 0.0,
        }
    )
    for call in reached:
        values = by_role[call["role"]]
        values["calls"] += 1
        for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
            values[field] += call.get(field) or 0
        values["elapsed_seconds"] += call.get("elapsed_seconds") or 0.0
        cost = call.get("reported_cost_usd")
        if cost is None:
            values["calls_with_unknown_cost"] += 1
        else:
            values["reported_cost_usd"] += float(cost)
    total = {
        key: sum(values[key] for values in by_role.values())
        for key in (
            "calls",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "reported_cost_usd",
            "calls_with_unknown_cost",
            "elapsed_seconds",
        )
    }
    total["by_role"] = dict(sorted(by_role.items()))
    return total


def export_detailed_log(
    database: str | Path,
    output: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, int | str]:
    """Write one run header plus one self-contained JSON line per workflow."""
    database = Path(database)
    output = Path(output)
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if run_id is None:
            latest = connection.execute(
                "SELECT run_id FROM runs ORDER BY created_at_utc DESC LIMIT 1"
            ).fetchone()
            if latest is None:
                raise ValueError("The database contains no runs.")
            run_id = str(latest["run_id"])
        run = connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise ValueError(f"Run {run_id!r} does not exist.")
        configurations = {
            row["configuration_id"]: _decode(row["snapshot_json"])
            for row in connection.execute(
                "SELECT * FROM configurations WHERE run_id=?", (run_id,)
            )
        }
        executions = connection.execute(
            "SELECT * FROM executions WHERE run_id=? "
            "ORDER BY configuration_id,question_id",
            (run_id,),
        ).fetchall()
        attempts_by_execution: dict[str, list[sqlite3.Row]] = defaultdict(list)
        calls_by_attempt: dict[tuple[str, int], list[sqlite3.Row]] = defaultdict(list)
        grades: dict[str, sqlite3.Row] = {}
        archived: list[sqlite3.Row] = []
        if executions:
            for row in connection.execute(
                "SELECT a.* FROM attempts a JOIN executions e USING(execution_id) "
                "WHERE e.run_id=? ORDER BY a.execution_id,a.attempt_index",
                (run_id,),
            ):
                attempts_by_execution[row["execution_id"]].append(row)
            for row in connection.execute(
                "SELECT c.* FROM call_slots c JOIN executions e USING(execution_id) "
                "WHERE e.run_id=? ORDER BY c.execution_id,c.attempt_index,"
                "CASE role WHEN 'solver' THEN 0 ELSE 1 END",
                (run_id,),
            ):
                calls_by_attempt[(row["execution_id"], row["attempt_index"])].append(
                    row
                )
            grades = {
                row["execution_id"]: row
                for row in connection.execute(
                    "SELECT g.* FROM grades g JOIN executions e USING(execution_id) "
                    "WHERE e.run_id=?",
                    (run_id,),
                )
            }
            if "execution_retries" in tables:
                archived = connection.execute(
                    "SELECT r.* FROM execution_retries r "
                    "JOIN executions e USING(execution_id) WHERE e.run_id=? "
                    "ORDER BY r.execution_id,r.retry_index",
                    (run_id,),
                ).fetchall()

    header = dict(run)
    header["settings"] = _decode(header.pop("settings_json"))
    records: list[dict[str, Any]] = [
        {
            "record_type": "run",
            "log_schema_version": "detailed-workflow-log-v1",
            "source_database": str(database),
            "run": header,
            "configuration_count": len(configurations),
            "workflow_count": len(executions),
            "archived_retry_count": len(archived),
        }
    ]
    for execution_row in executions:
        execution = dict(execution_row)
        execution["question"] = _decode(execution.pop("question_json"))
        attempts = []
        workflow_calls = []
        for attempt_row in attempts_by_execution[execution["execution_id"]]:
            attempt = dict(attempt_row)
            attempt["parsed_answer"] = _decode(attempt.pop("parsed_answer_json"))
            calls = [
                _call_record(row)
                for row in calls_by_attempt[
                    (execution["execution_id"], attempt["attempt_index"])
                ]
            ]
            attempt["calls"] = calls
            attempts.append(attempt)
            workflow_calls.extend(calls)
        records.append(
            {
                "record_type": "workflow",
                "run_id": run_id,
                "configuration_id": execution["configuration_id"],
                "configuration": configurations[execution["configuration_id"]],
                "execution": execution,
                "attempts": attempts,
                "grade": (
                    dict(grades[execution["execution_id"]])
                    if execution["execution_id"] in grades
                    else None
                ),
                "call_totals": _totals(workflow_calls),
            }
        )
    for retry in archived:
        row = dict(retry)
        row["snapshot"] = _decode(row.pop("snapshot_json"))
        records.append({"record_type": "archived_retry", "run_id": run_id, **row})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "run_id": run_id,
        "workflow_count": len(executions),
        "archived_retry_count": len(archived),
        "record_count": len(records),
        "output": str(output),
    }
