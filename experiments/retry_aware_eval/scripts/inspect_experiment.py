#!/usr/bin/env python3
"""Inspect an experiment SQLite ledger without modifying it."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import statistics


TABLE_PURPOSES = {
    "metadata": "database schema version",
    "runs": "run status, frozen settings, and timestamps",
    "configurations": "immutable solver/verifier sequence snapshots",
    "executions": "one configuration-question workflow and its final answer",
    "attempts": "per-retry answers, parsed options, and verifier decisions",
    "call_slots": "every planned/executed/skipped API call, raw I/O, usage, cost, and latency",
    "grades": "gold option, returned option, and final correctness",
    "execution_retries": "archived failed/interrupted workflow attempts retained across resume",
}


def inspect_database(database: Path, run_id: str | None, recent: int) -> dict[str, object]:
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    try:
        schema = {}
        for table, purpose in TABLE_PURPOSES.items():
            columns = [
                row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
            ]
            schema[table] = {"purpose": purpose, "columns": columns}
        call_columns = set(schema["call_slots"]["columns"])
        execution_columns = set(schema["executions"]["columns"])

        if run_id is None:
            row = connection.execute(
                "SELECT run_id FROM runs ORDER BY created_at_utc DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValueError("The database contains no runs.")
            run_id = str(row["run_id"])
        run = connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise ValueError(f"Run {run_id!r} does not exist.")

        execution_counts = dict(
            connection.execute(
                "SELECT status,COUNT(*) FROM executions WHERE run_id=? GROUP BY status",
                (run_id,),
            ).fetchall()
        )
        call_counts = dict(
            connection.execute(
                "SELECT c.status,COUNT(*) FROM call_slots c "
                "JOIN executions e USING(execution_id) WHERE e.run_id=? GROUP BY c.status",
                (run_id,),
            ).fetchall()
        )
        reasoning_select = (
            "c.reasoning_tokens" if "reasoning_tokens" in call_columns else "NULL"
        )
        calls = connection.execute(
            "SELECT c.role,c.model,c.provider,c.input_tokens,c.output_tokens,"
            f"{reasoning_select} AS reasoning_tokens,"
            "c.reported_cost_usd,c.elapsed_seconds FROM call_slots c "
            "JOIN executions e USING(execution_id) "
            "WHERE e.run_id=? AND c.status='executed'",
            (run_id,),
        ).fetchall()
        call_groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for call in calls:
            call_groups[f"{call['role']}:{call['model']}"].append(call)
        latency = {
            key: {
                "calls": len(rows),
                "input_tokens": sum(row["input_tokens"] or 0 for row in rows),
                "output_tokens": sum(row["output_tokens"] or 0 for row in rows),
                "reasoning_tokens": sum(
                    row["reasoning_tokens"] or 0 for row in rows
                ) if "reasoning_tokens" in call_columns else None,
                "reported_cost_usd": sum(
                    row["reported_cost_usd"] or 0.0 for row in rows
                ),
                "calls_with_unknown_cost": sum(
                    row["reported_cost_usd"] is None for row in rows
                ),
                "total_seconds": sum(row["elapsed_seconds"] for row in rows),
                "mean_seconds": statistics.fmean(
                    row["elapsed_seconds"] for row in rows
                ),
                "median_seconds": statistics.median(
                    row["elapsed_seconds"] for row in rows
                ),
                "max_seconds": max(row["elapsed_seconds"] for row in rows),
            }
            for key, rows in sorted(call_groups.items())
        }
        optional_slowest = ""
        if "reasoning_tokens" in call_columns:
            optional_slowest += ",c.reasoning_tokens"
        if "finish_reason" in call_columns:
            optional_slowest += ",c.finish_reason"
        if "response_id" in call_columns:
            optional_slowest += ",c.response_id"
        slowest = [
            dict(row)
            for row in connection.execute(
                "SELECT e.question_id,c.attempt_index,c.role,c.model,c.provider,"
                "c.input_tokens,c.output_tokens,c.reported_cost_usd,c.elapsed_seconds"
                f"{optional_slowest} "
                "FROM call_slots c JOIN executions e USING(execution_id) "
                "WHERE e.run_id=? AND c.status='executed' "
                "ORDER BY c.elapsed_seconds DESC LIMIT ?",
                (run_id, recent),
            )
        ]
        active = [
            dict(row)
            for row in connection.execute(
                "SELECT question_id,configuration_id,started_at_utc FROM executions "
                "WHERE run_id=? AND status='running' ORDER BY started_at_utc",
                (run_id,),
            )
        ]
        wall_seconds = None
        if run["finished_at_utc"] is not None:
            wall_seconds = (
                datetime.fromisoformat(run["finished_at_utc"])
                - datetime.fromisoformat(run["created_at_utc"])
            ).total_seconds()
        workflow_elapsed_seconds = None
        if "elapsed_seconds" in execution_columns:
            workflow_elapsed_seconds = connection.execute(
                "SELECT SUM(elapsed_seconds) FROM executions WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
        return {
            "database": str(database),
            "schema": schema,
            "run": {
                "run_id": run_id,
                "name": run["name"],
                "status": run["status"],
                "created_at_utc": run["created_at_utc"],
                "finished_at_utc": run["finished_at_utc"],
                "wall_seconds": wall_seconds,
                "settings": json.loads(run["settings_json"]),
            },
            "execution_status_counts": execution_counts,
            "call_status_counts": call_counts,
            "executed_call_totals": {
                "calls": len(calls),
                "input_tokens": sum(row["input_tokens"] or 0 for row in calls),
                "output_tokens": sum(row["output_tokens"] or 0 for row in calls),
                "reasoning_tokens": sum(
                    row["reasoning_tokens"] or 0 for row in calls
                ) if "reasoning_tokens" in call_columns else None,
                "reported_cost_usd": sum(
                    row["reported_cost_usd"] or 0.0 for row in calls
                ),
                "elapsed_seconds": sum(row["elapsed_seconds"] or 0.0 for row in calls),
                "workflow_elapsed_seconds": workflow_elapsed_seconds,
            },
            "latency_by_role_and_model": latency,
            "active_executions": active,
            "slowest_executed_calls": slowest,
        }
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--recent", type=int, default=10)
    args = parser.parse_args()
    if args.recent <= 0:
        parser.error("--recent must be positive")
    print(
        json.dumps(
            inspect_database(args.database, args.run_id, args.recent),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
