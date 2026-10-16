#!/usr/bin/env python3
"""Export completed, graded workflow executions into the static-panel JSONL schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retry_eval.static_baselines import BalancedPanel, PanelRecord  # noqa: E402
from retry_eval.benchmark import load_mathqa_benchmark  # noqa: E402


def export_panel(
    database: Path,
    run_id: str,
    output: Path,
    *,
    partition_by_question: dict[str, str] | None = None,
) -> list[PanelRecord]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT e.execution_id, e.configuration_id, e.question_id, g.is_correct,
                   c.snapshot_json
            FROM executions AS e
            JOIN grades AS g ON g.execution_id = e.execution_id
            JOIN configurations AS c
              ON c.run_id = e.run_id AND c.configuration_id = e.configuration_id
            WHERE e.run_id = ? AND e.status = 'completed'
            ORDER BY e.configuration_id, e.question_id, e.execution_id
            """,
            (run_id,),
        ).fetchall()
        records: list[PanelRecord] = []
        for row in rows:
            if (
                partition_by_question is not None
                and row["question_id"] not in partition_by_question
            ):
                raise ValueError(
                    f"Question {row['question_id']} is absent from the supplied benchmark."
                )
            costs = connection.execute(
                """
                SELECT reported_cost_usd
                FROM call_slots
                WHERE execution_id = ? AND status = 'executed'
                """,
                (row["execution_id"],),
            ).fetchall()
            if not costs:
                raise ValueError(f"Execution {row['execution_id']} has no executed calls.")
            if any(cost[0] is None for cost in costs):
                raise ValueError(
                    f"Execution {row['execution_id']} has an executed call without reported USD cost."
                )
            records.append(
                PanelRecord(
                    record_id=str(row["execution_id"]),
                    configuration_id=str(row["configuration_id"]),
                    question_id=str(row["question_id"]),
                    reward=float(row["is_correct"]),
                    realized_cost_usd=sum(float(cost[0]) for cost in costs),
                    partition=(
                        partition_by_question[row["question_id"]]
                        if partition_by_question is not None
                        else "development"
                    ),
                    configuration_snapshot=json.loads(row["snapshot_json"]),
                )
            )
    finally:
        connection.close()
    BalancedPanel(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(
                {
                    "record_id": row.record_id,
                    "configuration_id": row.configuration_id,
                    "question_id": row.question_id,
                    "reward": row.reward,
                    "realized_cost_usd": row.realized_cost_usd,
                    "partition": row.partition,
                    "configuration_snapshot": row.configuration_snapshot,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        type=Path,
        help="Benchmark JSONL used to attach development/reserved_test partitions.",
    )
    args = parser.parse_args()
    partition_by_question = None
    if args.benchmark is not None:
        partition_by_question = {
            item.question_id: item.partition
            for item in load_mathqa_benchmark(args.benchmark)
        }
    records = export_panel(
        args.database,
        args.run_id,
        args.output,
        partition_by_question=partition_by_question,
    )
    print(f"exported_records={len(records)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
