"""SQLite experiment ledger with explicit planned, executed, and skipped slots."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
from uuid import uuid4

from .configuration import WorkflowConfiguration
from .domain import CompletionRequest, CompletionResponse, Question


SCHEMA_VERSION = "3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    max_retries INTEGER NOT NULL CHECK (max_retries >= 0),
    status TEXT NOT NULL CHECK (status IN ('prepared', 'running', 'completed', 'failed')),
    settings_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    finished_at_utc TEXT
);

CREATE TABLE IF NOT EXISTS configurations (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    configuration_id TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (run_id, configuration_id)
);

CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    configuration_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    question_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    returned_attempt INTEGER,
    final_answer TEXT,
    stop_reason TEXT,
    error TEXT,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    elapsed_seconds REAL CHECK (elapsed_seconds >= 0),
    FOREIGN KEY (run_id, configuration_id)
        REFERENCES configurations(run_id, configuration_id),
    UNIQUE (run_id, configuration_id, question_id)
);

CREATE TABLE IF NOT EXISTS attempts (
    execution_id TEXT NOT NULL REFERENCES executions(execution_id),
    attempt_index INTEGER NOT NULL CHECK (attempt_index >= 1),
    solver_model TEXT NOT NULL,
    verifier_model TEXT,
    status TEXT NOT NULL CHECK (status IN
        ('planned', 'reached', 'completed', 'skipped_early_stop', 'failed')),
    solver_answer TEXT,
    parsed_answer_json TEXT,
    selected_option TEXT,
    solver_parse_error TEXT,
    solver_option_correct INTEGER CHECK (solver_option_correct IN (0, 1)),
    verifier_accepted INTEGER CHECK (verifier_accepted IN (0, 1)),
    verifier_parse_error TEXT,
    PRIMARY KEY (execution_id, attempt_index)
);

CREATE TABLE IF NOT EXISTS call_slots (
    execution_id TEXT NOT NULL,
    attempt_index INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('solver', 'verifier')),
    model TEXT,
    status TEXT NOT NULL CHECK (status IN
        ('planned', 'executed', 'failed', 'skipped_early_stop',
         'skipped_execution_failure', 'not_applicable_final')),
    request_json TEXT,
    started_at_utc TEXT,
    finished_at_utc TEXT,
    response_json TEXT,
    response_body_text TEXT,
    response_text TEXT,
    returned_model TEXT,
    provider TEXT,
    response_id TEXT,
    http_status INTEGER,
    finish_reason TEXT,
    usage_json TEXT,
    input_tokens INTEGER CHECK (input_tokens >= 0),
    output_tokens INTEGER CHECK (output_tokens >= 0),
    reasoning_tokens INTEGER CHECK (reasoning_tokens >= 0),
    reported_cost_usd REAL CHECK (reported_cost_usd >= 0),
    elapsed_seconds REAL CHECK (elapsed_seconds >= 0),
    accepted INTEGER CHECK (accepted IN (0, 1)),
    parse_error TEXT,
    error_type TEXT,
    error_message TEXT,
    error TEXT,
    PRIMARY KEY (execution_id, attempt_index, role),
    FOREIGN KEY (execution_id, attempt_index)
        REFERENCES attempts(execution_id, attempt_index)
);

CREATE TABLE IF NOT EXISTS grades (
    execution_id TEXT PRIMARY KEY REFERENCES executions(execution_id),
    grader_version TEXT NOT NULL,
    correct_option TEXT NOT NULL,
    returned_option TEXT,
    is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
    graded_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_retries (
    execution_id TEXT NOT NULL REFERENCES executions(execution_id),
    retry_index INTEGER NOT NULL CHECK (retry_index >= 1),
    snapshot_json TEXT NOT NULL,
    archived_at_utc TEXT NOT NULL,
    PRIMARY KEY (execution_id, retry_index)
);
"""


ADDITIVE_COLUMNS = {
    "executions": {
        "elapsed_seconds": "REAL CHECK (elapsed_seconds >= 0)",
    },
    "attempts": {
        "solver_parse_error": "TEXT",
        "solver_option_correct": "INTEGER CHECK (solver_option_correct IN (0, 1))",
    },
    "call_slots": {
        "started_at_utc": "TEXT",
        "finished_at_utc": "TEXT",
        "response_body_text": "TEXT",
        "response_id": "TEXT",
        "http_status": "INTEGER",
        "finish_reason": "TEXT",
        "usage_json": "TEXT",
        "reasoning_tokens": "INTEGER CHECK (reasoning_tokens >= 0)",
        "error_type": "TEXT",
        "error_message": "TEXT",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ExperimentStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if existing is not None and existing[0] not in ("1", "2", SCHEMA_VERSION):
                raise ValueError("Unsupported database schema version.")
            for table, columns in ADDITIVE_COLUMNS.items():
                present = {
                    row["name"]
                    for row in connection.execute(f"PRAGMA table_info({table})")
                }
                for name, declaration in columns.items():
                    if name not in present:
                        connection.execute(
                            f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                        )
            connection.execute(
                "INSERT INTO metadata(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (SCHEMA_VERSION,),
            )

    def create_run(
        self,
        name: str,
        *,
        max_retries: int,
        settings: Mapping[str, Any] | None = None,
    ) -> str:
        if not name:
            raise ValueError("Run name is required.")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer.")
        run_id = str(uuid4())
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,NULL)",
                (run_id, name, max_retries, "prepared", _json(settings or {}), _now()),
            )
        return run_id

    def finish_run(self, run_id: str, *, status: str) -> None:
        if status not in ("completed", "failed"):
            raise ValueError("A run can finish only as completed or failed.")
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE runs SET status=?,finished_at_utc=? "
                "WHERE run_id=? AND status IN ('prepared','running')",
                (status, _now(), run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Run is missing or already terminal.")

    def resume_run(self, run_id: str) -> set[tuple[str, str]]:
        """Prepare an interrupted run and return completed configuration-question keys."""
        interruption = "Execution was interrupted before completion and queued for resume."
        with closing(self.connect()) as connection, connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if run is None:
                raise ValueError("Run does not exist.")
            if run["status"] == "completed":
                raise ValueError("A completed run cannot be resumed.")
            running_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT execution_id FROM executions "
                    "WHERE run_id=? AND status='running'",
                    (run_id,),
                )
            ]
            for execution_id in running_ids:
                connection.execute(
                    "UPDATE attempts SET status='failed' WHERE execution_id=? "
                    "AND status IN ('planned','reached')",
                    (execution_id,),
                )
                connection.execute(
                    "UPDATE call_slots SET status='skipped_execution_failure',error=? "
                    "WHERE execution_id=? AND status='planned'",
                    (interruption, execution_id),
                )
                connection.execute(
                    "UPDATE executions SET status='failed',error=?,finished_at_utc=? "
                    "WHERE execution_id=?",
                    (interruption, _now(), execution_id),
                )
            connection.execute(
                "UPDATE runs SET status='running',finished_at_utc=NULL WHERE run_id=?",
                (run_id,),
            )
            return {
                (str(row[0]), str(row[1]))
                for row in connection.execute(
                    "SELECT configuration_id,question_id FROM executions "
                    "WHERE run_id=? AND status='completed'",
                    (run_id,),
                )
            }

    def register_configuration(
        self, run_id: str, configuration: WorkflowConfiguration
    ) -> str:
        with closing(self.connect()) as connection, connection:
            run = connection.execute(
                "SELECT max_retries FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if run is None:
                raise ValueError("Run does not exist.")
            if run[0] != configuration.max_retries:
                raise ValueError("Configuration retry count differs from the run.")
            connection.execute(
                "INSERT OR IGNORE INTO configurations VALUES (?,?,?)",
                (run_id, configuration.configuration_id, _json(configuration.snapshot())),
            )
        return configuration.configuration_id

    def start_execution(
        self,
        run_id: str,
        configuration: WorkflowConfiguration,
        question: Question,
    ) -> str:
        configuration_id = self.register_configuration(run_id, configuration)
        execution_id = str(uuid4())
        question_snapshot = {
            "question_id": question.question_id,
            "problem": question.problem,
            "options": question.options,
        }
        with closing(self.connect()) as connection, connection:
            existing = connection.execute(
                "SELECT * FROM executions WHERE run_id=? AND configuration_id=? "
                "AND question_id=?",
                (run_id, configuration_id, question.question_id),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO executions(execution_id,run_id,configuration_id,"
                    "question_id,question_json,status,started_at_utc) "
                    "VALUES (?,?,?,?,?,'running',?)",
                    (
                        execution_id,
                        run_id,
                        configuration_id,
                        question.question_id,
                        _json(question_snapshot),
                        _now(),
                    ),
                )
            elif existing["status"] == "failed":
                execution_id = str(existing["execution_id"])
                self._archive_execution(connection, execution_id)
                connection.execute(
                    "DELETE FROM grades WHERE execution_id=?", (execution_id,)
                )
                connection.execute(
                    "DELETE FROM call_slots WHERE execution_id=?", (execution_id,)
                )
                connection.execute(
                    "DELETE FROM attempts WHERE execution_id=?", (execution_id,)
                )
                connection.execute(
                    "UPDATE executions SET question_json=?,status='running',"
                    "returned_attempt=NULL,final_answer=NULL,stop_reason=NULL,error=NULL,"
                    "started_at_utc=?,finished_at_utc=NULL,elapsed_seconds=NULL "
                    "WHERE execution_id=?",
                    (_json(question_snapshot), _now(), execution_id),
                )
            else:
                raise ValueError(
                    "Execution already exists and is not eligible for resume."
                )
            connection.execute(
                "UPDATE runs SET status='running' WHERE run_id=? AND status='prepared'",
                (run_id,),
            )
            for index, solver_model in enumerate(configuration.solver_models, start=1):
                verifier_model = (
                    configuration.verifier_models[index - 1]
                    if index <= configuration.max_retries
                    else None
                )
                connection.execute(
                    "INSERT INTO attempts(execution_id,attempt_index,solver_model,"
                    "verifier_model,status) VALUES (?,?,?,?, 'planned')",
                    (execution_id, index, solver_model, verifier_model),
                )
                connection.execute(
                    "INSERT INTO call_slots(execution_id,attempt_index,role,model,status) "
                    "VALUES (?,?, 'solver',?, 'planned')",
                    (execution_id, index, solver_model),
                )
                connection.execute(
                    "INSERT INTO call_slots(execution_id,attempt_index,role,model,status) "
                    "VALUES (?,?, 'verifier',?,?)",
                    (
                        execution_id,
                        index,
                        verifier_model,
                        "planned" if verifier_model is not None else "not_applicable_final",
                    ),
                )
        return execution_id

    def start_call(
        self,
        execution_id: str,
        attempt_index: int,
        role: str,
        request: CompletionRequest,
        *,
        started_at_utc: str,
    ) -> None:
        """Durably record request initiation before sending it to OpenRouter."""
        if role not in ("solver", "verifier"):
            raise ValueError("Unknown call role.")
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE call_slots SET request_json=?,started_at_utc=? "
                "WHERE execution_id=? AND attempt_index=? AND role=? "
                "AND status='planned' AND started_at_utc IS NULL",
                (
                    _json(request.as_payload()),
                    started_at_utc,
                    execution_id,
                    attempt_index,
                    role,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Call slot is missing or already started.")

    @staticmethod
    def _archive_execution(
        connection: sqlite3.Connection, execution_id: str
    ) -> None:
        execution = connection.execute(
            "SELECT * FROM executions WHERE execution_id=?", (execution_id,)
        ).fetchone()
        if execution is None:
            raise ValueError("Execution does not exist.")
        attempts = connection.execute(
            "SELECT * FROM attempts WHERE execution_id=? ORDER BY attempt_index",
            (execution_id,),
        ).fetchall()
        calls = connection.execute(
            "SELECT * FROM call_slots WHERE execution_id=? ORDER BY attempt_index,role",
            (execution_id,),
        ).fetchall()
        grade = connection.execute(
            "SELECT * FROM grades WHERE execution_id=?", (execution_id,)
        ).fetchone()
        next_index = connection.execute(
            "SELECT COALESCE(MAX(retry_index),0)+1 FROM execution_retries "
            "WHERE execution_id=?",
            (execution_id,),
        ).fetchone()[0]
        snapshot = {
            "execution": dict(execution),
            "attempts": [dict(row) for row in attempts],
            "call_slots": [dict(row) for row in calls],
            "grade": dict(grade) if grade is not None else None,
        }
        connection.execute(
            "INSERT INTO execution_retries VALUES (?,?,?,?)",
            (execution_id, next_index, _json(snapshot), _now()),
        )

    def record_call(
        self,
        execution_id: str,
        attempt_index: int,
        role: str,
        response: CompletionResponse,
        *,
        elapsed_seconds: float,
        accepted: bool | None = None,
        parse_error: str | None = None,
    ) -> None:
        if role not in ("solver", "verifier"):
            raise ValueError("Unknown call role.")
        if role == "solver" and accepted is not None:
            raise ValueError("Solver calls cannot have verifier verdicts.")
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE call_slots SET status='executed',finished_at_utc=?,response_json=?,"
                "response_body_text=?,response_text=?,returned_model=?,provider=?,"
                "response_id=?,http_status=?,finish_reason=?,usage_json=?,input_tokens=?,"
                "output_tokens=?,reasoning_tokens=?,reported_cost_usd=?,elapsed_seconds=?,"
                "accepted=?,parse_error=? WHERE execution_id=? AND attempt_index=? "
                "AND role=? AND status='planned' AND started_at_utc IS NOT NULL",
                (
                    _now(),
                    _json(response.raw_response),
                    response.response_body_text,
                    response.content,
                    response.returned_model,
                    response.provider,
                    response.response_id,
                    response.http_status,
                    response.finish_reason,
                    _json(response.usage),
                    response.input_tokens,
                    response.output_tokens,
                    response.reasoning_tokens,
                    response.cost_usd,
                    elapsed_seconds,
                    None if accepted is None else int(accepted),
                    parse_error,
                    execution_id,
                    attempt_index,
                    role,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Call slot is missing or not executable.")

    def record_call_failure(
        self,
        execution_id: str,
        attempt_index: int,
        role: str,
        error: str,
        *,
        elapsed_seconds: float,
        response: CompletionResponse | None = None,
        raw_response: Mapping[str, Any] | None = None,
        response_body_text: str | None = None,
        http_status: int | None = None,
        error_type: str | None = None,
    ) -> None:
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE call_slots SET status='failed',finished_at_utc=?,response_json=?,"
                "response_body_text=?,response_text=?,returned_model=?,provider=?,"
                "response_id=?,http_status=?,finish_reason=?,usage_json=?,input_tokens=?,"
                "output_tokens=?,reasoning_tokens=?,reported_cost_usd=?,error_type=?,"
                "error_message=?,error=?,elapsed_seconds=? WHERE execution_id=? "
                "AND attempt_index=? AND role=? AND status='planned' "
                "AND started_at_utc IS NOT NULL",
                (
                    _now(),
                    _json(response.raw_response)
                    if response is not None
                    else (_json(raw_response) if raw_response is not None else None),
                    response.response_body_text
                    if response is not None
                    else response_body_text,
                    response.content if response is not None else None,
                    response.returned_model if response is not None else None,
                    response.provider if response is not None else None,
                    response.response_id if response is not None else None,
                    response.http_status if response is not None else http_status,
                    response.finish_reason if response is not None else None,
                    _json(response.usage) if response is not None else None,
                    response.input_tokens if response is not None else None,
                    response.output_tokens if response is not None else None,
                    response.reasoning_tokens if response is not None else None,
                    response.cost_usd if response is not None else None,
                    error_type,
                    error,
                    error,
                    elapsed_seconds,
                    execution_id,
                    attempt_index,
                    role,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Call slot is missing or not executable.")

    def record_solver_answer(
        self,
        execution_id: str,
        attempt_index: int,
        answer: str,
        parsed_answer: Mapping[str, str] | None,
        selected_option: str | None,
        parse_error: str | None,
    ) -> None:
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE attempts SET status='reached',solver_answer=?,parsed_answer_json=?,"
                "selected_option=?,solver_parse_error=? WHERE execution_id=? "
                "AND attempt_index=? AND status='planned'",
                (
                    answer,
                    _json(parsed_answer) if parsed_answer is not None else None,
                    selected_option,
                    parse_error,
                    execution_id,
                    attempt_index,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Attempt is missing or already reached.")

    def record_verifier_result(
        self,
        execution_id: str,
        attempt_index: int,
        accepted: bool | None,
        parse_error: str | None,
    ) -> None:
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE attempts SET verifier_accepted=?,verifier_parse_error=? "
                "WHERE execution_id=? AND attempt_index=? AND status='reached'",
                (
                    None if accepted is None else int(accepted),
                    parse_error,
                    execution_id,
                    attempt_index,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Verifier result needs a reached attempt.")

    def complete_attempt(self, execution_id: str, attempt_index: int) -> None:
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE attempts SET status='completed' "
                "WHERE execution_id=? AND attempt_index=? AND status='reached'",
                (execution_id, attempt_index),
            )
            if cursor.rowcount != 1:
                raise ValueError("Attempt is missing or not reached.")

    def finish_execution(
        self,
        execution_id: str,
        *,
        returned_attempt: int,
        final_answer: str,
        stop_reason: str,
        elapsed_seconds: float,
    ) -> None:
        if stop_reason not in ("verifier_accepted", "retry_limit_reached"):
            raise ValueError("Unknown successful stop reason.")
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "UPDATE attempts SET status='skipped_early_stop' "
                "WHERE execution_id=? AND attempt_index>? AND status='planned'",
                (execution_id, returned_attempt),
            )
            connection.execute(
                "UPDATE call_slots SET status='skipped_early_stop' "
                "WHERE execution_id=? AND attempt_index>? AND status='planned'",
                (execution_id, returned_attempt),
            )
            cursor = connection.execute(
                "UPDATE executions SET status='completed',returned_attempt=?,final_answer=?,"
                "stop_reason=?,finished_at_utc=?,elapsed_seconds=? "
                "WHERE execution_id=? AND status='running'",
                (
                    returned_attempt,
                    final_answer,
                    stop_reason,
                    _now(),
                    elapsed_seconds,
                    execution_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Execution is missing or already terminal.")

    def fail_execution(
        self, execution_id: str, error: str, *, elapsed_seconds: float
    ) -> None:
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "UPDATE attempts SET status='failed' "
                "WHERE execution_id=? AND status IN ('planned','reached')",
                (execution_id,),
            )
            connection.execute(
                "UPDATE call_slots SET status='skipped_execution_failure' "
                "WHERE execution_id=? AND status='planned'",
                (execution_id,),
            )
            connection.execute(
                "UPDATE executions SET status='failed',error=?,finished_at_utc=?,"
                "elapsed_seconds=? "
                "WHERE execution_id=? AND status='running'",
                (error, _now(), elapsed_seconds, execution_id),
            )

    def save_grade(
        self,
        execution_id: str,
        *,
        grader_version: str,
        correct_option: str,
        returned_option: str | None,
        is_correct: bool,
    ) -> None:
        with closing(self.connect()) as connection, connection:
            execution = connection.execute(
                "SELECT status FROM executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            if execution is None or execution[0] != "completed":
                raise ValueError("Only completed executions can be graded.")
            connection.execute(
                "INSERT INTO grades VALUES (?,?,?,?,?,?)",
                (
                    execution_id,
                    grader_version,
                    correct_option,
                    returned_option,
                    int(is_correct),
                    _now(),
                ),
            )
            connection.execute(
                "UPDATE attempts SET solver_option_correct=(selected_option=?) "
                "WHERE execution_id=? AND selected_option IS NOT NULL",
                (correct_option, execution_id),
            )

    def execution_record(self, execution_id: str) -> dict[str, Any]:
        with closing(self.connect()) as connection:
            execution = connection.execute(
                "SELECT * FROM executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            if execution is None:
                raise ValueError("Execution does not exist.")
            attempts = connection.execute(
                "SELECT * FROM attempts WHERE execution_id=? ORDER BY attempt_index",
                (execution_id,),
            ).fetchall()
            calls = connection.execute(
                "SELECT * FROM call_slots WHERE execution_id=? "
                "ORDER BY attempt_index, CASE role WHEN 'solver' THEN 0 ELSE 1 END",
                (execution_id,),
            ).fetchall()
            grade = connection.execute(
                "SELECT * FROM grades WHERE execution_id=?", (execution_id,)
            ).fetchone()
        return {
            "execution": dict(execution),
            "attempts": [dict(row) for row in attempts],
            "calls": [dict(row) for row in calls],
            "grade": dict(grade) if grade is not None else None,
        }

    def executed_calls(self, execution_id: str) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM call_slots WHERE execution_id=? AND status='executed' "
                "ORDER BY attempt_index, CASE role WHEN 'solver' THEN 0 ELSE 1 END",
                (execution_id,),
            ).fetchall()
        return [dict(row) for row in rows]

