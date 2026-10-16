"""Small, auditable OpenRouter pilot runs over a fixed MathQA subset."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .benchmark import BenchmarkItem
from .configuration import WorkflowConfiguration, generate_configurations
from .domain import CompletionClient
from .grading import grade_execution
from .profiles import (
    GPT_OSS_20B,
    MINISTRAL_3_8B,
    QIAN_MODEL_IDS,
    QIAN_MODEL_PROFILES,
    QWEN3,
)
from .storage import ExperimentStore
from .workflow import WorkflowRunner


SMOKE_PRESET = "smoke"
QIAN_27_PRESET = "qian-27"
PILOT_PRESETS = (SMOKE_PRESET, QIAN_27_PRESET)


@dataclass(frozen=True)
class PilotPlan:
    preset: str
    question_count: int
    partition_counts: dict[str, int]
    configuration_count: int
    workflow_count: int
    minimum_calls: int
    maximum_calls: int
    model_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "question_count": self.question_count,
            "partition_counts": self.partition_counts,
            "configuration_count": self.configuration_count,
            "workflow_count": self.workflow_count,
            "minimum_calls": self.minimum_calls,
            "maximum_calls": self.maximum_calls,
            "model_ids": list(self.model_ids),
        }


def configurations_for_preset(
    preset: str, configuration_ids: set[str] | None = None
) -> tuple[WorkflowConfiguration, ...]:
    if preset == SMOKE_PRESET:
        configurations = (
            WorkflowConfiguration(
                solver_models=(GPT_OSS_20B, MINISTRAL_3_8B),
                verifier_models=(QWEN3,),
                max_retries=1,
            ),
        )
    elif preset == QIAN_27_PRESET:
        configurations = tuple(
            generate_configurations(
                QIAN_MODEL_IDS, QIAN_MODEL_IDS, max_retries=1
            )
        )
    else:
        raise ValueError(f"Unknown pilot preset: {preset!r}.")
    if configuration_ids is None:
        return configurations
    available = {row.configuration_id for row in configurations}
    unknown = configuration_ids - available
    if unknown:
        raise ValueError(f"Unknown configuration IDs: {sorted(unknown)}")
    selected = tuple(
        row for row in configurations if row.configuration_id in configuration_ids
    )
    if not selected:
        raise ValueError("At least one configuration ID is required.")
    return selected


def plan_pilot(
    preset: str,
    items: Sequence[BenchmarkItem],
    configuration_ids: set[str] | None = None,
) -> PilotPlan:
    if not items:
        raise ValueError("The pilot needs at least one question.")
    configurations = configurations_for_preset(preset, configuration_ids)
    workflow_count = len(items) * len(configurations)
    max_retries = configurations[0].max_retries
    minimum_calls_per_workflow = 1 if max_retries == 0 else 2
    maximum_calls_per_workflow = 2 * max_retries + 1
    models = tuple(
        dict.fromkeys(
            model
            for configuration in configurations
            for model in configuration.solver_models + configuration.verifier_models
        )
    )
    return PilotPlan(
        preset=preset,
        question_count=len(items),
        partition_counts=dict(Counter(item.partition for item in items)),
        configuration_count=len(configurations),
        workflow_count=workflow_count,
        minimum_calls=workflow_count * minimum_calls_per_workflow,
        maximum_calls=workflow_count * maximum_calls_per_workflow,
        model_ids=models,
    )


def run_pilot(
    *,
    store: ExperimentStore,
    client: CompletionClient,
    items: Sequence[BenchmarkItem],
    preset: str,
    run_name: str,
    settings: dict[str, object],
    resume_run_id: str | None = None,
    configuration_ids: set[str] | None = None,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[str, dict[str, object]]:
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer.")
    configurations = configurations_for_preset(preset, configuration_ids)
    if resume_run_id is None:
        run_id = store.create_run(run_name, max_retries=1, settings=settings)
        completed_keys: set[tuple[str, str]] = set()
    else:
        run_id = resume_run_id
        completed_keys = store.resume_run(run_id)
    completed = 0

    def execute(configuration: WorkflowConfiguration, item: BenchmarkItem) -> None:
        outcome = WorkflowRunner(client, store, QIAN_MODEL_PROFILES).run_question(
            run_id, configuration, item.question()
        )
        grade_execution(store, outcome.execution_id, correct_option=item.correct_option)

    tasks = [
        (configuration, item)
        for configuration in configurations
        for item in items
        if (configuration.configuration_id, item.question_id) not in completed_keys
    ]
    total = len(tasks)
    try:
        if workers == 1:
            for configuration, item in tasks:
                execute(configuration, item)
                completed += 1
                if progress is not None:
                    progress(completed, total)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                task_iterator = iter(tasks)
                pending = {
                    executor.submit(execute, *next(task_iterator))
                    for _ in range(min(workers, total))
                }
                while pending:
                    finished, pending = wait(
                        pending, return_when=FIRST_COMPLETED
                    )
                    for future in finished:
                        future.result()
                    for _ in finished:
                        completed += 1
                        if progress is not None:
                            progress(completed, total)
                        try:
                            task = next(task_iterator)
                        except StopIteration:
                            continue
                        pending.add(executor.submit(execute, *task))
    except BaseException:
        store.finish_run(run_id, status="failed")
        raise
    store.finish_run(run_id, status="completed")
    return run_id, summarize_run(store, run_id, items, preset)


def summarize_run(
    store: ExperimentStore,
    run_id: str,
    items: Sequence[BenchmarkItem],
    preset: str,
) -> dict[str, object]:
    partition_by_question = {item.question_id: item.partition for item in items}
    with closing(store.connect()) as connection:
        run = connection.execute(
            "SELECT status,created_at_utc,finished_at_utc FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        executions = connection.execute(
            "SELECT question_id,status,stop_reason,elapsed_seconds "
            "FROM executions WHERE run_id=?",
            (run_id,),
        ).fetchall()
        grades = connection.execute(
            "SELECT e.question_id,g.is_correct FROM executions e "
            "JOIN grades g ON g.execution_id=e.execution_id WHERE e.run_id=?",
            (run_id,),
        ).fetchall()
        calls = connection.execute(
            "SELECT c.status,c.role,c.model,c.input_tokens,c.output_tokens,"
            "c.reasoning_tokens,c.reported_cost_usd,c.elapsed_seconds FROM call_slots c "
            "JOIN executions e ON e.execution_id=c.execution_id WHERE e.run_id=?",
            (run_id,),
        ).fetchall()
        configuration_count = connection.execute(
            "SELECT COUNT(*) FROM configurations WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        archived_retry_count = connection.execute(
            "SELECT COUNT(*) FROM execution_retries r "
            "JOIN executions e USING(execution_id) WHERE e.run_id=?",
            (run_id,),
        ).fetchone()[0]
    if run is None:
        raise ValueError("Run does not exist.")
    grade_counts: dict[str, dict[str, int]] = {}
    for partition in sorted(set(partition_by_question.values())):
        partition_grades = [
            row for row in grades if partition_by_question[row["question_id"]] == partition
        ]
        grade_counts[partition] = {
            "graded": len(partition_grades),
            "correct": sum(int(row["is_correct"]) for row in partition_grades),
        }
    executed = [row for row in calls if row["status"] == "executed"]
    known_costs = [row["reported_cost_usd"] for row in executed if row["reported_cost_usd"] is not None]
    usage_by_role_and_model: dict[str, dict[str, int | float]] = {}
    for role, model in sorted({(row["role"], row["model"]) for row in executed}):
        rows = [
            row
            for row in executed
            if row["role"] == role and row["model"] == model
        ]
        usage_by_role_and_model[f"{role}:{model}"] = {
            "calls": len(rows),
            "input_tokens": sum(row["input_tokens"] or 0 for row in rows),
            "output_tokens": sum(row["output_tokens"] or 0 for row in rows),
            "reasoning_tokens": sum(row["reasoning_tokens"] or 0 for row in rows),
            "reported_cost_usd": sum(
                row["reported_cost_usd"] or 0.0 for row in rows
            ),
            "calls_with_unknown_cost": sum(
                row["reported_cost_usd"] is None for row in rows
            ),
            "elapsed_seconds": sum(row["elapsed_seconds"] or 0.0 for row in rows),
        }
    return {
        "schema_version": "openrouter-pilot-summary-v2",
        "run_id": run_id,
        "preset": preset,
        "status": run["status"],
        "created_at_utc": run["created_at_utc"],
        "finished_at_utc": run["finished_at_utc"],
        "configuration_count": configuration_count,
        "archived_technical_retry_count": archived_retry_count,
        "execution_counts": dict(Counter(row["status"] for row in executions)),
        "stop_reason_counts": dict(Counter(row["stop_reason"] for row in executions)),
        "call_status_counts": dict(Counter(row["status"] for row in calls)),
        "executed_call_count": len(executed),
        "reported_cost_usd": sum(float(cost) for cost in known_costs),
        "executed_calls_with_unknown_cost": len(executed) - len(known_costs),
        "input_tokens": sum(row["input_tokens"] or 0 for row in executed),
        "output_tokens": sum(row["output_tokens"] or 0 for row in executed),
        "reasoning_tokens": sum(row["reasoning_tokens"] or 0 for row in executed),
        "call_elapsed_seconds": sum(
            row["elapsed_seconds"] or 0.0 for row in executed
        ),
        "workflow_elapsed_seconds": sum(
            row["elapsed_seconds"] or 0.0 for row in executions
        ),
        "usage_by_role_and_model": usage_by_role_and_model,
        "grade_counts": grade_counts,
    }
