"""Independent MathQA answer-key grading."""

from __future__ import annotations

from dataclasses import dataclass

from .storage import ExperimentStore


GRADER_VERSION = "mathqa-option-v1"


@dataclass(frozen=True)
class Grade:
    execution_id: str
    returned_option: str | None
    correct_option: str
    is_correct: bool


def grade_execution(
    store: ExperimentStore, execution_id: str, *, correct_option: str
) -> Grade:
    if correct_option not in tuple("abcde"):
        raise ValueError("MathQA correct_option must be one lowercase letter a-e.")
    record = store.execution_record(execution_id)
    execution = record["execution"]
    if execution["status"] != "completed":
        raise ValueError("Only completed executions can be graded.")
    returned_attempt = execution["returned_attempt"]
    attempt = next(
        row for row in record["attempts"] if row["attempt_index"] == returned_attempt
    )
    returned_option = attempt["selected_option"]
    is_correct = returned_option == correct_option
    store.save_grade(
        execution_id,
        grader_version=GRADER_VERSION,
        correct_option=correct_option,
        returned_option=returned_option,
        is_correct=is_correct,
    )
    return Grade(execution_id, returned_option, correct_option, is_correct)

