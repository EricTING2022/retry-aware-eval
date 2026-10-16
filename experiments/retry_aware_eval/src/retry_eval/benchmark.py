"""MathQA benchmark records kept separate from runtime prompts and grading keys."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping

from .domain import Question


OPTION_LABELS = ("a", "b", "c", "d", "e")
_OPTION_MARKER = re.compile(r"(?:^|,\s*)([a-e])\s*\)\s*")


def parse_mathqa_options(raw_options: str) -> dict[str, str]:
    """Parse the official MathQA ``a ) ... , b ) ...`` option string."""
    if raw_options.lstrip().startswith("["):
        parsed_literal = ast.literal_eval(raw_options)
        if not isinstance(parsed_literal, list) or len(parsed_literal) != 5:
            raise ValueError("Stringified MathQA option lists must contain five entries.")
        raw_options = ", ".join(str(value) for value in parsed_literal)
    matches = list(_OPTION_MARKER.finditer(raw_options))
    labels = tuple(match.group(1) for match in matches)
    if labels != OPTION_LABELS:
        raise ValueError(f"Expected MathQA options a-e, found {labels!r}.")
    parsed: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_options)
        value = raw_options[match.end() : end].strip().rstrip(",").strip()
        if not value:
            raise ValueError(f"Option {match.group(1)!r} is empty.")
        parsed[match.group(1)] = value
    return parsed


def format_options(options: Mapping[str, str]) -> str:
    if tuple(options) != OPTION_LABELS:
        raise ValueError("Options must contain ordered labels a-e.")
    return ", ".join(f"{label}) {options[label]}" for label in OPTION_LABELS)


@dataclass(frozen=True)
class BenchmarkItem:
    question_id: str
    source_index: int
    partition: str
    problem: str
    options: Mapping[str, str]
    correct_option: str
    category: str

    def __post_init__(self) -> None:
        if self.source_index < 0:
            raise ValueError("source_index must be nonnegative.")
        if self.partition not in ("development", "reserved_test"):
            raise ValueError("Unknown benchmark partition.")
        if tuple(self.options) != OPTION_LABELS:
            raise ValueError("Benchmark options must contain ordered labels a-e.")
        if self.correct_option not in OPTION_LABELS:
            raise ValueError("correct_option must be one of a-e.")

    def question(self) -> Question:
        """Return the prompt-safe question; the answer key is intentionally omitted."""
        return Question(
            question_id=self.question_id,
            problem=self.problem,
            options=format_options(self.options),
        )


def load_mathqa_benchmark(path: str | Path) -> list[BenchmarkItem]:
    records: list[BenchmarkItem] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            options = raw.get("options")
            if not isinstance(options, dict):
                raise ValueError(f"Line {line_number}: options must be an object.")
            item = BenchmarkItem(
                question_id=str(raw["question_id"]),
                source_index=int(raw["source_index"]),
                partition=str(raw["partition"]),
                problem=str(raw["problem"]),
                options={str(key): str(value) for key, value in options.items()},
                correct_option=str(raw["correct_option"]).lower(),
                category=str(raw.get("category", "")),
            )
            if item.question_id in seen_ids:
                raise ValueError(f"Duplicate question_id {item.question_id!r}.")
            seen_ids.add(item.question_id)
            records.append(item)
    if not records:
        raise ValueError("Benchmark file contains no questions.")
    return records
