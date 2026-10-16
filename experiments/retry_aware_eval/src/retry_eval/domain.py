"""Small domain types shared by orchestration, providers, and storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class Question:
    question_id: str
    problem: str
    options: str

    def __post_init__(self) -> None:
        if not self.question_id or not self.problem or not self.options:
            raise ValueError("Question id, problem, and options are required.")


@dataclass(frozen=True)
class ModelProfile:
    model: str
    solver_parameters: Mapping[str, Any] = field(default_factory=dict)
    verifier_parameters: Mapping[str, Any] = field(default_factory=dict)
    solver_prompt_suffix: str = ""


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    messages: Sequence[Mapping[str, str]]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [dict(message) for message in self.messages],
            **dict(self.parameters),
        }


@dataclass(frozen=True)
class CompletionResponse:
    content: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    raw_response: Mapping[str, Any]
    returned_model: str | None = None
    provider: str | None = None
    response_id: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    reasoning_tokens: int | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    response_body_text: str | None = None


class CompletionClient(Protocol):
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute exactly one model request without automatic retries."""


@dataclass(frozen=True)
class RunOutcome:
    execution_id: str
    final_answer: str
    returned_attempt: int
    stop_reason: str

