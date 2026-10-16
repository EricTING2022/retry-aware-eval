"""Deterministic retry state machine with parameterized model assignments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from time import perf_counter
from typing import Mapping

from .client import OpenRouterResponseError
from .configuration import WorkflowConfiguration
from .domain import (
    CompletionClient,
    CompletionRequest,
    ModelProfile,
    Question,
    RunOutcome,
)
from .prompts import solver_messages, verifier_messages
from .storage import ExperimentStore


class WorkflowRunner:
    def __init__(
        self,
        client: CompletionClient,
        store: ExperimentStore,
        profiles: Mapping[str, ModelProfile],
    ) -> None:
        self.client = client
        self.store = store
        self.profiles = dict(profiles)

    def run_question(
        self,
        run_id: str,
        configuration: WorkflowConfiguration,
        question: Question,
    ) -> RunOutcome:
        missing = set(configuration.solver_models + configuration.verifier_models) - set(
            self.profiles
        )
        if missing:
            raise ValueError(f"Missing model profiles: {sorted(missing)}")

        execution_id = self.store.start_execution(run_id, configuration, question)
        workflow_started = perf_counter()
        try:
            for attempt_index, solver_model in enumerate(
                configuration.solver_models, start=1
            ):
                solver_profile = self.profiles[solver_model]
                solver_request = CompletionRequest(
                    model=solver_model,
                    messages=solver_messages(question, solver_profile),
                    parameters=dict(solver_profile.solver_parameters),
                )
                solver_response, elapsed = self._complete(
                    execution_id, attempt_index, "solver", solver_request
                )
                self.store.record_call(
                    execution_id,
                    attempt_index,
                    "solver",
                    solver_response,
                    elapsed_seconds=elapsed,
                )
                parsed_answer, selected_option, solver_parse_error = parse_solver_answer(
                    solver_response.content
                )
                self.store.record_solver_answer(
                    execution_id,
                    attempt_index,
                    solver_response.content,
                    parsed_answer,
                    selected_option,
                    parse_error=solver_parse_error,
                )

                is_final_attempt = attempt_index == configuration.max_retries + 1
                if is_final_attempt:
                    self.store.complete_attempt(execution_id, attempt_index)
                    self.store.finish_execution(
                        execution_id,
                        returned_attempt=attempt_index,
                        final_answer=solver_response.content,
                        stop_reason="retry_limit_reached",
                        elapsed_seconds=perf_counter() - workflow_started,
                    )
                    return RunOutcome(
                        execution_id=execution_id,
                        final_answer=solver_response.content,
                        returned_attempt=attempt_index,
                        stop_reason="retry_limit_reached",
                    )

                verifier_model = configuration.verifier_models[attempt_index - 1]
                verifier_profile = self.profiles[verifier_model]
                verifier_request = CompletionRequest(
                    model=verifier_model,
                    messages=verifier_messages(question, solver_response.content),
                    parameters=dict(verifier_profile.verifier_parameters),
                )
                verifier_response, elapsed = self._complete(
                    execution_id, attempt_index, "verifier", verifier_request
                )
                accepted, parse_error = parse_verifier_verdict(
                    verifier_response.content
                )
                self.store.record_call(
                    execution_id,
                    attempt_index,
                    "verifier",
                    verifier_response,
                    elapsed_seconds=elapsed,
                    accepted=accepted,
                    parse_error=parse_error,
                )
                self.store.record_verifier_result(
                    execution_id, attempt_index, accepted, parse_error
                )
                self.store.complete_attempt(execution_id, attempt_index)

                if accepted is True:
                    self.store.finish_execution(
                        execution_id,
                        returned_attempt=attempt_index,
                        final_answer=solver_response.content,
                        stop_reason="verifier_accepted",
                        elapsed_seconds=perf_counter() - workflow_started,
                    )
                    return RunOutcome(
                        execution_id=execution_id,
                        final_answer=solver_response.content,
                        returned_attempt=attempt_index,
                        stop_reason="verifier_accepted",
                    )
        except BaseException as error:
            self.store.fail_execution(
                execution_id,
                f"{type(error).__name__}: {error}",
                elapsed_seconds=perf_counter() - workflow_started,
            )
            raise

        raise AssertionError("The bounded workflow must return from an attempt.")

    def _complete(
        self,
        execution_id: str,
        attempt_index: int,
        role: str,
        request: CompletionRequest,
    ):
        started_at_utc = datetime.now(timezone.utc).isoformat()
        self.store.start_call(
            execution_id,
            attempt_index,
            role,
            request,
            started_at_utc=started_at_utc,
        )
        started = perf_counter()
        try:
            response = self.client.complete(request)
        except BaseException as error:
            elapsed = perf_counter() - started
            self.store.record_call_failure(
                execution_id,
                attempt_index,
                role,
                f"{type(error).__name__}: {error}",
                elapsed_seconds=elapsed,
                response=(
                    error.response if isinstance(error, OpenRouterResponseError) else None
                ),
                raw_response=getattr(error, "raw_response", None),
                response_body_text=getattr(error, "response_body_text", None),
                http_status=getattr(error, "http_status", None),
                error_type=type(error).__name__,
            )
            raise
        return response, perf_counter() - started


def parse_solver_answer(
    text: str,
) -> tuple[dict[str, str] | None, str | None, str | None]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        return None, None, f"Invalid solver JSON: {error}"
    if not isinstance(value, dict) or set(value) != {"calculation", "option", "value"}:
        return (
            None,
            None,
            "Solver output must contain exactly calculation, option, and value.",
        )
    if any(not isinstance(value[key], str) for key in value):
        return None, None, "All solver fields must be strings."
    if len(value["calculation"]) > 160 or len(value["value"]) > 120:
        return None, None, "Solver calculation or value exceeds its length limit."
    if "\n" in value["calculation"] or "\n" in value["value"]:
        return (
            None,
            None,
            "Solver calculation and value must not contain line breaks.",
        )
    if value["option"] not in tuple("abcde"):
        return None, None, "Solver option must be one lowercase letter a-e."
    return value, value["option"], None


def parse_verifier_verdict(text: str) -> tuple[bool | None, str | None]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        return None, f"Invalid verifier JSON: {error}"
    if not isinstance(value, dict) or set(value) != {"accepted"}:
        return None, "Verifier output must contain exactly the accepted field."
    if type(value["accepted"]) is not bool:
        return None, "Verifier accepted value must be Boolean."
    return value["accepted"], None

